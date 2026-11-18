"""Laya on Hindi titles, measured against the adjudicated labels of a Hindi sample.

The values come from ``record_scores`` on disk, written by the ``laya`` scoring stage.
This module runs no model. It reports:

* transfer metrics on the titles: precision, recall and flag rate at a decision
  threshold, and ROC-AUC, each with a bootstrap interval,
* the expected calibration error (ECE) of the calibrated values, with an interval,
* the same numbers reweighted to the corpus by stratum, when the sheet has a design file.

``laya.decision_threshold`` is null until a person sets it from a precision-recall curve.
With no threshold, the thresholded metrics are not measured. The caller can pass a
threshold, and the output then says it is not the configured one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from modeling.eval.corpus_transfer import (
    MIN_LABELLED_ROWS,
    NO_SCORE,
    STALE_MODEL_VERSION,
    _is_null,
    parse_label,
)
from modeling.eval.metrics import bootstrap_metric
from modeling.eval.weighted import (
    DEFAULT_BINS,
    DEFAULT_SEED,
    ECE_MIN_N,
    StratumWeights,
    ece_interval,
    fmt,
    rate_intervals,
    render_reweighted,
    stratum_weights,
)

NOT_A_LAYA_SCORE = "not_a_laya_score"
NOT_HINDI = "not_hindi"
ID_MISSING_FROM_SCORES = "id_missing_from_scores"
LAYA_NOTE = "Laya scores the title, so the labels apply to the title"
ECE_DEFINITION = (
    "sum over bins of (bin share of weight) * |observed positive rate - mean calibrated value|, "
    "on the positive-class value, with equal-width bins"
)
RATE_NAMES = ("positive_rate", "flag_rate", "precision", "recall")


@dataclass
class LayaTransferResult:
    model_version: str
    threshold: float | None
    threshold_source: str
    sample_rows: int = 0
    labelled: int = 0
    used: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    drops_by_stratum: dict[str, dict[str, int]] = field(default_factory=dict)
    computed: bool = False
    refusal: str | None = None
    labeller_note: str | None = None
    seed: int = DEFAULT_SEED
    bins: int = DEFAULT_BINS
    #: ``sample`` is unweighted, ``corpus`` is reweighted by stratum.
    sample: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, Any] | None = None
    strata_info: StratumWeights | None = None
    weighted_reason: str | None = None

    def _block(self, block: dict[str, Any]) -> list[str]:
        lines = []
        for name in ("positive_rate", "flag_rate", "precision", "recall", "roc_auc"):
            if name in block:
                lines.append(f"    {name.replace('_', ' ')} {fmt(block[name])}")
            elif name in RATE_NAMES:
                lines.append(f"    {name.replace('_', ' ')} not measured, no decision threshold")
        return lines

    def _ece_lines(self, block: dict[str, Any], n: float, *, effective: bool = False) -> list[str]:
        label = "n_eff" if effective else "n"
        lines = [
            f"    ece {fmt(block['ece'])} ({self.bins} equal-width bins, {label} {n:.0f})",
            f"    ece definition: {ECE_DEFINITION}",
        ]
        if "ece_boot_mean" in block:
            lines.append(
                f"    mean of the bootstrap draws of ece {block['ece_boot_mean']:.3f}. "
                "ECE is biased upward under resampling, so the interval can exclude the point "
                "value."
            )
        if n < ECE_MIN_N:
            lines.append(
                f"    caveat: {label} is below {ECE_MIN_N}, so each bin holds few rows and the "
                "ECE is noisy. Read the interval, not the point."
            )
        return lines

    def render(self) -> list[str]:
        lines = [
            "laya hindi transfer on titles",
            f"  {LAYA_NOTE}",
            f"  sample rows {self.sample_rows}, labelled {self.labelled}, used {self.used}, "
            "excluded "
            + (
                ", ".join(f"{k}={v}" for k, v in sorted(self.excluded.items()))
                if self.excluded
                else "none"
            ),
        ]
        if not self.computed:
            lines.append(f"  REFUSED: {self.refusal}")
        else:
            lines.append(
                f"  model {self.model_version}, threshold {self.threshold} "
                f"({self.threshold_source}), seed {self.seed}"
            )
            lines.append("  sample rows, unweighted, 95% bootstrap intervals:")
            lines += self._block(self.sample)
            lines += self._ece_lines(self.sample, self.used)
            if self.drops_by_stratum:
                lines.append("  rows left out, by stratum code and reason:")
                for code, reasons in sorted(self.drops_by_stratum.items()):
                    lines.append(
                        f"    {code}: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
                    )
            if self.corpus is not None and self.strata_info is not None:
                lines += render_reweighted(
                    {k: v for k, v in self.corpus.items() if k not in ("ece", "ece_boot_mean")},
                    self.strata_info,
                )
                n_eff = self.strata_info.n_eff or 0.0
                lines += self._ece_lines(self.corpus, n_eff, effective=True)
            else:
                lines.append(
                    f"  reweighted to the corpus: not measured, {self.weighted_reason}"
                )
        if self.labeller_note:
            lines.insert(1, f"  {self.labeller_note}")
            lines.append(f"  {self.labeller_note}")
        return lines


def evaluate_laya_transfer(
    sheet: Any,
    scores: pd.DataFrame | None,
    *,
    model_version: str,
    threshold: float | None,
    threshold_source: str = "laya.decision_threshold",
    min_labelled: int | None = None,
    n_bootstrap: int = 1000,
    seed: int = DEFAULT_SEED,
    bins: int = DEFAULT_BINS,
) -> LayaTransferResult:
    """Metrics over Hindi rows that have an adjudicated label and a Laya value.

    ``sheet`` is a ``modeling.eval.labelling.ResolvedSheet``. Its refusal, for labels that a
    model made and no person reviewed, becomes the refusal here. ``scores`` is
    ``record_scores``. A value counts only when its ``model_versions`` map has a ``laya``
    stamp of the configured version.
    """
    from modeling.eval.labelling import ResolvedSheet
    from modeling.io import _as_version_dict

    if not isinstance(sheet, ResolvedSheet):
        raise TypeError("evaluate_laya_transfer needs a ResolvedSheet: call resolve_sheet first")
    frame = sheet.frame
    min_labelled = MIN_LABELLED_ROWS if min_labelled is None else min_labelled
    out = LayaTransferResult(
        model_version,
        threshold,
        threshold_source,
        sample_rows=len(frame),
        labeller_note=sheet.labeller_note,
        seed=seed,
        bins=bins,
    )
    if sheet.refusal:
        out.refusal = sheet.refusal
        return out
    parsed = [parse_label(v) for v in frame["label"]]
    mask = np.array([p in (0, 1) and p is not None for p in parsed], dtype=bool)
    out.labelled = int(mask.sum())
    if out.labelled < min_labelled:
        out.refusal = (
            f"{out.labelled} labelled rows, need at least {min_labelled} (MIN_LABELLED_ROWS)"
        )
        return out

    value: dict[str, float] = {}
    reason_of: dict[str, str] = {}
    if scores is not None and len(scores) and "misinfo_prob" in scores.columns:
        for rec in scores.to_dict(orient="records"):
            rid = str(rec["record_id"])
            stamp = _as_version_dict(rec.get("model_versions")).get("laya")
            if stamp is None:
                reason_of[rid] = NOT_A_LAYA_SCORE
            elif stamp.split("+")[0] != model_version:
                reason_of[rid] = STALE_MODEL_VERSION
            elif _is_null(rec["misinfo_prob"]):
                reason_of[rid] = NO_SCORE
            else:
                value[rid] = float(rec["misinfo_prob"])
    rows = frame[mask].copy()
    rows["y"] = [p for p, keep in zip(parsed, mask, strict=True) if keep]
    reasons: Counter[str] = Counter()
    row_reason: dict[Any, str] = {}
    keep_idx = []
    for idx, row in rows.iterrows():
        rid = str(row["id"])
        if str(row.get("lang", "hi")).lower() != "hi":
            reason = NOT_HINDI
        elif rid in value:
            keep_idx.append(idx)
            continue
        else:
            reason = reason_of.get(rid, ID_MISSING_FROM_SCORES)
        reasons[reason] += 1
        row_reason[idx] = reason
    out.excluded = dict(reasons)
    if "stratum" in frame:
        drops: dict[str, Counter[str]] = {}
        for idx, reason in row_reason.items():
            code = frame.at[idx, "stratum"]
            if isinstance(code, str):
                drops.setdefault(code, Counter())[reason] += 1
        out.drops_by_stratum = {k: dict(v) for k, v in drops.items()}
    used = rows.loc[keep_idx]
    out.used = len(used)
    if out.used < min_labelled:
        out.refusal = f"{out.used} labelled rows have a Laya value, need at least {min_labelled}"
        return out
    y = used["y"].to_numpy(dtype=int)
    if len(set(y.tolist())) < 2:
        out.refusal = "the usable labelled rows hold one class only; AUC is undefined"
        return out
    prob = np.array([value[str(i)] for i in used["id"]], dtype=float)

    def block(w: np.ndarray, strata: np.ndarray | None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        means: dict[str, float] = {}
        result["ece"] = ece_interval(
            y, prob, w, n_bootstrap=n_bootstrap, seed=seed, bins=bins, strata=strata, means=means
        )
        if "ece" in means:
            result["ece_boot_mean"] = means["ece"]
        flag = (prob >= (threshold if threshold is not None else 2.0)).astype(int)
        rates = rate_intervals(y, flag, w, n_bootstrap=n_bootstrap, seed=seed, strata=strata)
        if threshold is None:
            # With no threshold nothing is flagged, so only the positive rate is real.
            rates = {"positive_rate": rates["positive_rate"]}
        result.update(rates)
        return result

    out.sample = block(np.ones(len(y)), None)
    out.sample["roc_auc"] = bootstrap_metric(
        y, prob, roc_auc_score, n_bootstrap=n_bootstrap, seed=seed
    )
    sw = stratum_weights(frame, used) if sheet.has_design else None
    if sw is not None:
        out.strata_info = sw
        out.corpus = block(sw.weights, sw.strata)
    else:
        out.weighted_reason = (
            "a labelled row has no stratum in the design file"
            if sheet.has_design
            else "the sheet has no design file"
        )
    out.computed = True
    return out
