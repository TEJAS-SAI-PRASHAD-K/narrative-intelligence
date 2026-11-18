"""Laya on Hindi titles, measured against the adjudicated labels of a Hindi sample.

The values come from ``record_scores`` on disk, written by the ``laya`` scoring stage.
This module runs no model. It reports:

* transfer metrics on the titles: precision, recall and flag rate at a decision
  threshold, and ROC-AUC, each with a bootstrap interval,
* the expected calibration error (ECE) of the calibrated values, with an interval,
* the same numbers reweighted to the corpus by the sample weight, when the sheet has one.

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
from modeling.eval.weighted import ece_interval, fmt, rate_intervals

NOT_A_LAYA_SCORE = "not_a_laya_score"
NOT_HINDI = "not_hindi"
ID_MISSING_FROM_SCORES = "id_missing_from_scores"
LAYA_NOTE = "Laya scores the title, so the labels apply to the title"


@dataclass
class LayaTransferResult:
    model_version: str
    threshold: float | None
    threshold_source: str
    sample_rows: int = 0
    labelled: int = 0
    used: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    computed: bool = False
    refusal: str | None = None
    labeller_note: str | None = None
    #: ``sample`` is unweighted, ``corpus`` is reweighted by sample weight.
    sample: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, Any] | None = None

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
                f"({self.threshold_source})"
            )
            for title, block in (
                ("sample (unweighted)", self.sample),
                ("reweighted to the corpus", self.corpus),
            ):
                if block is None:
                    lines.append(
                        "  reweighted to the corpus: not measured, the sheet has no weights"
                    )
                    continue
                lines.append(f"  {title}, 95% bootstrap intervals:")
                for name in ("positive_rate", "flag_rate", "precision", "recall", "roc_auc", "ece"):
                    if name in block:
                        lines.append(f"    {name.replace('_', ' ')} {fmt(block[name])}")
                    elif name in ("precision", "recall", "flag_rate", "positive_rate"):
                        lines.append(
                            f"    {name.replace('_', ' ')} not measured, no decision threshold"
                        )
        if self.labeller_note:
            lines.insert(1, f"  {self.labeller_note}")
            lines.append(f"  {self.labeller_note}")
        return lines


def evaluate_laya_transfer(
    sheet: pd.DataFrame,
    scores: pd.DataFrame | None,
    *,
    model_version: str,
    threshold: float | None,
    threshold_source: str = "laya.decision_threshold",
    min_labelled: int | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    labeller_note: str | None = None,
    bins: int = 10,
) -> LayaTransferResult:
    """Metrics over Hindi rows that have an adjudicated label and a Laya value.

    ``sheet`` is a ``ResolvedSheet.frame``. ``scores`` is ``record_scores``. A value counts
    only when its ``model_versions`` map has a ``laya`` stamp of the configured version.
    """
    from modeling.io import _as_version_dict

    min_labelled = MIN_LABELLED_ROWS if min_labelled is None else min_labelled
    out = LayaTransferResult(
        model_version,
        threshold,
        threshold_source,
        sample_rows=len(sheet),
        labeller_note=labeller_note,
    )
    parsed = [parse_label(v) for v in sheet["label"]]
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
    rows = sheet[mask].copy()
    rows["y"] = [p for p, keep in zip(parsed, mask, strict=True) if keep]
    reasons: Counter[str] = Counter()
    keep_idx = []
    for idx, row in rows.iterrows():
        rid = str(row["id"])
        if str(row.get("lang", "hi")).lower() != "hi":
            reasons[NOT_HINDI] += 1
        elif rid in value:
            keep_idx.append(idx)
        else:
            reasons[reason_of.get(rid, ID_MISSING_FROM_SCORES)] += 1
    out.excluded = dict(reasons)
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

    def block(w: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {}
        result["ece"] = ece_interval(y, prob, w, n_bootstrap=n_bootstrap, seed=seed, bins=bins)
        flag = (prob >= (threshold if threshold is not None else 2.0)).astype(int)
        rates = rate_intervals(y, flag, w, n_bootstrap=n_bootstrap, seed=seed)
        if threshold is None:
            # With no threshold nothing is flagged, so only the positive rate is real.
            rates = {"positive_rate": rates["positive_rate"]}
        result.update(rates)
        return result

    out.sample = block(np.ones(len(y)))
    out.sample["roc_auc"] = bootstrap_metric(
        y, prob, roc_auc_score, n_bootstrap=n_bootstrap, seed=seed
    )
    if "weight" in used and used["weight"].notna().all():
        out.corpus = block(used["weight"].to_numpy(dtype=float))
    out.computed = True
    return out
