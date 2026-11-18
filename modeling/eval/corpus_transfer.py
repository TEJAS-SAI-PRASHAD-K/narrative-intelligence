"""Benchmark-to-corpus transfer: sample real records, then score against human labels.

Two halves that must agree on one definition of "scorable":

* ``modeling.eval.labelling.build_weighted_sample`` draws the records a person will
  label. It never writes a model score into the sheet, because a labeller who sees a
  score anchors on it.
* ``evaluate_transfer`` joins the labelled sheet to the scores on disk and computes
  metrics over the rows that have both a human label and a usable score.

Every row that does not reach the metrics is counted under a reason code. A null
score is not a negative prediction, and a blank label is not a 0.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.eval.metrics import classification_report
from modeling.eval.weighted import bootstrap_statistics, fmt, rate_intervals

log = logging.getLogger(__name__)

#: Fewest hand-labelled rows (and fewest labelled rows that also carry a usable
#: score) before transfer metrics are computed. With fewer, a bootstrap interval
#: on ROC-AUC spans most of [0, 1] and the point estimate invites a conclusion the
#: data cannot carry. Half of the default 100-row sample.
MIN_LABELLED_ROWS = 50

#: Minimum text length the misinfo stage scores (``modeling/scoring.py`` filters on
#: the same 20). Kept in step by hand; a record shorter than this is never scored.
MIN_TEXT_CHARS = 20

#: Languages a sample may be restricted to.
SAMPLE_LANGUAGES = ("en", "hi")

# Reason codes. Each is counted, never dropped silently.
LANGUAGE_NOT_SCORED = "language_not_scored"
TEXT_TOO_SHORT = "text_too_short"
NO_SCORE = "no_score"
STALE_MODEL_VERSION = "stale_model_version"
UNLABELLED = "unlabelled"
INVALID_LABEL = "invalid_label"
NOT_IN_CORPUS_SCORES = "id_missing_from_scores"

CLASS_NAMES = ["not_misinfo", "misinfo"]


def unscorable_reason(
    lang: Any, text: Any, language_allowed: Callable[[str | None], bool]
) -> str | None:
    """Why the misinfo stage would skip this record, or ``None`` if it scores it."""
    if not language_allowed(None if _is_null(lang) else str(lang)):
        return LANGUAGE_NOT_SCORED
    # The misinfo stage measures the stripped text, so the sampler must too.
    if len("" if _is_null(text) else str(text).strip()) < MIN_TEXT_CHARS:
        return TEXT_TOO_SHORT
    return None


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value) or value is pd.NA


def read_label_sheet(path: Path) -> pd.DataFrame:
    """Read the sheet with every cell as text so a blank stays blank, never 0 or NaN-as-0."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def parse_label(cell: Any) -> int | None | str:
    """``0`` or ``1`` for a human label, ``None`` for blank, ``"invalid"`` otherwise."""
    text = "" if _is_null(cell) else str(cell).strip()
    if text == "":
        return None
    if text in ("0", "0.0"):
        return 0
    if text in ("1", "1.0"):
        return 1
    return "invalid"


@dataclass
class TransferResult:
    sample_rows: int = 0
    labelled: int = 0
    unlabelled: int = 0
    invalid_labels: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    used: int = 0
    threshold: float | None = None
    model_version: str | None = None
    computed: bool = False
    refusal: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    #: Set when the labels come from a model that no person reviewed.
    labeller_note: str | None = None
    #: Sample weight statistics, or ``None`` when the sheet has no weight column.
    weighted: dict[str, Any] | None = None
    #: Flag rate on rows that report on a false claim and on the rest.
    reporting: dict[str, Any] | None = None

    def render(self, benchmark_macro_f1: float | None = None) -> list[str]:
        lines = self._render(benchmark_macro_f1)
        if self.labeller_note:
            lines.insert(1, f"  {self.labeller_note}")
            lines.append(f"  {self.labeller_note}")
        return lines

    def _render(self, benchmark_macro_f1: float | None) -> list[str]:
        lines = [
            "corpus transfer",
            f"  sample rows {self.sample_rows}, labelled {self.labelled}, "
            f"not yet labelled {self.unlabelled}, invalid labels {self.invalid_labels}",
            f"  labelled rows used {self.used}, excluded "
            + (
                ", ".join(f"{k}={v}" for k, v in sorted(self.excluded.items()))
                if self.excluded
                else "none"
            ),
        ]
        if not self.computed:
            lines.append(f"  REFUSED: {self.refusal}")
            return lines
        m = self.metrics
        lines += [
            f"  model {self.model_version}, threshold {self.threshold}, "
            f"labelled-positive rate {m['positive_rate']:.3f}, flag rate {m['flag_rate']:.3f}",
            f"  macro-F1 {m['macro_f1']}  ROC-AUC {m['roc_auc']}  PR-AUC {m['pr_auc']}",
            f"  misinfo class: precision {m['precision']:.3f}, recall {m['recall']:.3f}, "
            f"F1 {m['f1']:.3f}",
        ]
        if benchmark_macro_f1 is not None:
            gap = benchmark_macro_f1 - m["macro_f1_value"]
            lines.append(
                f"  benchmark macro-F1 {benchmark_macro_f1:.3f}, corpus macro-F1 "
                f"{m['macro_f1_value']:.3f}, gap {gap:+.3f}"
            )
        if self.weighted is not None:
            w = self.weighted
            lines.append("  reweighted to the corpus by sample weight, 95% bootstrap intervals:")
            for name in ("positive_rate", "flag_rate", "precision", "recall"):
                lines.append(f"    {name.replace('_', ' ')} {fmt(w.get(name))}")
        if self.reporting is not None:
            r = self.reporting
            lines.append(
                f"  flag rate on rows that report on a false claim ({r['n_reporting']} rows) "
                f"{fmt(r['reporting'])}, on the other rows ({r['n_other']} rows) "
                f"{fmt(r['other'])}"
            )
        return lines


def evaluate_transfer(
    sheet: pd.DataFrame,
    scores: pd.DataFrame | None,
    *,
    language_allowed: Callable[[str | None], bool],
    model_version: str,
    threshold: float | None,
    min_labelled: int | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    labeller_note: str | None = None,
) -> TransferResult:
    """Metrics over rows with a human label and a usable score; counts for the rest.

    ``scores`` is the ``record_scores`` table (``record_id``, ``misinfo_prob``,
    optionally ``model_versions``). A score stamped by a different model version
    is excluded as stale. Labels come only from the sheet.
    """
    from modeling.io import _as_version_dict

    min_labelled = MIN_LABELLED_ROWS if min_labelled is None else min_labelled

    out = TransferResult(
        sample_rows=len(sheet),
        threshold=threshold,
        model_version=model_version,
        labeller_note=labeller_note,
    )
    parsed = [parse_label(v) for v in sheet["label"]] if "label" in sheet else [None] * len(sheet)
    out.unlabelled = sum(p is None for p in parsed)
    out.invalid_labels = sum(p == "invalid" for p in parsed)
    labelled_mask = np.array([p in (0, 1) and p is not None for p in parsed], dtype=bool)
    out.labelled = int(labelled_mask.sum())

    if out.labelled < min_labelled:
        out.refusal = (
            f"{out.labelled} labelled rows, need at least {min_labelled} "
            f"(MIN_LABELLED_ROWS). Blank labels are not 0 and are not counted."
        )
        return out
    if threshold is None:
        out.refusal = f"no decision_threshold configured for misinfo {model_version}"
        return out

    rows = sheet[labelled_mask].copy()
    rows["y_true"] = [p for p, keep in zip(parsed, labelled_mask, strict=True) if keep]

    score_by_id: dict[str, float] = {}
    stale: set[str] = set()
    if scores is not None and len(scores) and "misinfo_prob" in scores.columns:
        for rec in scores.to_dict(orient="records"):
            rid = str(rec["record_id"])
            versions = _as_version_dict(rec.get("model_versions"))
            stamp = versions.get("misinfo")
            if stamp is not None and stamp.split("+")[0] != model_version:
                stale.add(rid)
                continue
            value = rec["misinfo_prob"]
            if not _is_null(value):
                score_by_id[rid] = float(value)

    reasons: Counter[str] = Counter()
    keep_rows = []
    for idx, row in rows.iterrows():
        rid = str(row["id"])
        if rid in score_by_id:
            keep_rows.append(idx)
            continue
        reason = unscorable_reason(row.get("lang"), row.get("text"), language_allowed)
        if reason is None:
            reason = STALE_MODEL_VERSION if rid in stale else NO_SCORE
        reasons[reason] += 1
    out.excluded = dict(reasons)

    used = rows.loc[keep_rows]
    out.used = len(used)
    y_true = used["y_true"].to_numpy(dtype=int)
    if out.used < min_labelled:
        out.refusal = (
            f"{out.used} labelled rows have a usable score, need at least {min_labelled} "
            f"(MIN_LABELLED_ROWS). Excluded rows are listed above."
        )
        return out
    if len(set(y_true.tolist())) < 2:
        out.refusal = "the usable labelled rows hold one class only; AUC is undefined"
        return out

    y_score = np.array([score_by_id[str(i)] for i in used["id"]], dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        module="misinfo",
        split_description=f"hand-labelled corpus sample, n={out.used}, no training overlap",
        y_score=y_score,
        class_names=CLASS_NAMES,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    positive = report.per_class["misinfo"]
    if "weight" in used and used["weight"].notna().all():
        weights = used["weight"].to_numpy(dtype=float)
        out.weighted = rate_intervals(
            y_true, y_pred, weights, n_bootstrap=n_bootstrap, seed=seed
        )
    if "reports" in used:
        out.reporting = _reporting_flag_rates(
            used["reports"].to_numpy(dtype=float), y_pred, y_true, n_bootstrap, seed
        )
    out.metrics = {
        "macro_f1": report.macro_f1,
        "macro_f1_value": report.macro_f1.value,
        "roc_auc": report.roc_auc,
        "pr_auc": report.pr_auc,
        "brier": report.brier,
        "precision": positive["precision"],
        "recall": positive["recall"],
        "f1": positive["f1"],
        "positive_rate": float(y_true.mean()),
        "flag_rate": float(y_pred.mean()),
    }
    out.computed = True
    return out


def _reporting_flag_rates(
    reports: np.ndarray, flag: np.ndarray, y_true: np.ndarray, n_bootstrap: int, seed: int
) -> dict[str, Any] | None:
    """Flag rate on rows that report on a false claim, against the rest, unweighted.

    Rows with no adjudicated ``reports_on_false_claim`` are left out of both groups.
    """
    known = ~np.isnan(reports)
    if not known.any():
        return None
    r, f, y = reports[known].astype(int), flag[known], y_true[known]
    ones = np.ones(len(r))

    def rate(group: int):
        def fn(idx):
            mask = r[idx] == group
            return float(f[idx][mask].mean()) if mask.any() else None

        return fn

    both = bootstrap_statistics(
        y, ones, {"reporting": rate(1), "other": rate(0)}, n_bootstrap=n_bootstrap, seed=seed
    )
    return {
        "n_reporting": int((r == 1).sum()),
        "n_other": int((r == 0).sum()),
        **both,
    }
