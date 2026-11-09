"""Benchmark-to-corpus transfer: sample real records, then score against human labels.

Two halves that must agree on one definition of "scorable":

* ``build_sample`` draws the records a person will label. It never writes a model
  score into the sheet, because a labeller who sees a score anchors on it.
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

SAMPLE_COLUMNS = ("id", "source", "lang", "text", "label")
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


@dataclass
class SampleResult:
    frame: pd.DataFrame
    pool: int
    excluded: dict[str, int] = field(default_factory=dict)


def build_sample(
    records: pd.DataFrame,
    *,
    n: int,
    language_allowed: Callable[[str | None], bool],
    lang: str | None = None,
    seed: int = 0,
) -> SampleResult:
    """Stratified-by-source sample of records the misinfo stage can score.

    ``lang`` restricts the pool to one language; records with an unknown language
    are not in a restricted pool. Records the stage would skip are counted in
    ``excluded`` by reason instead of being sampled, so the labeller's effort is
    never spent on a row that cannot be scored.
    """
    if lang is not None and lang not in SAMPLE_LANGUAGES:
        raise ValueError(f"lang must be one of {SAMPLE_LANGUAGES}, got {lang!r}")
    pool = records
    if lang is not None:
        pool = pool[pool["lang"].map(lambda v: not _is_null(v) and str(v).lower() == lang)]

    reasons = [
        unscorable_reason(row_lang, row_text, language_allowed)
        for row_lang, row_text in zip(pool["lang"], pool["text"], strict=True)
    ]
    excluded = dict(Counter(r for r in reasons if r))
    scorable = pool[[r is None for r in reasons]]
    if not len(scorable):
        return SampleResult(scorable.iloc[0:0], len(pool), excluded)

    # The transfer gap is a per-platform question, so draw a quota from each source
    # and top up from what is left, rather than let one platform fill the sample.
    sources = sorted(scorable["source"].unique())
    quota = max(1, n // len(sources))
    parts = [
        scorable[scorable["source"] == s].sample(
            n=min(quota, int((scorable["source"] == s).sum())), random_state=seed
        )
        for s in sources
    ]
    chosen = pd.concat(parts)
    if len(chosen) > n:
        chosen = chosen.sample(n=n, random_state=seed)
    elif len(chosen) < n:
        rest = scorable.drop(index=chosen.index)
        extra = rest.sample(n=min(n - len(chosen), len(rest)), random_state=seed)
        chosen = pd.concat([chosen, extra])
    chosen = chosen.sort_values(["source", "id"]).reset_index(drop=True)
    chosen["label"] = ""  # 1 = misinformation-like, 0 = not; filled in by a person
    columns = [c for c in SAMPLE_COLUMNS if c in chosen]
    return SampleResult(chosen[columns], len(pool), excluded)


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

    def render(self, benchmark_macro_f1: float | None = None) -> list[str]:
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
) -> TransferResult:
    """Metrics over rows with a human label and a usable score; counts for the rest.

    ``scores`` is the ``record_scores`` table (``record_id``, ``misinfo_prob``,
    optionally ``model_versions``). A score stamped by a different model version
    is excluded as stale. Labels come only from the sheet.
    """
    from modeling.io import _as_version_dict

    min_labelled = MIN_LABELLED_ROWS if min_labelled is None else min_labelled

    out = TransferResult(
        sample_rows=len(sheet), threshold=threshold, model_version=model_version
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
