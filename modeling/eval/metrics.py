"""Metrics, with confidence intervals attached.

**Accuracy is banned from the report.** On a corpus where 8% of records are
misinformation-like, a model that predicts "not misinformation" for everything
scores 92% and is worthless. Every classification report here carries precision,
recall and F1 per class and macro-averaged, plus ROC-AUC and PR-AUC.

**PR-AUC matters more than ROC-AUC here.** ROC-AUC is computed against the true
negative rate, and when negatives outnumber positives ten to one, a large
absolute number of false positives barely moves it. PR-AUC is computed against
precision, so it degrades exactly when the model starts crying wolf -- which is
the failure mode this product cannot afford.

**Every number gets a 95% bootstrap confidence interval.** A 2-point F1
difference on a 500-row test set is noise, and reporting it as an improvement is
how a project talks itself into a worse model. The interval makes that visible
without anyone having to remember to be sceptical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

log = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP = 1000


@dataclass
class Interval:
    """A point estimate with a bootstrap interval. Renders as "0.78 ±0.04"."""

    value: float
    low: float
    high: float
    n_bootstrap: int = DEFAULT_BOOTSTRAP

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "n_bootstrap": self.n_bootstrap,
        }

    def separated_from(self, other: Interval) -> bool:
        """Whether two intervals fail to overlap.

        A crude but honest test for "is this difference real". Non-overlapping
        intervals are sufficient for a difference; overlapping ones are *not*
        sufficient for no difference, and this method is named to avoid
        implying otherwise.
        """
        return self.high < other.low or other.high < self.low


@dataclass
class ClassificationReport:
    """Everything a reader needs to judge one model on one test set."""

    module: str
    split_description: str
    n_test: int
    class_names: list[str]
    per_class: dict[str, dict[str, float]]
    macro_f1: Interval
    roc_auc: Interval | None
    pr_auc: Interval | None
    brier: float | None
    confusion: list[list[int]]
    positive_rate: float
    #: Set when the numbers come from fixture data rather than a real benchmark.
    is_demo: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def headline(self) -> str:
        """The one line that must never appear without its split strategy."""
        parts = [f"macro-F1 {self.macro_f1}"]
        if self.pr_auc is not None:
            parts.append(f"PR-AUC {self.pr_auc}")
        parts.append(self.split_description)
        line = ", ".join(parts)
        return ("[DEMO FIXTURE -- NOT A RESULT] " + line) if self.is_demo else line

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "split": self.split_description,
            "n_test": self.n_test,
            "is_demo": self.is_demo,
            "positive_rate": round(self.positive_rate, 4),
            "macro_f1": self.macro_f1.as_dict(),
            "roc_auc": self.roc_auc.as_dict() if self.roc_auc else None,
            "pr_auc": self.pr_auc.as_dict() if self.pr_auc else None,
            "brier": round(self.brier, 4) if self.brier is not None else None,
            "per_class": self.per_class,
            "confusion": self.confusion,
            **self.extra,
        }


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap over resampled test rows.

    Resamples *rows*, not groups. That is the right unit here because the split
    already guaranteed group disjointness -- the remaining uncertainty is over
    which rows happened to land in this test set.
    """
    point = float(metric_fn(y_true, y_score))
    n = len(y_true)
    if n < 10:
        # Too few rows for a meaningful interval. Say so rather than emitting a
        # confident-looking [point, point].
        return Interval(point, float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, size=n)
        resampled_true = y_true[index]
        # A resample with one class present makes AUC undefined. Skipping those
        # draws biases the interval slightly narrow; the alternative is a
        # crash or a fabricated 0.5.
        if len(np.unique(resampled_true)) < 2:
            continue
        try:
            samples.append(float(metric_fn(resampled_true, y_score[index])))
        except ValueError:
            continue
    if len(samples) < n_bootstrap // 10:
        return Interval(point, float("nan"), float("nan"), len(samples))
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(low), float(high), len(samples))


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    module: str,
    split_description: str,
    y_score: np.ndarray | None = None,
    class_names: list[str] | None = None,
    seed: int = 0,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    is_demo: bool = False,
) -> ClassificationReport:
    """Build the full report. ``y_score`` is the positive-class probability."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    names = class_names or [str(label) for label in labels]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        names[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }

    macro = bootstrap_metric(
        y_true,
        y_pred,
        lambda t, p: f1_score(t, p, average="macro", zero_division=0),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    roc = pr = None
    brier = None
    binary = len(labels) == 2
    if y_score is not None and binary:
        y_score = np.asarray(y_score, dtype=float)
        roc = bootstrap_metric(y_true, y_score, roc_auc_score, n_bootstrap=n_bootstrap, seed=seed)
        pr = bootstrap_metric(
            y_true, y_score, average_precision_score, n_bootstrap=n_bootstrap, seed=seed
        )
        if y_score.min() >= 0 and y_score.max() <= 1:
            brier = float(brier_score_loss(y_true, y_score))

    return ClassificationReport(
        module=module,
        split_description=split_description,
        n_test=len(y_true),
        class_names=names,
        per_class=per_class,
        macro_f1=macro,
        roc_auc=roc,
        pr_auc=pr,
        brier=brier,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        positive_rate=float(np.mean(y_true == labels[-1])) if len(y_true) else 0.0,
        is_demo=is_demo,
    )


def fold_summary(reports: list[ClassificationReport]) -> dict[str, Any]:
    """Mean +/- std across CV folds.

    Reported alongside the per-fold numbers, never instead of them: a mean that
    hides one catastrophic fold is worse than no summary at all.
    """
    if not reports:
        return {}
    macro = np.array([r.macro_f1.value for r in reports])
    out: dict[str, Any] = {
        "n_folds": len(reports),
        "macro_f1_mean": round(float(macro.mean()), 4),
        "macro_f1_std": round(float(macro.std(ddof=1)) if len(macro) > 1 else 0.0, 4),
        "macro_f1_per_fold": [round(float(v), 4) for v in macro],
    }
    prs = [r.pr_auc.value for r in reports if r.pr_auc is not None]
    if prs:
        out["pr_auc_mean"] = round(float(np.mean(prs)), 4)
        out["pr_auc_std"] = round(float(np.std(prs, ddof=1)) if len(prs) > 1 else 0.0, 4)
        out["pr_auc_per_fold"] = [round(float(v), 4) for v in prs]
    worst = min(reports, key=lambda r: r.macro_f1.value)
    out["worst_fold_macro_f1"] = round(worst.macro_f1.value, 4)
    return out


def threshold_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float
) -> dict[str, float]:
    """The lowest threshold that reaches a precision target, and its recall.

    Not 0.5. In this product a false "bot" or "misinformation" flag is costlier
    than a miss -- it is an accusation about a person or a claim -- so the
    operating point is chosen from the precision-recall curve at a stated
    precision, and the recall that costs is reported alongside it.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more entry than thresholds.
    viable = np.flatnonzero(precision[:-1] >= target_precision)
    if not len(viable):
        best = int(np.argmax(precision[:-1]))
        log.warning(
            "no threshold reaches precision %.2f; best achievable is %.3f at recall %.3f",
            target_precision,
            precision[best],
            recall[best],
        )
        return {
            "threshold": float(thresholds[best]),
            "precision": float(precision[best]),
            "recall": float(recall[best]),
            "target_met": 0.0,
        }
    # Among thresholds meeting the target, take the one with the best recall.
    chosen = int(viable[np.argmax(recall[viable])])
    return {
        "threshold": float(thresholds[chosen]),
        "precision": float(precision[chosen]),
        "recall": float(recall[chosen]),
        "target_met": 1.0,
    }


def score_distribution(values: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Histogram plus quantiles. The honest report for an unsupervised score."""
    values = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if not len(values):
        return {"n": 0}
    counts, edges = np.histogram(
        values, bins=bins, range=(float(values.min()), float(values.max()))
    )
    return {
        "n": int(len(values)),
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "quantiles": {
            str(q): round(float(np.quantile(values, q / 100)), 4)
            for q in (1, 5, 25, 50, 75, 95, 99)
        },
        "histogram": {"counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges]},
    }


def group_slice_report(
    y_score: np.ndarray, groups: np.ndarray, *, min_size: int = 20
) -> dict[str, Any]:
    """Score distribution per slice -- the fairness check.

    Reports the distribution of a score across corpus slices (language groups,
    topical slices). You do not need to fix what this finds; you do need to
    report it. Slices below ``min_size`` are reported as too small rather than
    given a number nobody should trust.
    """
    out: dict[str, Any] = {}
    groups = np.asarray(groups, dtype=object)
    for value in sorted({g for g in groups.tolist() if g is not None}, key=str):
        mask = groups == value
        subset = np.asarray(y_score, dtype=float)[mask]
        subset = subset[~np.isnan(subset)]
        if len(subset) < min_size:
            out[str(value)] = {"n": int(len(subset)), "note": "too small to report"}
            continue
        out[str(value)] = {
            "n": int(len(subset)),
            "mean": round(float(subset.mean()), 4),
            "median": round(float(np.median(subset)), 4),
            "p90": round(float(np.quantile(subset, 0.9)), 4),
        }
    return out
