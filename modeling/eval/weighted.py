"""Weighted classification rates and calibration error, each with a bootstrap interval.

The labelling sample is stratified, so a row stands for ``weight`` corpus rows. A
metric that describes the corpus uses those weights. A metric on the raw sample uses
weights of 1. Both go through the same functions so the two cannot drift apart.

A statistic with an empty denominator is ``None``, never 0. A precision over zero flagged
rows is not measured, and 0.0 would claim a measured failure.
"""

from __future__ import annotations

import numpy as np

from modeling.eval.metrics import Interval

DEFAULT_BINS = 10


def _ratio(num: float, den: float) -> float | None:
    return None if den <= 0 else float(num / den)


def rates(y_true: np.ndarray, flag: np.ndarray, w: np.ndarray) -> dict[str, float | None]:
    """Weighted positive rate, flag rate, precision and recall."""
    total = float(w.sum())
    tp = float(w[(y_true == 1) & (flag == 1)].sum())
    return {
        "positive_rate": _ratio(float(w[y_true == 1].sum()), total),
        "flag_rate": _ratio(float(w[flag == 1].sum()), total),
        "precision": _ratio(tp, float(w[flag == 1].sum())),
        "recall": _ratio(tp, float(w[y_true == 1].sum())),
    }


def expected_calibration_error(
    y_true: np.ndarray, prob: np.ndarray, w: np.ndarray, bins: int = DEFAULT_BINS
) -> float | None:
    """Weighted ECE over equal-width bins of the predicted probability."""
    total = float(w.sum())
    if total <= 0 or not len(y_true):
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        weight = float(w[mask].sum())
        if weight <= 0:
            continue
        gap = abs(
            float(np.average(y_true[mask], weights=w[mask]))
            - float(np.average(prob[mask], weights=w[mask]))
        )
        ece += weight / total * gap
    return ece


def _interval(
    point: float | None, samples: list[float], n_bootstrap: int, alpha: float = 0.05
) -> Interval | None:
    """``None`` when the statistic is undefined. NaN bounds when too few draws are defined."""
    if point is None:
        return None
    if len(samples) < max(1, n_bootstrap // 10):
        return Interval(point, float("nan"), float("nan"), len(samples))
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(low), float(high), len(samples))


def bootstrap_statistics(
    y_true: np.ndarray,
    w: np.ndarray,
    stats: dict[str, object],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Interval | None]:
    """Percentile bootstrap over rows for each named statistic.

    ``stats`` maps a name to ``f(index) -> float | None`` over a row index array. A draw
    where a statistic is undefined is skipped for that statistic only. Every statistic
    sees the same draws.
    """
    n = len(y_true)
    points = {name: fn(np.arange(n)) for name, fn in stats.items()}
    if n < 10:
        return {
            name: None if p is None else Interval(p, float("nan"), float("nan"), 0)
            for name, p in points.items()
        }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in stats}
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, size=n)
        for name, fn in stats.items():
            value = fn(index)
            if value is not None:
                samples[name].append(value)
    return {name: _interval(points[name], samples[name], n_bootstrap) for name in stats}


def rate_intervals(
    y_true: np.ndarray,
    flag: np.ndarray,
    w: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Interval | None]:
    """Intervals for ``rates``."""

    def pick(name: str):
        return lambda idx: rates(y_true[idx], flag[idx], w[idx])[name]

    names = ("positive_rate", "flag_rate", "precision", "recall")
    return bootstrap_statistics(
        y_true, w, {n: pick(n) for n in names}, n_bootstrap=n_bootstrap, seed=seed
    )


def ece_interval(
    y_true: np.ndarray,
    prob: np.ndarray,
    w: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    bins: int = DEFAULT_BINS,
) -> Interval | None:
    stats = {"ece": lambda idx: expected_calibration_error(y_true[idx], prob[idx], w[idx], bins)}
    return bootstrap_statistics(y_true, w, stats, n_bootstrap=n_bootstrap, seed=seed)["ece"]


def fmt(value: Interval | float | None) -> str:
    """Render a statistic. ``None`` is written out, never as 0."""
    if value is None:
        return "not measured"
    return str(value) if isinstance(value, Interval) else f"{value:.3f}"
