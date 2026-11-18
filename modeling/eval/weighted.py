"""Weighted classification rates and calibration error, each with a bootstrap interval.

The labelling sample is stratified, so a row stands for ``weight`` corpus rows. A
metric that describes the corpus uses those weights. A metric on the raw sample uses
weights of 1. Both go through the same functions so the two cannot drift apart.

A statistic with an empty denominator is ``None``, never 0. A precision over zero flagged
rows is not measured, and 0.0 would claim a measured failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modeling.eval.metrics import Interval

DEFAULT_BINS = 10
#: Seed of every interval in the labelling outputs.
DEFAULT_SEED = 20260813
#: Below this many rows an ECE bin holds too few rows for its gap to mean much.
ECE_MIN_N = 200
#: A stratum with fewer used rows than this gets a loud warning in the output.
MIN_STRATUM_USED = 5


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


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval of a proportion. ``(nan, nan)`` for no rows."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    # The bound at an observed 0 or n is exact, so float noise must not move it.
    low = 0.0 if successes == 0 else max(0.0, float(centre - half))
    high = 1.0 if successes == n else min(1.0, float(centre + half))
    return low, high


def kish_n_eff(w: np.ndarray) -> float | None:
    """Kish effective sample size, (sum w)^2 / sum w^2. ``None`` for no rows."""
    denominator = float((w**2).sum())
    return None if denominator <= 0 else float(w.sum() ** 2 / denominator)


def _draw(rng: np.random.Generator, n: int, groups: list[np.ndarray] | None) -> np.ndarray:
    """Row indices of one bootstrap draw, resampled inside each stratum when given."""
    if groups is None:
        return rng.integers(0, n, size=n)
    return np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])


def bootstrap_statistics(
    y_true: np.ndarray,
    w: np.ndarray,
    stats: dict[str, object],
    *,
    n_bootstrap: int,
    seed: int,
    strata: np.ndarray | None = None,
    means: dict[str, float] | None = None,
) -> dict[str, Interval | None]:
    """Percentile bootstrap over rows for each named statistic.

    ``stats`` maps a name to ``f(index) -> float | None`` over a row index array. A draw
    where a statistic is undefined is skipped for that statistic only. Every statistic
    sees the same draws. With ``strata``, rows are resampled inside each stratum, so
    every draw keeps the stratum sizes that the weights assume. When ``means`` is given, it
    receives the mean of the draws of each statistic.
    """
    n = len(y_true)
    points = {name: fn(np.arange(n)) for name, fn in stats.items()}
    if n < 10:
        return {
            name: None if p is None else Interval(p, float("nan"), float("nan"), 0)
            for name, p in points.items()
        }
    groups = None
    if strata is not None:
        groups = [np.where(strata == label)[0] for label in pd.unique(strata)]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in stats}
    for _ in range(n_bootstrap):
        index = _draw(rng, n, groups)
        for name, fn in stats.items():
            value = fn(index)
            if value is not None:
                samples[name].append(value)
    if means is not None:
        means.update({name: float(np.mean(v)) for name, v in samples.items() if v})
    return {name: _interval(points[name], samples[name], n_bootstrap) for name in stats}


def rate_intervals(
    y_true: np.ndarray,
    flag: np.ndarray,
    w: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> dict[str, Interval | None]:
    """Intervals for ``rates``."""

    def pick(name: str):
        return lambda idx: rates(y_true[idx], flag[idx], w[idx])[name]

    names = ("positive_rate", "flag_rate", "precision", "recall")
    return bootstrap_statistics(
        y_true,
        w,
        {n: pick(n) for n in names},
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata=strata,
    )


def ece_interval(
    y_true: np.ndarray,
    prob: np.ndarray,
    w: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    bins: int = DEFAULT_BINS,
    strata: np.ndarray | None = None,
    means: dict[str, float] | None = None,
) -> Interval | None:
    """Interval of the ECE. ``means`` receives ``ece``, the mean of the bootstrap draws.

    ECE is biased upward under resampling, so the interval can exclude its own point value.
    The mean of the draws shows the size of that bias.
    """
    stats = {"ece": lambda idx: expected_calibration_error(y_true[idx], prob[idx], w[idx], bins)}
    return bootstrap_statistics(
        y_true, w, stats, n_bootstrap=n_bootstrap, seed=seed, strata=strata, means=means
    )["ece"]


def fmt(value: Interval | float | None) -> str:
    """Render a statistic. ``None`` is written out, never as 0."""
    if value is None:
        return "not measured"
    if isinstance(value, Interval):
        if np.isnan(value.low) or np.isnan(value.high):
            return f"{value.value:.3f} [interval not measured, too few rows]"
        return str(value)
    return f"{value:.3f}"


def false_positive_rate(y_true: np.ndarray, flag: np.ndarray, w: np.ndarray) -> float | None:
    """Weighted flag rate among rows labelled 0. ``None`` when no row is labelled 0."""
    negatives = y_true == 0
    return _ratio(float(w[negatives & (flag == 1)].sum()), float(w[negatives].sum()))


def fpr_by_reporting(
    y_true: np.ndarray,
    flag: np.ndarray,
    reports: np.ndarray,
    w: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> dict[str, Interval | None]:
    """False-positive rate on rows that report on a false claim, and on the other rows.

    ``reports`` is 0 or 1 for every row. A false positive is a flagged row with label 0.
    ``difference`` is the rate on reporting rows minus the rate on the other rows. Its
    interval comes from the same draws as the two rates.
    """

    def group(value: int):
        def fn(idx):
            mask = reports[idx] == value
            return false_positive_rate(y_true[idx][mask], flag[idx][mask], w[idx][mask])

        return fn

    def difference(idx):
        high, low = group(1)(idx), group(0)(idx)
        return None if high is None or low is None else high - low

    return bootstrap_statistics(
        y_true,
        w,
        {"reporting": group(1), "other": group(0), "difference": difference},
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata=strata,
    )


@dataclass
class StratumWeights:
    """Weights that stand each used row for ``pool / n_used`` corpus rows of its stratum."""

    weights: np.ndarray
    strata: np.ndarray
    #: Stratum code to its pool and the number of rows that reached the metrics.
    per_stratum: dict[str, dict[str, int]]
    #: Strata with rows in the pool and no used row. Their part of the corpus is not in
    #: any reweighted figure.
    empty: list[str]
    n_eff: float | None
    #: Strata with fewer than ``MIN_STRATUM_USED`` used rows. Their part of the corpus rests
    #: on very few rows.
    small: list[str] = field(default_factory=list)
    #: Strata with weight and fewer than 2 used rows. A one-row stratum has zero bootstrap
    #: variance, so the interval of a reweighted figure is too narrow.
    single: list[str] = field(default_factory=list)

    @property
    def interval_reliable(self) -> bool:
        return not self.single


def stratum_weights(all_rows: pd.DataFrame, used: pd.DataFrame) -> StratumWeights | None:
    """Renormalise the design weights over the rows that survive the label and score drops.

    The design weight ``pool / drawn`` assumes that every drawn row is used. A dropped row
    breaks that, and the reweighted figures would then under-count its stratum. Here the
    weight is ``pool / n_used`` per stratum. ``None`` when the frame has no design.
    """
    if "stratum" not in all_rows or all_rows["stratum"].isna().all():
        return None
    pools = all_rows.dropna(subset=["stratum"]).groupby("stratum")["pool"].first().astype(int)
    n_used = used["stratum"].value_counts()
    per = {
        str(k): {"pool": int(pools[k]), "n_used": int(n_used.get(k, 0))}
        for k in sorted(pools.index)
    }
    empty = [k for k, v in per.items() if v["pool"] > 0 and v["n_used"] == 0]
    if used["stratum"].isna().any():
        return None
    weights = np.array(
        [per[str(k)]["pool"] / per[str(k)]["n_used"] for k in used["stratum"]], dtype=float
    )
    live = {k: v for k, v in per.items() if v["pool"] > 0 and v["n_used"] > 0}
    small = [k for k, v in live.items() if v["n_used"] < MIN_STRATUM_USED]
    single = [k for k, v in live.items() if v["n_used"] < 2]
    return StratumWeights(
        weights,
        used["stratum"].astype(str).to_numpy(),
        per,
        empty,
        kish_n_eff(weights),
        small,
        single,
    )


def stratum_warning(sw: StratumWeights) -> list[str]:
    """Loud lines for strata that no used row represents."""
    lines = [
        f"WARNING: stratum {k} (pool {sw.per_stratum[k]['pool']}) has no used row, so every "
        "reweighted figure leaves that part of the corpus out"
        for k in sw.empty
    ]
    lines += [
        f"WARNING: stratum {k} (pool {sw.per_stratum[k]['pool']}) has only "
        f"{sw.per_stratum[k]['n_used']} used rows, fewer than {MIN_STRATUM_USED}, so its "
        "part of every reweighted figure is uncertain"
        for k in sw.small
    ]
    if sw.single:
        lines.append(
            "WARNING: reweighted intervals are NOT RELIABLE. Stratum "
            + ", ".join(sw.single)
            + " has fewer than 2 used rows, so the bootstrap cannot vary it and the interval "
            "is too narrow"
        )
    return lines


REPORTS_UNKNOWN = "reports_unknown"
UNCHECKED_CLAIM = "unchecked_claim"


def reporting_fpr(
    all_rows: pd.DataFrame,
    used: pd.DataFrame,
    y_true: np.ndarray,
    flag: np.ndarray,
    *,
    has_design: bool,
    n_bootstrap: int,
    seed: int,
) -> dict | None:
    """False-positive rate for rows that report on a false claim against the other rows.

    ``used`` holds the rows that reach the metrics, in the order of ``y_true`` and ``flag``.
    It needs ``reports`` (0, 1 or NaN) and may hold ``unchecked`` (bool) and ``stratum``.
    A false positive is a flagged row that carries label 0, so the rate is the flag rate
    among rows labelled 0. It comes in two forms: ``unweighted`` over the sample rows, and
    ``weighted`` to the corpus, with the bootstrap resampled inside each stratum.

    Two kinds of row are left out and counted under ``dropped``: a row with no adjudicated
    ``reports_on_false_claim`` (``reports_unknown``) and a row that the labeller marked
    ``unchecked``. The weights are renormalised over the rows that stay, so the dropped rows
    do not skew the figures. ``None`` when no row stays.
    """
    reports = used["reports"].to_numpy(dtype=float)
    unchecked = (
        used["unchecked"].to_numpy(dtype=bool) if "unchecked" in used else np.zeros(len(used), bool)
    )
    dropped = {
        REPORTS_UNKNOWN: int(np.isnan(reports).sum()),
        UNCHECKED_CLAIM: int((unchecked & ~np.isnan(reports)).sum()),
    }
    known = ~np.isnan(reports) & ~unchecked
    if not known.any():
        return None
    r = reports[known].astype(int)
    y, f = y_true[known], flag[known]
    negatives = y == 0
    out: dict = {
        "n_negatives_reporting": int((negatives & (r == 1)).sum()),
        "n_negatives_other": int((negatives & (r == 0)).sum()),
        "dropped": {k: v for k, v in dropped.items() if v},
        "unweighted": fpr_by_reporting(
            y, f, r, np.ones(len(r)), n_bootstrap=n_bootstrap, seed=seed
        ),
        "weighted": None,
        "wilson": {
            name: _wilson_block(int(((f == 1) & negatives & (r == value)).sum()),
                                int((negatives & (r == value)).sum()))
            for name, value in (("reporting", 1), ("other", 0))
        },
    }
    if has_design:
        sw = stratum_weights(all_rows, used[known])
        if sw is not None:
            out["weighted"] = fpr_by_reporting(
                y, f, r, sw.weights, n_bootstrap=n_bootstrap, seed=seed, strata=sw.strata
            )
            out["weighted_n_eff"] = sw.n_eff
            out["weighted_strata"] = sw
    return out


def _wilson_block(flagged: int, n: int) -> dict:
    low, high = wilson_interval(flagged, n)
    return {"flagged": flagged, "n": n, "ci_low": low, "ci_high": high}


def _wilson_text(block: dict) -> str:
    if not block["n"]:
        return "not measured, no row"
    return (
        f"{block['flagged']} of {block['n']}, Wilson 95% "
        f"[{block['ci_low']:.3f}, {block['ci_high']:.3f}]"
    )


def render_reporting_fpr(section: dict) -> list[str]:
    lines = [
        "  false-positive rate (flag rate among rows labelled 0), rows that report on a false "
        f"claim ({section['n_negatives_reporting']} rows labelled 0) against the other rows "
        f"({section['n_negatives_other']} rows labelled 0):",
        f"    unweighted, sample rows: reporting {fmt(section['unweighted']['reporting'])} "
        f"(n {section['n_negatives_reporting']}), "
        f"other {fmt(section['unweighted']['other'])} (n {section['n_negatives_other']})",
        f"    difference, reporting minus other: {fmt(section['unweighted'].get('difference'))}",
        f"    reporting rows, {_wilson_text(section['wilson']['reporting'])}",
        f"    other rows, {_wilson_text(section['wilson']['other'])}",
    ]
    if section["dropped"]:
        lines.append(
            "    rows left out of this rate, by reason: "
            + ", ".join(f"{k}={v}" for k, v in sorted(section["dropped"].items()))
        )
    if section["weighted"] is not None:
        lines.append(
            f"    reweighted to the corpus (n_eff {section['weighted_n_eff']:.1f}): reporting "
            f"{fmt(section['weighted']['reporting'])}, other {fmt(section['weighted']['other'])}"
        )
        lines.append(
            "    difference, reporting minus other, reweighted: "
            f"{fmt(section['weighted'].get('difference'))}"
        )
        lines += [f"    {line}" for line in stratum_warning(section["weighted_strata"])]
    return lines


def render_reweighted(name_values: dict, sw: StratumWeights, indent: str = "    ") -> list[str]:
    """Lines for reweighted figures, with the Kish effective sample size beside them."""
    n_eff = "not defined" if sw.n_eff is None else f"{sw.n_eff:.1f}"
    lines = [
        f"  reweighted to the corpus by stratum, 95% bootstrap intervals within strata "
        f"(n_eff {n_eff} of {len(sw.weights)} rows):"
    ]
    lines += [f"{indent}{k.replace('_', ' ')} {fmt(v)}" for k, v in name_values.items()]
    lines += [f"  {line}" for line in stratum_warning(sw)]
    return lines
