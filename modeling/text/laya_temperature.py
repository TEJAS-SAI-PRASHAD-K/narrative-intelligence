"""Temperature and calibration diagnostics for the two-option Laya ``noul`` head.

``laya.train.finetune`` fits one temperature T on a held-out slice of the training data.
It clamps T to ``[0.5, 5]``. Both fine-tunes ended on the upper clamp, so the fit never
converged and the vendor-reported expected calibration error stayed high. This module
answers one question without any model weights: where does the optimum lie on the
validation rows, and does temperature scaling add anything once isotonic regression runs?

Everything here works on the logit gap ``g = z_true - z_false``. For two options,
``softmax(z / T)[1]`` equals ``sigmoid(g / T)``, so one number per row is enough. The gap is
the raw logit difference, before any temperature.

The module has no dependency on ``laya`` or ``torch``. The GPU part, which produces the gaps,
lives in ``scripts/diagnose_laya_calibration.py``.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from modeling.eval.calibrate import expected_calibration_error

#: Seed of the bootstrap and of every random choice in the diagnosis. It is the project seed.
SEED = 20260813

#: Search range of the diagnostic fit. The vendor clamp is [0.5, 5]. A fit that needs a scale
#: outside the wide range below is not a temperature problem, and the report says so.
T_MIN = 0.05
T_MAX = 100.0
#: The clamp of ``laya.common``. Kept here so a report can say which side of it a fit falls.
VENDOR_T_MIN = 0.5
VENDOR_T_MAX = 5.0

#: A fit within this relative distance of a bound counts as ending on the bound.
BOUND_RTOL = 1e-3
#: Bin counts. The repo reads P(true) in 10 bins. The vendor reads top-1 confidence in 15.
ECE_BINS_PROB = 10
ECE_BINS_TOP1 = 15
#: Inverse regularisation of the intercept fit. Bounded, so a separable slice stays finite.
PLATT_C = 1e4
#: Size of the vendor calibration slice (``calib_max`` in configs/models.yaml).
VENDOR_CALIB_ROWS = 400


def validate_inputs(gap: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    gap = np.asarray(gap, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if len(gap) != len(labels) or not len(gap):
        raise ValueError("gap and labels must be non-empty and of the same length")
    if not np.all(np.isfinite(gap)):
        raise ValueError("the logit gap holds a non-finite value")
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("labels must be 0 or 1")
    return gap, labels


def nll(gap: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean binary negative log likelihood of ``sigmoid(gap / T)``.

    This is the objective of the vendor fit, written for two options and in a form that
    does not overflow: ``log(1 + e^z) - y z`` with ``z = g / T``.
    """
    z = gap / temperature
    return float(np.mean(np.logaddexp(0.0, z) - labels * z))


def nll_curve(
    gap: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None
) -> dict[str, list[float]]:
    """NLL over a log grid of temperatures, 0.05 to 100 by default."""
    gap, labels = validate_inputs(gap, labels)
    if grid is None:
        grid = np.logspace(np.log10(T_MIN), np.log10(T_MAX), 61)
    values = [nll(gap, labels, float(t)) for t in grid]
    return {"temperature": [float(t) for t in grid], "nll": values}


@dataclass(frozen=True)
class TemperatureFit:
    value: float
    nll: float
    nll_at_one: float
    lower: float
    upper: float
    #: True when the optimum is a bound of the search range. The fit has not converged.
    at_bound: bool
    #: ``"lower"``, ``"upper"`` or ``None``.
    bound: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "nll": self.nll,
            "nll_at_one": self.nll_at_one,
            "search_range": [self.lower, self.upper],
            "at_bound": self.at_bound,
            "bound": self.bound,
        }


def fit_temperature(
    gap: np.ndarray, labels: np.ndarray, lower: float = T_MIN, upper: float = T_MAX
) -> TemperatureFit:
    """The temperature that minimises the NLL inside ``[lower, upper]``.

    The loss is convex in ``s = 1 / T``, so a bounded scalar search over ``s`` has one
    minimum and no local traps. A result on a bound is reported as such. It is never
    returned as if it were an optimum.
    """
    gap, labels = validate_inputs(gap, labels)
    if not 0.0 < lower < upper:
        raise ValueError("need 0 < lower < upper")
    result = minimize_scalar(
        lambda s: nll(gap, labels, 1.0 / s),
        bounds=(1.0 / upper, 1.0 / lower),
        method="bounded",
        options={"xatol": 1e-10},
    )
    value = float(1.0 / result.x)
    bound = None
    if abs(value - upper) <= BOUND_RTOL * upper:
        bound = "upper"
    elif abs(value - lower) <= BOUND_RTOL * lower:
        bound = "lower"
    return TemperatureFit(
        value=value,
        nll=nll(gap, labels, value),
        nll_at_one=nll(gap, labels, 1.0),
        lower=lower,
        upper=upper,
        at_bound=bound is not None,
        bound=bound,
    )


def fit_platt(gap: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Slope and intercept of ``sigmoid(a * g + b)``, to tell a bias from a scale problem.

    A temperature has no intercept. When the validation base rate differs from the
    calibration base rate, only an intercept can correct it. The gap is standardised before
    the fit and the result is mapped back, so a gap of large scale cannot stall the solver.
    A convergence warning is kept in the result and never dropped.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    gap, labels = validate_inputs(gap, labels)
    mean, std = float(gap.mean()), float(gap.std())
    std = std if std > 0 else 1.0
    z = (gap - mean) / std
    model = LogisticRegression(C=PLATT_C, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(z.reshape(-1, 1), labels)
    notes = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    slope_z, intercept_z = float(model.coef_[0, 0]), float(model.intercept_[0])
    a = slope_z / std
    b = intercept_z - slope_z * mean / std
    p = expit(a * gap + b)
    eps = 1e-12
    loss = float(-np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps)))
    return {
        "slope": a,
        "intercept": b,
        "nll": loss,
        "converged": not notes,
        "convergence_warnings": notes,
    }


# --- expected calibration error ---------------------------------------------------------
def top1_ece(labels: np.ndarray, prob: np.ndarray, bins: int = ECE_BINS_TOP1) -> float:
    """ECE of the top-1 confidence, the quantity that the vendor report calls ECE."""
    prob = np.asarray(prob, dtype=float)
    labels = np.asarray(labels).astype(int)
    conf = np.maximum(prob, 1.0 - prob)
    correct = ((prob >= 0.5).astype(int) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(conf, edges, right=True) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.sum() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total / len(prob))


def prob_ece(labels: np.ndarray, prob: np.ndarray) -> float:
    return float(expected_calibration_error(labels, prob, ECE_BINS_PROB))


def bootstrap_interval(
    labels: np.ndarray,
    prob: np.ndarray,
    groups: np.ndarray | None,
    statistic: Any,
    *,
    n_boot: int = 1000,
    seed: int = SEED,
    level: float = 0.95,
) -> dict[str, float]:
    """Point value and percentile interval of ``statistic(labels, prob)``.

    Rows that share a group are resampled together, because they are not independent. With
    no groups, rows are resampled one by one and the report must say so.
    """
    labels = np.asarray(labels).astype(int)
    prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        clusters = [np.array([i]) for i in range(len(labels))]
    else:
        _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse)
        clusters = np.split(order, np.cumsum(counts)[:-1])
    values = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[j] for j in pick])
        values[i] = statistic(labels[rows], prob[rows])
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        "value": float(statistic(labels, prob)),
        "low": float(low),
        "high": float(high),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "cluster": "group" if groups is not None else "row",
    }


# --- out-of-fold calibration ------------------------------------------------------------
def fold_ids(groups: np.ndarray, n_splits: int = 5, seed: int = SEED) -> np.ndarray:
    """Assign each row to a fold so that a group is never on both sides of a split."""
    names, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    if len(names) < n_splits:
        raise ValueError(f"{len(names)} groups cannot fill {n_splits} folds")
    order = np.random.default_rng(seed).permutation(len(names))
    fold_of_group = np.empty(len(names), dtype=int)
    fold_of_group[order] = np.arange(len(names)) % n_splits
    return fold_of_group[inverse]


def served_scores(gap: np.ndarray, temperature: float) -> np.ndarray:
    """The score that the scorer sees: ``laya`` rounds P(true) to 4 decimals."""
    return np.round(expit(gap / temperature), 4)


def _apply(method: str, train: tuple, test_gap: np.ndarray, t_ckpt: float) -> np.ndarray:
    gap, labels = train
    if method == "temperature":
        return expit(test_gap / fit_temperature(gap, labels).value)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(gap, labels)
        return model.predict(test_gap)
    if method == "temperature_isotonic":
        t = fit_temperature(gap, labels).value
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(gap / t, labels)
        return model.predict(test_gap / t)
    if method == "isotonic_served":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(served_scores(gap, t_ckpt), labels)
        return model.predict(served_scores(test_gap, t_ckpt))
    raise ValueError(f"unknown method {method!r}")


def out_of_fold(
    method: str,
    gap: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    t_ckpt: float,
    *,
    n_splits: int = 5,
    seed: int = SEED,
) -> np.ndarray:
    """Predictions for every row from a calibrator that never saw the row's group."""
    gap, labels = validate_inputs(gap, labels)
    folds = fold_ids(groups, n_splits, seed)
    out = np.empty(len(gap))
    for k in range(n_splits):
        test = folds == k
        train = ~test
        if len(set(labels[train].tolist())) < 2:
            raise ValueError(f"fold {k} has one class only in its training rows")
        out[test] = _apply(method, (gap[train], labels[train]), gap[test], t_ckpt)
    return out


#: Fold seeds of the guard. One split of 400 rows is noisy, so the difference is averaged
#: over several splits.
FOLD_SEEDS = tuple(SEED + i for i in range(5))


def oof_isotonic(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, n_splits: int, seed: int
) -> np.ndarray:
    folds = fold_ids(groups, n_splits, seed)
    out = np.empty(len(scores))
    for k in range(n_splits):
        test = folds == k
        train = ~test
        if len(set(labels[train].tolist())) < 2:
            raise ValueError(f"fold {k} has one class only in its training rows")
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores[train], labels[train])
        out[test] = model.predict(scores[test])
    return out


def isotonic_oof_ece(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    fold_seeds: tuple[int, ...] = FOLD_SEEDS,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Out-of-fold ECE of an isotonic map on served scores, against the raw scores.

    The in-sample ECE of an isotonic fit is near zero by construction and says nothing.
    The difference ``ECE(out of fold) - ECE(raw)`` is averaged over several fold seeds. A
    group bootstrap with the project seed gives its 95 percent interval. The calibrator
    makes the scores worse only when the lower end of that interval is above zero. A fixed
    tolerance would reject a calibrated input by chance on a small validation set.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    raw = np.clip(scores, 0.0, 1.0)
    oofs = [
        oof_isotonic(scores, labels, groups, n_splits=n_splits, seed=s) for s in fold_seeds
    ]
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    clusters = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)

    def difference(rows: np.ndarray) -> float:
        before = prob_ece(labels[rows], raw[rows])
        return float(np.mean([prob_ece(labels[rows], o[rows]) for o in oofs]) - before)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        diffs[i] = difference(np.concatenate([clusters[j] for j in pick]))
    low, high = np.quantile(diffs, [0.025, 0.975])
    everything = np.arange(len(scores))
    return {
        "ece_before": prob_ece(labels, raw),
        "ece_after_oof": float(np.mean([prob_ece(labels, o) for o in oofs])),
        "difference": difference(everything),
        "difference_low": float(low),
        "difference_high": float(high),
        "fold_seeds": list(fold_seeds),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def ids_hash(ids: Any) -> str:
    """A hash of the validation ids, in order. It ties a stored gap file to its rows."""
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8") + b"\n")
    return digest.hexdigest()


# --- the report -------------------------------------------------------------------------
def _ece_block(labels: np.ndarray, prob: np.ndarray, groups: np.ndarray | None, n_boot: int):
    return {
        "ece_prob": bootstrap_interval(labels, prob, groups, prob_ece, n_boot=n_boot),
        "ece_top1": bootstrap_interval(labels, prob, groups, top1_ece, n_boot=n_boot),
    }


def subsample_argmins(
    gap: np.ndarray,
    labels: np.ndarray,
    *,
    rows: int = VENDOR_CALIB_ROWS,
    repeats: int = 200,
    seed: int = SEED,
) -> dict[str, Any]:
    """Fit T on random slices the size of the vendor calibration slice.

    The vendor fits on ``calib_max`` rows. This shows how far one fit of that size can move.
    """
    rng = np.random.default_rng(seed)
    rows = min(rows, len(gap))
    fits = []
    for _ in range(repeats):
        pick = rng.choice(len(gap), size=rows, replace=False)
        if len(set(labels[pick].tolist())) < 2:
            continue
        fits.append(fit_temperature(gap[pick], labels[pick]))
    values = np.array([f.value for f in fits])
    return {
        "rows": int(rows),
        "repeats": int(len(fits)),
        "seed": int(seed),
        "percentiles": {
            str(q): float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)
        },
        "share_above_vendor_max": float(np.mean(values > VENDOR_T_MAX)),
        "share_at_search_bound": float(np.mean([f.at_bound for f in fits])),
    }


def diagnose(
    gap: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    *,
    checkpoint_temperature: float,
    sources: np.ndarray | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """The full diagnosis of one checkpoint on its validation rows."""
    gap, labels = validate_inputs(gap, labels)
    t_ckpt = float(checkpoint_temperature)
    fit = fit_temperature(gap, labels)
    curve = nll_curve(gap, labels)
    grid_best = int(np.argmin(curve["nll"]))

    report: dict[str, Any] = {
        "rows": int(len(gap)),
        "positives": int(labels.sum()),
        "base_rate": float(labels.mean()),
        "gap_quantiles": {
            str(q): float(np.percentile(gap, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)
        },
        "accuracy_at_zero": float(np.mean((gap > 0).astype(int) == labels)),
        "checkpoint_temperature": t_ckpt,
        "nll_curve": curve,
        "nll_grid_argmin": curve["temperature"][grid_best],
        "temperature_fit": fit.as_dict(),
        "nll_at_checkpoint_temperature": nll(gap, labels, t_ckpt),
        "platt": fit_platt(gap, labels),
        "subsample_fit": subsample_argmins(gap, labels),
    }

    if sources is not None:
        per_source: dict[str, Any] = {}
        for name in sorted(set(np.asarray(sources).astype(str).tolist())):
            mask = np.asarray(sources).astype(str) == name
            if mask.sum() < 20 or len(set(labels[mask].tolist())) < 2:
                per_source[name] = {"rows": int(mask.sum()), "skipped": "too few rows or one class"}
                continue
            sub = fit_temperature(gap[mask], labels[mask])
            per_source[name] = {
                "rows": int(mask.sum()),
                "base_rate": float(labels[mask].mean()),
                "temperature_fit": sub.as_dict(),
                "nll_grid": nll_curve(gap[mask], labels[mask]),
            }
        report["per_source"] = per_source

    raw_prob = expit(gap / t_ckpt)
    variants: dict[str, Any] = {
        "raw_checkpoint_temperature": _ece_block(labels, raw_prob, groups, n_boot),
        "raw_temperature_one": _ece_block(labels, expit(gap), groups, n_boot),
    }
    report["ece_in_sample"] = {
        "temperature_only": prob_ece(labels, expit(gap / fit.value)),
    }
    if groups is not None:
        for method in ("temperature", "isotonic", "temperature_isotonic", "isotonic_served"):
            prob = out_of_fold(method, gap, labels, groups, t_ckpt)
            variants[method] = _ece_block(labels, prob, groups, n_boot)
        report["ece_protocol"] = (
            "Raw rows use no fit. The other rows are out of fold: five folds, split by group, "
            f"seed {SEED}. Intervals come from a group bootstrap with seed {SEED}."
        )
    else:
        report["ece_protocol"] = "No group ids. Only the raw rows were computed."
    report["ece"] = variants
    report["findings"] = findings(report)
    return report


def findings(report: dict[str, Any]) -> list[str]:
    """Short machine-made notes that point at the likely cause. The reader decides."""
    notes: list[str] = []
    fit = report["temperature_fit"]
    if fit["at_bound"]:
        notes.append(
            f"The optimum is the {fit['bound']} bound {fit['search_range']} of the wide search. "
            "The NLL has no interior minimum, so temperature alone cannot calibrate these "
            "logits. Check label orientation and the base rate before the scale."
        )
    elif fit["value"] > VENDOR_T_MAX:
        notes.append(
            f"The optimum {fit['value']:.3g} is inside the wide search and above the vendor "
            f"clamp {VENDOR_T_MAX}. The logits are very overconfident, and the clamp is the "
            "reason that the vendor fit stops at 5."
        )
    elif fit["value"] < VENDOR_T_MIN:
        notes.append(f"The optimum {fit['value']:.3g} is below the vendor clamp {VENDOR_T_MIN}.")
    else:
        notes.append(
            f"The optimum {fit['value']:.3g} is inside the vendor clamp on the validation rows. "
            "The calibration slice differs from validation, so see the subsample fit."
        )
    platt = report["platt"]
    if report["nll_at_checkpoint_temperature"] - platt["nll"] > 0.02 and (
        fit["nll"] - platt["nll"] > 0.02
    ):
        notes.append(
            f"A fit with an intercept (b = {platt['intercept']:.3g}) lowers the NLL by more than "
            "0.02 against the best temperature. A bias is present, and temperature cannot fix it."
        )
    sub = report["subsample_fit"]
    if sub["share_above_vendor_max"] > 0.2:
        notes.append(
            f"On {sub['rows']}-row slices, {sub['share_above_vendor_max']:.0%} of the fits pass "
            "the vendor clamp. One small slice is enough to reach the clamp."
        )
    if report["checkpoint_temperature"] >= VENDOR_T_MAX - 1e-9:
        notes.append("The checkpoint temperature is on the vendor upper clamp.")
    return notes


#: Index of the ``noul`` type in ``laya.common.QTYPES``, and its two-option bucket key.
NOUL_TYPE = 2
NOUL_BUCKET = "noul:2"


def install_noul_temperature(agent: Any, temperature: float) -> None:
    """Make a loaded agent divide its ``noul`` logits by ``temperature``.

    The agent divides the raw logits and rounds P(true) to 4 decimals afterwards, so this
    scale acts before the rounding. Re-scaling an already rounded probability would act
    after the ties exist. ``clamp_temperature`` runs at checkpoint load only, so the wide
    range is not clamped here. The per-bucket value is removed, because it takes priority
    over the per-type value.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
    per_type = list(agent.temperature)
    per_type[NOUL_TYPE] = float(temperature)
    agent.temperature = per_type
    agent.temperature_by_options = {
        k: v for k, v in dict(agent.temperature_by_options).items() if k != NOUL_BUCKET
    }
    agent.lang_temperatures = {}


def gap_cache_path(path: Any) -> Any:
    """The gap file name with the ``.npz`` suffix. ``numpy`` adds it on a write, so a path
    without it would be written to one name and read from another."""
    from pathlib import Path

    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def write_gap_cache(path: Any, **fields: Any) -> Any:
    path = gap_cache_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fields)
    return path


def read_gap_cache(
    path: Any, *, model: str, checkpoint_sha256: str, validation_ids_hash: str
) -> dict[str, Any]:
    """Read a stored gap file and refuse it unless it matches the model, the weights and rows.

    Raises ``ValueError`` that names every mismatch.
    """
    path = gap_cache_path(path)
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    stored = np.load(path, allow_pickle=False)
    problems = []
    for key, expected in (
        ("model", model),
        ("checkpoint_sha256", checkpoint_sha256),
        ("validation_ids_hash", validation_ids_hash),
    ):
        found = str(stored[key]) if key in stored.files else None
        if found != expected:
            problems.append(f"{key} is {found!r}, expected {expected!r}")
    if problems:
        raise ValueError(f"{path} does not match this run: " + "; ".join(problems))
    return {key: stored[key] for key in stored.files}


def refit_search_reason() -> str:
    return (
        f"Search range [{T_MIN}, {T_MAX}] is wider than the vendor clamp "
        f"[{VENDOR_T_MIN}, {VENDOR_T_MAX}] because the vendor fit reached its upper clamp on "
        "both fine-tunes, and a fit on a bound has not converged."
    )
