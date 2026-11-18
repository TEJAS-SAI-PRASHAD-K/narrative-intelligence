"""What a labelled sample with one class can still measure.

ROC-AUC, recall and precision need both classes, so an evaluator refuses them when the
usable labelled rows hold one class. A sample of negatives still measures four things:

* the observed positive rate, with a Wilson interval (0 of 150 gives an upper limit),
* the flag rate at the decision threshold, which is the false-positive rate,
* the flag rate on rows that report on a false claim against the other rows,
* the calibration of the positive-class probability.

The caller keeps the refusal and its reason code. This module only adds the figures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modeling.eval.metrics import Interval
from modeling.eval.weighted import (
    DEFAULT_BINS,
    bootstrap_statistics,
    ece_interval,
    expected_calibration_error,
    fmt,
    render_reporting_fpr,
    reporting_fpr,
    stratum_warning,
    stratum_weights,
    wilson_interval,
)

ONE_CLASS = "one_class_only"
NO_THRESHOLD = "no threshold configured"
REFUSAL = (
    f"the usable labelled rows hold one class only ({ONE_CLASS}). ROC-AUC, precision and "
    "recall are undefined and are not measured"
)
ECE_NOTE = (
    "ece of the positive-class probability, {bins} equal-width bins. With one class, every "
    "row has the same label, so the ece equals {equals}"
)


def _wilson(successes: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, n)
    return {
        "k": int(successes),
        "n": int(n),
        "value": None if n == 0 else successes / n,
        "ci_low": None if n == 0 else low,
        "ci_high": None if n == 0 else high,
        "method": "wilson 95%",
    }


def _mean_interval(
    prob: np.ndarray, w: np.ndarray, *, n_bootstrap: int, seed: int, strata: np.ndarray | None
) -> Interval | None:
    def mean(idx):
        total = float(w[idx].sum())
        return None if total <= 0 else float(np.average(prob[idx], weights=w[idx]))

    return bootstrap_statistics(
        np.zeros(len(prob), dtype=int),
        w,
        {"mean": mean},
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata=strata,
    )["mean"]


def _flag_interval(
    flag: np.ndarray, w: np.ndarray, *, n_bootstrap: int, seed: int, strata: np.ndarray | None
) -> Interval | None:
    def rate(idx):
        total = float(w[idx].sum())
        return None if total <= 0 else float(w[idx][flag[idx] == 1].sum() / total)

    return bootstrap_statistics(
        np.zeros(len(flag), dtype=int),
        w,
        {"rate": rate},
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata=strata,
    )["rate"]


def one_class_report(
    all_rows: pd.DataFrame,
    used: pd.DataFrame,
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    threshold: float | None,
    threshold_source: str | None = None,
    has_design: bool,
    n_bootstrap: int,
    seed: int,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Figures for a sample whose usable rows hold one class.

    ``used`` holds the rows that reach the metrics, in the order of ``y_true`` and ``prob``.
    ``threshold`` is ``None`` when none is configured, and no flag figure is then computed.
    Weighted figures use ``pool / n_used`` per stratum, as the transfer metrics do.
    """
    n = len(y_true)
    present = int(y_true[0])
    only_negatives = present == 0
    report: dict[str, Any] = {
        "class_present": "negative" if only_negatives else "positive",
        "n": n,
        "positive_rate": _wilson(int(y_true.sum()), n),
        "refused": ["roc_auc", "precision", "recall"],
        "reason_code": ONE_CLASS,
    }
    sw = stratum_weights(all_rows, used) if has_design else None
    weights = sw.weights if sw is not None else np.ones(n)
    strata = sw.strata if sw is not None else None
    report["weighted_reason"] = (
        None
        if sw is not None
        else "a labelled row has no stratum in the design file"
        if has_design
        else "the sheet has no design file"
    )
    if sw is not None:
        report["strata_info"] = sw

    calibration: dict[str, Any] = {
        "ece_note": ECE_NOTE.format(
            bins=bins,
            equals="the mean predicted probability"
            if only_negatives
            else "one minus the mean predicted probability",
        )
    }
    calibration["mean_probability"] = _mean_interval(
        prob, np.ones(n), n_bootstrap=n_bootstrap, seed=seed, strata=None
    )
    calibration["ece"] = ece_interval(
        y_true, prob, np.ones(n), n_bootstrap=n_bootstrap, seed=seed, bins=bins
    )
    calibration["ece_point"] = expected_calibration_error(y_true, prob, np.ones(n), bins)
    if sw is not None:
        calibration["mean_probability_weighted"] = _mean_interval(
            prob, weights, n_bootstrap=n_bootstrap, seed=seed, strata=strata
        )
        calibration["ece_weighted"] = ece_interval(
            y_true, prob, weights, n_bootstrap=n_bootstrap, seed=seed, bins=bins, strata=strata
        )
    report["calibration"] = calibration

    if not only_negatives:
        report["flag"] = {"not_measured": "a sample of positives only: that rate is recall"}
        return report
    if threshold is None:
        report["flag"] = {"not_measured": NO_THRESHOLD}
        report["reporting"] = None
        return report

    flag = (prob >= threshold).astype(int)
    block: dict[str, Any] = {
        "threshold": threshold,
        "threshold_source": threshold_source,
        "unweighted_wilson": _wilson(int(flag.sum()), n),
        "unweighted_bootstrap": _flag_interval(
            flag, np.ones(n), n_bootstrap=n_bootstrap, seed=seed, strata=None
        ),
        "weighted_bootstrap": None
        if sw is None
        else _flag_interval(flag, weights, n_bootstrap=n_bootstrap, seed=seed, strata=strata),
    }
    report["flag"] = block
    if sw is not None:
        per: dict[str, Any] = {}
        for code in sorted(set(strata.tolist())):
            mask = strata == code
            entry = _wilson(int(flag[mask].sum()), int(mask.sum()))
            entry["pool"] = sw.per_stratum[code]["pool"]
            per[code] = entry
        report["per_stratum"] = per
    report["reporting"] = reporting_fpr(
        all_rows,
        used,
        y_true,
        flag,
        has_design=has_design,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return report


def _wilson_text(block: dict[str, Any]) -> str:
    if not block["n"]:
        return "not measured, no row"
    return (
        f"{block['k']} of {block['n']}, {block['value']:.3f} "
        f"[{block['ci_low']:.3f}, {block['ci_high']:.3f}] ({block['method']})"
    )


def render_one_class(report: dict[str, Any]) -> list[str]:
    """Lines for the figures that one class can still give."""
    cls = report["class_present"]
    lines = [
        f"  what a sample of {cls}s only can still measure (n {report['n']}):",
        "    not measured: " + ", ".join(report["refused"]) + f" ({report['reason_code']})",
        f"    observed positive rate {_wilson_text(report['positive_rate'])}",
    ]
    flag = report["flag"]
    if "not_measured" in flag:
        lines.append(f"    flag rate not measured, {flag['not_measured']}")
    else:
        lines.append(
            f"    flag rate at threshold {flag['threshold']} ({flag['threshold_source']}), "
            "which is the false-positive rate:"
        )
        lines.append(f"      unweighted, {_wilson_text(flag['unweighted_wilson'])}")
        lines.append(f"      unweighted, bootstrap 95% {fmt(flag['unweighted_bootstrap'])}")
        if flag["weighted_bootstrap"] is not None:
            sw = report["strata_info"]
            n_eff = "not defined" if sw.n_eff is None else f"{sw.n_eff:.1f}"
            lines.append(
                f"      reweighted to the corpus, bootstrap 95% {fmt(flag['weighted_bootstrap'])} "
                f"(n_eff {n_eff})"
            )
            lines += [f"      {line}" for line in stratum_warning(sw)]
        else:
            lines.append(
                f"      reweighted to the corpus: not measured, {report['weighted_reason']}"
            )
    if report.get("per_stratum"):
        lines.append("    flag rate per stratum:")
        for code, entry in report["per_stratum"].items():
            lines.append(f"      {code} (pool {entry['pool']}): {_wilson_text(entry)}")
    elif "not_measured" not in flag:
        lines.append(f"    flag rate per stratum: not measured, {report['weighted_reason']}")
    if report.get("reporting"):
        lines += render_reporting_fpr(report["reporting"])
    elif "not_measured" not in flag:
        lines.append(
            "    flag rate on reporting rows: not measured, no row has a known report flag"
        )
    cal = report["calibration"]
    lines.append(f"    mean predicted probability {fmt(cal['mean_probability'])}")
    if "mean_probability_weighted" in cal:
        lines.append(
            f"    mean predicted probability, reweighted {fmt(cal['mean_probability_weighted'])}"
        )
    lines.append(f"    ece {fmt(cal['ece'])}")
    if "ece_weighted" in cal:
        lines.append(f"    ece, reweighted {fmt(cal['ece_weighted'])}")
    lines.append(f"    {cal['ece_note']}")
    return lines
