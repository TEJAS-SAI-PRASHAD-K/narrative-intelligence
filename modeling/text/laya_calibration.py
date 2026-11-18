"""The calibrator that turns a raw Laya score into a ``misinfo_prob``.

``record_scores.misinfo_prob`` is contractually a calibrated probability, and
fusion multiplies it. A raw Laya ``noul`` value is not one: on the English
validation split of the comparison the mean confidence is about 0.97 and the
expected calibration error is 0.159. Written as it is, Hindi narratives would
outrank English ones in fusion for a reason that has nothing to do with
misinformation.

So a Laya checkpoint serves ``misinfo_prob`` only with a calibrator beside it,
in the same way ``MisinfoClassifier.load`` refuses a checkpoint without
``calibrator.json``. The files here have their own names because ``laya.load``
owns the checkpoint directory and may write files of its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from modeling.checkpoint_identity import directory_identity
from modeling.eval.calibrate import Calibrator, expected_calibration_error
from modeling.text.laya_temperature import isotonic_oof_ece

log = logging.getLogger(__name__)

CALIBRATOR_FILE = "misinfo_prob_calibrator.json"
REPORT_FILE = "misinfo_prob_calibration_report.json"
#: Files that the fit adds to a checkpoint directory. They stay out of its identity.
CALIBRATOR_FILES = (CALIBRATOR_FILE, REPORT_FILE)

#: A fit is refused unless the ROC-AUC of the raw scores on the fit rows is above this.
#: Isotonic regression can only order what the scores already order. A score at or below
#: this floor is inverted or close to chance, and a calibrator fitted to it would present
#: noise as a probability. The multilingual fine-tune is far above it on the English
#: validation rows.
MIN_FIT_AUC = 0.60

#: A fit is refused when the lower end of the 95 percent interval of ``ECE(out of fold) -
#: ECE(raw)`` is above this. The ECE on the fit rows is near zero for any isotonic map, so it
#: cannot show a rise. The out-of-fold difference can. Zero is the right limit because the
#: interval, not a fixed margin, carries the noise of a small validation set.
OOF_DIFFERENCE_LIMIT = 0.0

#: The wide search range of the temperature refit, in ``laya_temperature``. The vendor clamp is
#: [0.5, 5]. A very overconfident head needs a larger scale, and a scale at 5 leaves extreme
#: rows saturated: the vendor rounds P(true) to 4 decimals, so a gap above about 9.2 * T
#: becomes exactly 1.0 and ties with every row above it. The clamp therefore does change
#: the served ranking.
TEMPERATURE_REFIT_KEY = "temperature_refit"

#: A calibrator is refused when its output takes fewer distinct values than this over the
#: fitted score range. A constant or near-constant map gives every record the same
#: probability and carries no information into fusion.
MIN_DISTINCT_OUTPUTS = 5
#: Output values are compared at this many decimals, and sampled at this many points.
_DISTINCT_DECIMALS = 4
_DISTINCT_GRID = 1001

#: Stated in every report and in the model card. The labels are the benchmark
#: labels of the comparison, so the calibration is not measured on Hindi.
LABEL_STATEMENT = (
    "Fit on the validation split of the English benchmarks (LIAR, FakeNewsNet, CoAID). "
    "No Hindi label took part. Calibration on Hindi is not measured."
)

#: The reason codes of a refused calibrator. The scoring stage prefixes them with ``laya:``.
MISSING = "calibrator_missing"
UNREADABLE = "calibrator_unreadable"
REJECTED = "calibrator_rejected"
#: The calibrator file is marked as made from the demo fixtures.
DEMO = "calibrator_demo"
#: The calibrator was fitted for other weights than the directory that serves it.
CHECKPOINT_MISMATCH = "calibrator_checkpoint_mismatch"
#: ``laya.temperature_refit`` is on, and the calibrator holds no refitted temperature.
TEMPERATURE_MISSING = "calibrator_temperature_missing"
#: ``laya.temperature_refit`` is off, and the calibrator was fitted with a refitted temperature.
TEMPERATURE_UNEXPECTED = "calibrator_temperature_unexpected"


class LayaCalibratorError(RuntimeError):
    """A calibrator cannot be fitted or used. ``reason`` is a counted reason code."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class LoadedCalibrator:
    calibrator: Calibrator
    method: str
    #: First 8 hex digits of a sha256 over the calibrator file and the checkpoint identity.
    digest: str
    #: The ``[min, max]`` of the raw scores that the calibrator was fitted on.
    fit_range: tuple[float, float]
    #: The refitted temperature that the scorer applies before it rounds, or ``None``.
    temperature: float | None = None

    @property
    def stamp(self) -> str:
        """The part of ``model_versions["laya"]`` that names the calibrator.

        A refitted temperature changes the served numbers, so the stamp shows it.
        """
        tag = "" if self.temperature is None else f"+t{self.temperature:.4g}"
        return f"{self.method}{tag}@{self.digest}"


def _is_monotonic(calibrator: Calibrator, slack: float = 1e-9) -> bool:
    grid = calibrator.transform(np.linspace(0.0, 1.0, 129))
    return bool(np.all(np.diff(grid) >= -slack))


def distinct_outputs(calibrator: Calibrator, fit_range: tuple[float, float]) -> int:
    """How many distinct values the calibrator gives over its fitted score range."""
    low, high = fit_range
    grid = np.linspace(low, high, _DISTINCT_GRID)
    return int(len(np.unique(np.round(calibrator.transform(grid), _DISTINCT_DECIMALS))))


def fit_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str = "isotonic",
    groups: np.ndarray | None = None,
) -> tuple[Calibrator, dict[str, Any]]:
    """Fit on validation scores and return the calibrator with its report.

    Raises :class:`LayaCalibratorError` with reason ``calibrator_rejected`` when the
    labels hold one class only, when the raw scores do not rank the labels above
    ``MIN_FIT_AUC``, when ``Calibrator`` rejects the fit because it inverted the ranking,
    when the fitted map has fewer than ``MIN_DISTINCT_OUTPUTS`` distinct values, or when
    the out-of-fold ECE is higher than the raw ECE by more than ``MAX_OOF_ECE_RISE``.
    Nothing is returned in that case, so nothing can be written.

    ``groups`` holds one group id per row. With groups, the report has the out-of-fold ECE,
    from five folds that never split a group. Without groups the report says that the check
    did not run, and the reason is counted in ``oof_skipped``.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if not len(scores) or len(scores) != len(labels):
        raise LayaCalibratorError(REJECTED, "scores and labels are empty or differ in length")
    if not np.all(np.isfinite(scores)):
        raise LayaCalibratorError(REJECTED, "the validation scores hold a non-finite value")
    if len(set(labels.tolist())) < 2:
        raise LayaCalibratorError(REJECTED, "the validation labels hold one class only")
    if method not in {"isotonic", "platt"}:
        raise ValueError(f"method must be isotonic or platt, got {method!r}")

    auc = float(roc_auc_score(labels, scores))
    if not auc > MIN_FIT_AUC:
        raise LayaCalibratorError(
            REJECTED,
            f"the raw scores have ROC-AUC {auc:.3f} on the fit rows, which is not above the "
            f"floor {MIN_FIT_AUC}. The scores are inverted or uninformative",
        )

    calibrator = Calibrator(method)
    result = calibrator.fit(scores, labels)
    if result.method == "none":
        raise LayaCalibratorError(
            REJECTED, f"the calibrator rejected the fit and wrote nothing: {result.note}"
        )
    fit_range = (float(scores.min()), float(scores.max()))
    distinct = distinct_outputs(calibrator, fit_range)
    if distinct < MIN_DISTINCT_OUTPUTS:
        raise LayaCalibratorError(
            REJECTED,
            f"the fitted map has {distinct} distinct values over the fit range, below "
            f"the minimum {MIN_DISTINCT_OUTPUTS}",
        )
    calibrated = calibrator.transform(scores)
    oof: dict[str, Any] = {}
    if groups is None:
        oof["oof_skipped"] = "no_groups"
    else:
        group_array = np.asarray(groups).ravel()
        if len(group_array) != len(scores):
            raise LayaCalibratorError(REJECTED, "groups and scores differ in length")
        if method != "isotonic":
            oof["oof_skipped"] = "method_not_isotonic"
        else:
            try:
                oof.update(isotonic_oof_ece(scores, labels, group_array))
            except ValueError as exc:
                oof["oof_skipped"] = f"fold_error: {exc}"
            else:
                if oof["difference_low"] > OOF_DIFFERENCE_LIMIT:
                    raise LayaCalibratorError(
                        REJECTED,
                        f"the out-of-fold ECE rises from {oof['ece_before']:.4f} to "
                        f"{oof['ece_after_oof']:.4f}, and the 95 percent interval of the "
                        f"difference [{oof['difference_low']:.4f}, "
                        f"{oof['difference_high']:.4f}] is above zero. The calibrator makes "
                        "the scores worse on rows it did not see",
                    )
    if "oof_skipped" in oof:
        log.warning("the out-of-fold ECE check did not run: %s", oof["oof_skipped"])
    report = {
        "method": result.method,
        "requested_method": method,
        "rows": int(len(scores)),
        "positives": int(labels.sum()),
        "raw_roc_auc": round(auc, 6),
        "fit_range": list(fit_range),
        "distinct_outputs": distinct,
        "brier_before": round(result.brier_before, 6),
        "brier_after": round(result.brier_after, 6),
        "ece_before": round(expected_calibration_error(labels, scores), 6),
        "ece_after": round(expected_calibration_error(labels, calibrated), 6),
        "oof": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in oof.items()},
        "degraded": bool(result.degraded),
        "note": result.note,
        "label_statement": LABEL_STATEMENT,
    }
    return calibrator, report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_calibrator(
    directory: Path,
    calibrator: Calibrator,
    report: dict[str, Any],
    *,
    demo: bool,
    checkpoint_identity: dict[str, Any],
    temperature_refit: dict[str, Any] | None = None,
) -> Path:
    """Write the state and the report into a checkpoint directory. Returns the state path.

    The state file also holds ``demo``, the ``fit_range`` and the identity of the
    checkpoint that the calibrator was fitted for. ``Calibrator.from_state`` ignores
    those keys.
    """
    directory = Path(directory)
    report = {**report, "demo": bool(demo), "checkpoint_identity": checkpoint_identity}
    extra: dict[str, Any] = {}
    if temperature_refit is not None:
        report[TEMPERATURE_REFIT_KEY] = temperature_refit
        extra[TEMPERATURE_REFIT_KEY] = temperature_refit
    _write_json(directory / REPORT_FILE, report)
    state_path = directory / CALIBRATOR_FILE
    _write_json(
        state_path,
        {
            **calibrator.state(),
            "demo": bool(demo),
            "fit_range": report["fit_range"],
            "checkpoint_identity": checkpoint_identity,
            **extra,
        },
    )
    return state_path


def load_calibrator(directory: Path, *, temperature_refit: bool = False) -> LoadedCalibrator:
    """Read the calibrator beside a checkpoint, or raise :class:`LayaCalibratorError`.

    A file that cannot be read or parsed, or that lacks a field, is
    ``calibrator_unreadable``. A state that is not fitted, that has method ``none``,
    that is not monotonic or that has fewer than ``MIN_DISTINCT_OUTPUTS`` distinct values
    is ``calibrator_rejected``. A file marked ``demo`` is ``calibrator_demo``. A file
    fitted for other weights than the ones in ``directory`` is
    ``calibrator_checkpoint_mismatch``. With ``temperature_refit`` set, a state without a
    refitted temperature is ``calibrator_temperature_missing``. With it unset, a state with
    one is ``calibrator_temperature_unexpected``, because its scores were fitted at another
    temperature than the scorer applies. In each case serving it would put a number under
    the name of a probability that nothing supports.
    """
    directory = Path(directory)
    path = directory / CALIBRATOR_FILE
    if not path.is_file():
        raise LayaCalibratorError(MISSING, f"{path} does not exist")
    try:
        raw = path.read_bytes()
        state = json.loads(raw.decode("utf-8"))
        if not isinstance(state, dict):
            raise ValueError("the state is not a JSON object")
        fitted = state.get("fitted")
        demo = state.get("demo")
        if not isinstance(fitted, bool):
            raise ValueError("`fitted` is not a boolean")
        if not isinstance(demo, bool):
            raise ValueError("`demo` is not a boolean")
        identity = state.get("checkpoint_identity")
        if not isinstance(identity, dict) or not isinstance(identity.get("sha256"), str):
            raise ValueError("`checkpoint_identity` is missing or has no sha256")
        fit_range = state.get("fit_range")
        if (
            not isinstance(fit_range, list)
            or len(fit_range) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in fit_range)
            or not all(np.isfinite(fit_range))
            or fit_range[0] > fit_range[1]
        ):
            raise ValueError("`fit_range` is not a finite [min, max] pair")
        method = state.get("method")
        calibrator = None
        if fitted and method in {"isotonic", "platt"}:
            calibrator = Calibrator.from_state(state)
            calibrator.transform(np.array([0.0, 0.5, 1.0]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LayaCalibratorError(UNREADABLE, f"{path} cannot be used: {exc}") from exc
    if calibrator is None:
        raise LayaCalibratorError(
            REJECTED, f"{path} holds method {method!r} fitted={fitted}, not a usable calibration"
        )
    if not _is_monotonic(calibrator):
        raise LayaCalibratorError(REJECTED, f"{path} does not preserve the score ranking")
    bounds = (float(fit_range[0]), float(fit_range[1]))
    distinct = distinct_outputs(calibrator, bounds)
    if distinct < MIN_DISTINCT_OUTPUTS:
        raise LayaCalibratorError(
            REJECTED,
            f"{path} gives {distinct} distinct values over its fit range, below the "
            f"minimum {MIN_DISTINCT_OUTPUTS}",
        )
    refit = state.get(TEMPERATURE_REFIT_KEY)
    value: float | None = None
    if temperature_refit:
        raw_value = refit.get("value") if isinstance(refit, dict) else None
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise LayaCalibratorError(
                TEMPERATURE_MISSING,
                f"{path} holds no refitted temperature, and laya.temperature_refit is on",
            )
        if not np.isfinite(raw_value) or raw_value <= 0:
            raise LayaCalibratorError(UNREADABLE, f"{path} holds an invalid temperature")
        value = float(raw_value)
    elif refit is not None:
        raise LayaCalibratorError(
            TEMPERATURE_UNEXPECTED,
            f"{path} was fitted with a refitted temperature, and laya.temperature_refit is off",
        )
    if demo:
        raise LayaCalibratorError(DEMO, f"{path} was fitted on the demo fixtures")
    current = directory_identity(directory, ignore=CALIBRATOR_FILES)
    fitted_for = (identity.get("sha256"), identity.get("files"))
    if current is None or (current["sha256"], current["files"]) != fitted_for:
        raise LayaCalibratorError(
            CHECKPOINT_MISMATCH,
            f"{path} was fitted for a checkpoint with identity {identity.get('sha256')}, "
            f"and {directory} now has identity {(current or {}).get('sha256')}",
        )
    digest = hashlib.sha256(raw + b"|" + current["sha256"].encode("ascii")).hexdigest()[:8]
    return LoadedCalibrator(calibrator, str(method), digest, bounds, value)


def read_temperature_refit(directory: Path) -> float:
    """The refitted temperature that the calibrator state beside a checkpoint holds.

    Raises :class:`LayaCalibratorError` with ``calibrator_temperature_missing`` when the file or
    the value is absent. The scorer never falls back to the checkpoint temperature in that case.
    """
    path = Path(directory) / CALIBRATOR_FILE
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        value = state[TEMPERATURE_REFIT_KEY]["value"]
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("not a positive number")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LayaCalibratorError(
            TEMPERATURE_MISSING, f"{path} holds no usable refitted temperature: {exc}"
        ) from exc
    return float(value)
