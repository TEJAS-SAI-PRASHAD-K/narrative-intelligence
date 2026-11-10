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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from modeling.checkpoint_identity import directory_identity
from modeling.eval.calibrate import Calibrator, expected_calibration_error

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

    @property
    def stamp(self) -> str:
        """The part of ``model_versions["laya"]`` that names the calibrator."""
        return f"{self.method}@{self.digest}"


def _is_monotonic(calibrator: Calibrator, slack: float = 1e-9) -> bool:
    grid = calibrator.transform(np.linspace(0.0, 1.0, 129))
    return bool(np.all(np.diff(grid) >= -slack))


def distinct_outputs(calibrator: Calibrator, fit_range: tuple[float, float]) -> int:
    """How many distinct values the calibrator gives over its fitted score range."""
    low, high = fit_range
    grid = np.linspace(low, high, _DISTINCT_GRID)
    return int(len(np.unique(np.round(calibrator.transform(grid), _DISTINCT_DECIMALS))))


def fit_calibrator(
    scores: np.ndarray, labels: np.ndarray, method: str = "isotonic"
) -> tuple[Calibrator, dict[str, Any]]:
    """Fit on validation scores and return the calibrator with its report.

    Raises :class:`LayaCalibratorError` with reason ``calibrator_rejected`` when the
    labels hold one class only, when the raw scores do not rank the labels above
    ``MIN_FIT_AUC``, when ``Calibrator`` rejects the fit because it inverted the ranking,
    or when the fitted map has fewer than ``MIN_DISTINCT_OUTPUTS`` distinct values.
    Nothing is returned in that case, so nothing can be written.
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
) -> Path:
    """Write the state and the report into a checkpoint directory. Returns the state path.

    The state file also holds ``demo``, the ``fit_range`` and the identity of the
    checkpoint that the calibrator was fitted for. ``Calibrator.from_state`` ignores
    those keys.
    """
    directory = Path(directory)
    report = {**report, "demo": bool(demo), "checkpoint_identity": checkpoint_identity}
    _write_json(directory / REPORT_FILE, report)
    state_path = directory / CALIBRATOR_FILE
    _write_json(
        state_path,
        {
            **calibrator.state(),
            "demo": bool(demo),
            "fit_range": report["fit_range"],
            "checkpoint_identity": checkpoint_identity,
        },
    )
    return state_path


def load_calibrator(directory: Path) -> LoadedCalibrator:
    """Read the calibrator beside a checkpoint, or raise :class:`LayaCalibratorError`.

    A file that cannot be read or parsed, or that lacks a field, is
    ``calibrator_unreadable``. A state that is not fitted, that has method ``none``,
    that is not monotonic or that has fewer than ``MIN_DISTINCT_OUTPUTS`` distinct values
    is ``calibrator_rejected``. A file marked ``demo`` is ``calibrator_demo``. A file
    fitted for other weights than the ones in ``directory`` is
    ``calibrator_checkpoint_mismatch``. In each case serving it would put a number under
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
    return LoadedCalibrator(calibrator, str(method), digest, bounds)
