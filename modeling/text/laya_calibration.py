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

from modeling.eval.calibrate import Calibrator, expected_calibration_error

CALIBRATOR_FILE = "misinfo_prob_calibrator.json"
REPORT_FILE = "misinfo_prob_calibration_report.json"

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


class LayaCalibratorError(RuntimeError):
    """A calibrator cannot be fitted or used. ``reason`` is a counted reason code."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class LoadedCalibrator:
    calibrator: Calibrator
    method: str
    #: First 8 hex digits of the sha256 of the calibrator file.
    digest: str

    @property
    def stamp(self) -> str:
        """The part of ``model_versions["laya"]`` that names the calibrator."""
        return f"{self.method}@{self.digest}"


def _is_monotonic(calibrator: Calibrator, slack: float = 1e-9) -> bool:
    grid = calibrator.transform(np.linspace(0.0, 1.0, 129))
    return bool(np.all(np.diff(grid) >= -slack))


def fit_calibrator(
    scores: np.ndarray, labels: np.ndarray, method: str = "isotonic"
) -> tuple[Calibrator, dict[str, Any]]:
    """Fit on validation scores and return the calibrator with its report.

    Raises :class:`LayaCalibratorError` with reason ``calibrator_rejected`` when the
    labels hold one class only, or when ``Calibrator`` rejects the fit because it
    inverted the ranking. Nothing is returned in that case, so nothing can be written.
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

    calibrator = Calibrator(method)
    result = calibrator.fit(scores, labels)
    if result.method == "none":
        raise LayaCalibratorError(
            REJECTED, f"the calibrator rejected the fit and wrote nothing: {result.note}"
        )
    calibrated = calibrator.transform(scores)
    report = {
        "method": result.method,
        "requested_method": method,
        "rows": int(len(scores)),
        "positives": int(labels.sum()),
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


def write_calibrator(directory: Path, calibrator: Calibrator, report: dict[str, Any]) -> Path:
    """Write the state and the report into a checkpoint directory. Returns the state path."""
    directory = Path(directory)
    _write_json(directory / REPORT_FILE, report)
    state_path = directory / CALIBRATOR_FILE
    _write_json(state_path, calibrator.state())
    return state_path


def load_calibrator(directory: Path) -> LoadedCalibrator:
    """Read the calibrator beside a checkpoint, or raise :class:`LayaCalibratorError`.

    A state that is not fitted, that has method ``none``, or that is not monotonic
    is ``calibrator_rejected``: serving it would pass raw scores through under the
    name of a probability. A file that cannot be read or parsed is
    ``calibrator_unreadable``.
    """
    path = Path(directory) / CALIBRATOR_FILE
    if not path.is_file():
        raise LayaCalibratorError(MISSING, f"{path} does not exist")
    try:
        raw = path.read_bytes()
        state = json.loads(raw.decode("utf-8"))
        if not isinstance(state, dict):
            raise ValueError("the state is not a JSON object")
        method = state.get("method")
        fitted = bool(state.get("fitted"))
        if fitted and method in {"isotonic", "platt"}:
            calibrator = Calibrator.from_state(state)
            calibrator.transform(np.array([0.0, 0.5, 1.0]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LayaCalibratorError(UNREADABLE, f"{path} cannot be used: {exc}") from exc
    if not fitted or method not in {"isotonic", "platt"}:
        raise LayaCalibratorError(
            REJECTED, f"{path} holds method {method!r} fitted={fitted}, not a usable calibration"
        )
    if not _is_monotonic(calibrator):
        raise LayaCalibratorError(REJECTED, f"{path} does not preserve the score ranking")
    return LoadedCalibrator(calibrator, str(method), hashlib.sha256(raw).hexdigest()[:8])
