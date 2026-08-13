"""Calibration: turn a decision function into a probability you can multiply.

Phase 4 fuses these scores into one risk number. That is only meaningful if each
input is a calibrated probability -- if ``misinfo_prob = 0.7`` genuinely means
"about 70% of records scored 0.7 are misinformation-like". Raw logits, SVM
decision values and uncalibrated tree ensembles all fail that badly and
differently, so multiplying them together produces a number with no
interpretation at all.

**Fit on a held-out validation split, never on the training data.** A model's
training-set probabilities are already overconfident by construction, and
calibrating on them bakes the overconfidence in.

**Report before and after.** A reliability diagram and a Brier score for the raw
model and the calibrated one. If calibration made things worse -- which happens
on small validation sets, where isotonic regression overfits the step function --
that is a finding to report, not to hide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

log = logging.getLogger(__name__)

#: Below this many validation rows, isotonic regression fits noise. Platt
#: scaling has two parameters and degrades far more gracefully.
MIN_ROWS_FOR_ISOTONIC = 200


@dataclass
class CalibrationResult:
    """A fitted calibrator plus the evidence that it helped (or did not)."""

    method: str
    brier_before: float
    brier_after: float
    reliability_before: dict[str, list[float]]
    reliability_after: dict[str, list[float]]
    n_calibration: int
    #: True when calibration made the Brier score worse. Reported, not hidden.
    degraded: bool = False
    note: str = ""

    @property
    def improvement(self) -> float:
        return self.brier_before - self.brier_after

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_calibration": self.n_calibration,
            "brier_before": round(self.brier_before, 4),
            "brier_after": round(self.brier_after, 4),
            "improvement": round(self.improvement, 4),
            "degraded": self.degraded,
            "note": self.note,
            "reliability_before": self.reliability_before,
            "reliability_after": self.reliability_after,
        }

    def summary(self) -> str:
        direction = "worsened" if self.degraded else "improved"
        return (
            f"{self.method} calibration on {self.n_calibration} rows: "
            f"Brier {self.brier_before:.4f} -> {self.brier_after:.4f} ({direction})"
        )


class Calibrator:
    """Isotonic or Platt, fitted on validation scores.

    Kept as a thin object rather than sklearn's ``CalibratedClassifierCV``
    because the models being calibrated are heterogeneous -- a fine-tuned
    transformer, an XGBoost ensemble -- and all this layer needs is
    ``score -> probability``.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "platt", "none"}:
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self._model: Any = None
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> CalibrationResult:
        scores = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels).ravel().astype(int)
        if len(scores) != len(labels):
            raise ValueError("scores and labels must be the same length")

        method = self.method
        note = ""
        if method == "isotonic" and len(scores) < MIN_ROWS_FOR_ISOTONIC:
            # Isotonic on a small validation set fits a step function to noise
            # and produces confident 0.0/1.0 outputs. Platt has two parameters
            # and fails gracefully.
            method = "platt"
            note = (
                f"fell back to Platt: {len(scores)} validation rows is below the "
                f"{MIN_ROWS_FOR_ISOTONIC}-row floor for isotonic regression"
            )
            log.warning("%s calibration: %s", self.method, note)

        brier_before = _safe_brier(labels, scores)
        reliability_before = reliability_curve(labels, scores)

        if method == "none":
            self._model = None
            self.fitted = True
            return CalibrationResult(
                "none", brier_before, brier_before, reliability_before,
                reliability_before, len(scores), note="calibration disabled",
            )

        if method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(scores, labels)
        else:
            self._model = LogisticRegression(C=1e10, solver="lbfgs")
            self._model.fit(scores.reshape(-1, 1), labels)

        self.method = method
        self.fitted = True
        calibrated = self.transform(scores)
        brier_after = _safe_brier(labels, calibrated)
        degraded = brier_after > brier_before

        if degraded:
            note = (note + "; " if note else "") + (
                "calibration increased the Brier score on its own fit data, which usually "
                "means the validation split is too small or the raw scores were already "
                "well calibrated"
            )
            log.warning(
                "%s calibration degraded Brier %.4f -> %.4f", method, brier_before, brier_after
            )

        return CalibrationResult(
            method=method,
            brier_before=brier_before,
            brier_after=brier_after,
            reliability_before=reliability_before,
            reliability_after=reliability_curve(labels, calibrated),
            n_calibration=len(scores),
            degraded=degraded,
            note=note,
        )

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in [0, 1]."""
        scores = np.asarray(scores, dtype=float).ravel()
        if not self.fitted:
            raise RuntimeError("calibrator has not been fitted")
        if self._model is None:
            return np.clip(scores, 0.0, 1.0)
        if self.method == "isotonic":
            return np.clip(self._model.predict(scores), 0.0, 1.0)
        return np.clip(self._model.predict_proba(scores.reshape(-1, 1))[:, 1], 0.0, 1.0)

    # --- persistence -----------------------------------------------------
    def state(self) -> dict[str, Any]:
        """Serializable state, stored beside the checkpoint.

        A calibrator that does not travel with its model is a calibrator that
        silently stops being applied after a redeploy.
        """
        if not self.fitted or self._model is None:
            return {"method": self.method, "fitted": self.fitted}
        if self.method == "isotonic":
            return {
                "method": "isotonic",
                "fitted": True,
                "x": self._model.X_thresholds_.tolist(),
                "y": self._model.y_thresholds_.tolist(),
            }
        return {
            "method": "platt",
            "fitted": True,
            "coef": self._model.coef_.tolist(),
            "intercept": self._model.intercept_.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Calibrator:
        calibrator = cls(state.get("method", "none"))
        if not state.get("fitted"):
            calibrator.fitted = True
            calibrator._model = None
            return calibrator
        if state["method"] == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(np.asarray(state["x"]), np.asarray(state["y"]))
            calibrator._model = model
        else:
            model = LogisticRegression()
            model.coef_ = np.asarray(state["coef"])
            model.intercept_ = np.asarray(state["intercept"])
            model.classes_ = np.array([0, 1])
            calibrator._model = model
        calibrator.fitted = True
        return calibrator


def reliability_curve(
    labels: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Observed frequency vs predicted probability, per bin.

    Plotted as the reliability diagram. Bins with no rows are dropped rather
    than plotted at zero -- an empty bin drawn at the origin makes a
    well-calibrated model look badly calibrated.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    index = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)

    predicted, observed, counts = [], [], []
    for b in range(n_bins):
        mask = index == b
        if not mask.any():
            continue
        predicted.append(round(float(scores[mask].mean()), 4))
        observed.append(round(float(labels[mask].mean()), 4))
        counts.append(int(mask.sum()))
    return {"predicted": predicted, "observed": observed, "counts": counts}


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed frequency."""
    curve = reliability_curve(labels, scores, n_bins)
    if not curve["counts"]:
        return float("nan")
    total = sum(curve["counts"])
    return float(
        sum(
            count * abs(p - o)
            for p, o, count in zip(
                curve["predicted"], curve["observed"], curve["counts"], strict=True
            )
        )
        / total
    )


def _safe_brier(labels: np.ndarray, scores: np.ndarray) -> float:
    """Brier score, tolerating scores outside [0, 1] by clipping.

    Clipping rather than refusing: an uncalibrated decision function legitimately
    lands outside the unit interval, and the "before" number is the whole point
    of the comparison.
    """
    return float(brier_score_loss(labels, np.clip(scores, 0.0, 1.0)))
