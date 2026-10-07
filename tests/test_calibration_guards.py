"""Calibration and checkpoint-availability guards.

Each test here covers a failure that is invisible in aggregate: the system
keeps producing numbers, the numbers look plausible, and they are wrong. That
is the only reason these guards exist, so they are worth pinning down.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from modeling.eval.calibrate import Calibrator


def _anti_correlated(n: int = 240, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Scores that run *backwards*: high score, mostly negative label.

    This is what a weak detector on an imbalanced set looks like to a Platt
    fit, and it is what makes the fit choose a negative slope.
    """
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, 0.8, size=n)
    # Positives get lower scores than negatives -- the inversion.
    scores = np.where(labels == 1, rng.uniform(0.0, 0.5, n), rng.uniform(0.5, 1.0, n))
    return scores, labels


def test_platt_fit_that_inverts_the_ranking_is_rejected():
    """A calibrator may rescale confidence; it may never reorder predictions.

    Platt is a logistic fit with a free sign, so on an anti-correlated signal
    it settles on a negative slope and silently inverts the model. Brier does
    not catch it -- predicting near the base rate improves Brier even while the
    ranking is destroyed -- so ranking is checked directly.
    """
    scores, labels = _anti_correlated()
    calibrator = Calibrator("platt")
    result = calibrator.fit(scores, labels)

    assert result.method == "none", "an inverting fit must not be kept"
    assert result.degraded is True
    assert "REJECTED" in result.note

    # Rejection means raw scores pass through, so the ranking is preserved
    # exactly -- including the fact that it is a *bad* ranking. Hiding a weak
    # signal behind a flipped sign is the thing being prevented.
    passed_through = calibrator.transform(scores)
    assert roc_auc_score(labels, passed_through) == pytest.approx(
        roc_auc_score(labels, scores)
    )


def test_a_well_oriented_platt_fit_is_kept():
    """The guard must not fire on an ordinary, correctly-oriented fit."""
    rng = np.random.default_rng(11)
    labels = rng.binomial(1, 0.5, size=240)
    scores = np.clip(
        np.where(labels == 1, rng.normal(0.7, 0.15, 240), rng.normal(0.3, 0.15, 240)), 0, 1
    )
    result = Calibrator("platt").fit(scores, labels)
    assert result.method == "platt"
    assert "REJECTED" not in result.note


def test_isotonic_is_monotonic_by_construction():
    """Isotonic cannot invert, so it must survive the guard on the same data."""
    scores, labels = _anti_correlated(n=400)
    result = Calibrator("isotonic").fit(scores, labels)
    assert result.method == "isotonic"
    assert "REJECTED" not in result.note


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError):
        Calibrator("quantile")


# --- checkpoint availability -------------------------------------------------
def test_metadata_only_directory_is_not_a_mounted_model(tmp_path):
    """A checkpoint whose tensors are gone must not report ready.

    Every checkpoint this project writes leaves JSON behind -- config,
    tokenizer, calibrator, registry. Counting those as weights makes a
    directory whose weights were deleted or never synced look mounted; it then
    fails at the first request instead of at the probe.
    """
    from nlp.availability import _has_weights

    directory = tmp_path / "misinfo" / "v0.1.0"
    directory.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "calibrator.json", "registry.json"):
        (directory / name).write_text("{}", encoding="utf-8")

    assert _has_weights(directory) is False

    (directory / "model.safetensors").write_bytes(b"\x00")
    assert _has_weights(directory) is True


@pytest.mark.parametrize(
    "weight_file",
    ["model.safetensors", "pytorch_model.bin", "model.pt", "estimator.pkl"],
)
def test_every_serialization_this_project_writes_counts_as_weights(tmp_path, weight_file):
    """The bot module writes a pickle and deepfake writes a .pt; both count."""
    from nlp.availability import _has_weights

    directory = tmp_path / "module" / "v1"
    directory.mkdir(parents=True)
    (directory / weight_file).write_bytes(b"\x00")
    assert _has_weights(directory) is True


def test_fixture_trained_checkpoint_is_refused(tmp_path):
    """A demo checkpoint has real weights, so only its metadata gives it away.

    Serving a model fitted on ~50 shape-faithful, value-meaningless rows is the
    "confident garbage" the registry refuses to degrade into, and it is
    indistinguishable from a real checkpoint on the filesystem alone.
    """
    from nlp.availability import _is_demo_checkpoint

    directory = tmp_path / "stance" / "v0.2.0"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"\x00")

    # No metadata at all: older checkpoints predate the field, so not-demo.
    assert _is_demo_checkpoint(directory) is False

    (directory / "registry.json").write_text(json.dumps({"is_demo": True}), encoding="utf-8")
    assert _is_demo_checkpoint(directory) is True

    (directory / "registry.json").write_text(json.dumps({"is_demo": False}), encoding="utf-8")
    assert _is_demo_checkpoint(directory) is False

    # Unreadable metadata must not be treated as demo: that would take a real
    # model offline over a corrupt side-file.
    (directory / "registry.json").write_text("{not json", encoding="utf-8")
    assert _is_demo_checkpoint(directory) is False
