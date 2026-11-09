"""The stance module: headline-grouped splits, label orientation, the scoring stage."""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from modeling import config as C
from modeling.datasets.fnc1 import FNC1
from modeling.text.stance_clf import StanceClassifier, resolve_stance_labels
from modeling.training import stance_split

LABELS = ["agree", "disagree", "discuss", "unrelated"]


def _write_fnc1(root: Path) -> Path:
    """Few headlines, many bodies each, like the real corpus, plus test overlap."""
    root.mkdir(parents=True)
    body_text = "An article body with enough words to pass the length floor, number {}."
    train_headlines = [f"Train headline number {i} about a public matter" for i in range(12)]
    # The last train headline also occurs in the test file.
    shared = "Shared headline that occurs in both files"
    train_headlines[-1] = shared
    with (root / "train_stances.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Headline", "Body ID", "Stance"])
        for h, headline in enumerate(train_headlines):
            for k in range(8):
                writer.writerow([headline, h * 8 + k, LABELS[(h + k) % 4]])
    with (root / "train_bodies.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Body ID", "articleBody"])
        for b in range(96):
            writer.writerow([b, body_text.format(b)])
    test_headlines = [shared] + [f"Test headline number {i} about a matter" for i in range(3)]
    with (root / "competition_test_stances.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Headline", "Body ID", "Stance"])
        for h, headline in enumerate(test_headlines):
            for k in range(5):
                writer.writerow([headline, h * 5 + k, LABELS[(h + k) % 4]])
    with (root / "competition_test_bodies.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Body ID", "articleBody"])
        for b in range(20):
            writer.writerow([b, f"A held-out article body with enough words, number {b}."])
    return root


def test_stance_split_has_no_headline_on_both_sides(tmp_path):
    root = _write_fnc1(tmp_path / "fnc")
    frame = FNC1().load(root).frame
    work, split, description, notes = stance_split(frame, C.get_settings())

    sides = {
        name: set(work["claim_id"].iloc[idx])
        for name, idx in (("train", split.train), ("val", split.val), ("test", split.test))
    }
    assert sides["val"], "the validation side must not be empty"
    assert not sides["train"] & sides["val"]
    assert not sides["train"] & sides["test"]
    assert not sides["val"] & sides["test"]
    assert "headline" in description


def test_stance_split_keeps_the_competition_test_set_whole(tmp_path):
    root = _write_fnc1(tmp_path / "fnc")
    frame = FNC1().load(root).frame
    work, split, _, notes = stance_split(frame, C.get_settings())

    assert (work["official_split"].iloc[split.test] == "competition_test").all()
    assert len(split.test) == int((frame["official_split"] == "competition_test").sum())
    # The one shared headline is removed from train, never from test.
    assert any(n.startswith("purged_overlap_with_test: 8 ") for n in notes)
    assert split.dropped["purged_overlap_with_test"] == 8


# ---------------------------------------------------------------------------
# label orientation of a fine-tuned head
# ---------------------------------------------------------------------------
class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeModel:
    def __init__(self, id2label):
        self.config = _FakeConfig(id2label)

    def to(self, device):
        return self

    def eval(self):
        return self


def _install_fake_transformers(monkeypatch, id2label):
    """A stand-in for `transformers` that loads no weights."""
    module = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: object())
    module.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda path: _FakeModel(id2label)
    )
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_stance_head_is_resolved_by_name_not_by_position(monkeypatch, tmp_path):
    permuted = {0: "unrelated", 1: "disagree", 2: "agree", 3: "discuss"}
    assert resolve_stance_labels(permuted) == ["unrelated", "deny", "support", "discuss"]

    _install_fake_transformers(monkeypatch, permuted)
    classifier = StanceClassifier()
    assert classifier.load(tmp_path)

    # Logit 2 is the largest, and in this head position 2 means "agree".
    monkeypatch.setattr(
        classifier, "_raw_probabilities", lambda *a, **k: np.array([[0.1, 0.1, 0.7, 0.1]])
    )
    (prediction,) = classifier.predict([("claim", "post")])
    assert prediction.label == "support"
    assert prediction.confidence == pytest.approx(0.7)


@pytest.mark.parametrize(
    "id2label",
    [
        {},
        None,
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2", 3: "LABEL_3"},  # the Hub default
        {0: "support", 1: "deny", 2: "discuss"},  # a class is missing
        {0: "support", 1: "support", 2: "discuss", 3: "unrelated"},  # a class repeats
        {0: "entailment", 1: "neutral", 2: "contradiction", 3: "unrelated"},
        {0: "support", 1: "deny", 2: "discuss", 4: "unrelated"},  # a gap in the indices
    ],
)
def test_a_stance_head_with_unreadable_labels_is_refused(monkeypatch, tmp_path, id2label):
    assert resolve_stance_labels(id2label) is None

    _install_fake_transformers(monkeypatch, id2label)
    classifier = StanceClassifier()
    assert classifier.load(tmp_path) is False
    assert classifier.trained is False
    assert classifier.load_error == "label_orientation_unresolved"
