"""The stance module: headline-grouped splits, label orientation, the scoring stage."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from modeling import config as C
from modeling.datasets.fnc1 import FNC1
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
    test_headlines = [shared] + [f"Test headline number {i} about a public matter" for i in range(3)]
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
