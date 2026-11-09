"""The group-aware split on the real benchmarks. Skips when they are absent.

Run it against a checkout without ``data/benchmarks`` by pointing ``DATA_DIR`` at
a data directory that has them.
"""

from __future__ import annotations

import pytest

from modeling.config import get_settings
from modeling.datasets import get_dataset
from modeling.datasets.splits import group_train_val_test
from modeling.text.misinfo_clf import build_training_frame

KEYS = ("liar", "fakenewsnet", "coaid")


def test_misinfo_split_is_the_documented_one():
    settings = get_settings()
    missing = [k for k in KEYS if not get_dataset(k).available()]
    if missing:
        pytest.skip(f"benchmarks not on disk under {settings.benchmarks_dir}: {missing}")

    loaded = {key: get_dataset(key).load().frame for key in KEYS}
    frame = build_training_frame(loaded)
    work, split = group_train_val_test(
        frame, group_col="group_id", label_col="label", seed=settings.seed
    )

    assert split.describe() == (
        "grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)"
    )
    assert round(float(work.iloc[split.test]["label"].mean()), 4) == 0.2991
