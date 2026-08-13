"""Leakage tests.

These are the highest-value tests in the repo. Everything else in Phase 2
reports a number; these decide whether the number means anything.

The tests come in three kinds:

1. **The detector fires.** A deliberately leaky post-level split must raise.
   A leakage assertion that has never been seen to fail is not evidence.
2. **The splitters don't leak.** Every public splitter, on adversarial data
   (one huge group, singleton groups, heavy class imbalance).
3. **Nobody bypasses the splitter.** A source scan asserting that no module
   outside ``datasets/splits.py`` imports a scikit-learn splitter.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling.datasets import splits as S

REPO_ROOT = Path(__file__).resolve().parent.parent


def _spread(value: int) -> int:
    """A simhash that is genuinely far from its neighbours.

    ``value << 40`` looks like a distinct hash and is not: consecutive stories
    then differ by one or two bits and the near-duplicate pass eats them all.
    Fixtures have to be adversarial in the right direction.
    """
    return (value * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)


def make_frame(n_groups: int = 30, per_group: int = 6, seed: int = 0) -> pd.DataFrame:
    """A corpus where each story appears as several near-identical posts.

    This is the shape that makes post-level splitting look fine and be wrong.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = int(g % 2)
        for k in range(per_group):
            rows.append(
                {
                    "text": f"story {g} variant {k} " + " ".join(
                        rng.choice(["alpha", "beta", "gamma", "delta"], size=5)
                    ),
                    "claim_id": f"claim-{g}",
                    "outlet": f"outlet-{g % 4}",
                    "label": label,
                    "simhash": _spread(g * 31 + k),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. the detector fires
# ---------------------------------------------------------------------------
def test_post_level_split_is_detected_as_leakage():
    """The canonical mistake: a random row-level split of grouped data.

    If this test ever stops raising, every metric in artifacts/ is suspect.
    """
    frame = make_frame()
    rng = np.random.default_rng(1)
    order = rng.permutation(len(frame))
    train, test = order[: int(0.8 * len(order))], order[int(0.8 * len(order)) :]

    with pytest.raises(S.LeakageError) as excinfo:
        S.assert_no_group_leakage(frame["claim_id"], train, test, names=("train", "test"))
    assert "claim-" in str(excinfo.value)
    assert "Split by group, not by row" in str(excinfo.value)


def test_leakage_detector_passes_on_a_clean_split():
    """The converse: the detector must not cry wolf on a correct split."""
    frame = make_frame()
    groups = sorted(frame["claim_id"].unique())
    train_groups = set(groups[:20])
    train = np.flatnonzero(frame["claim_id"].isin(train_groups).to_numpy())
    test = np.flatnonzero(~frame["claim_id"].isin(train_groups).to_numpy())
    S.assert_no_group_leakage(frame["claim_id"], train, test, names=("train", "test"))


def test_split_result_rejects_overlapping_index_sets():
    """A SplitResult that exists has been verified; construction is the check."""
    with pytest.raises(S.LeakageError):
        S.SplitResult(
            train=np.array([0, 1, 2]),
            val=np.array([2, 3]),  # 2 is in both
            test=np.array([4]),
            group_col="claim_id",
            n_groups=3,
            seed=0,
        )


def test_video_frame_split_leakage_is_detected():
    """Deepfake-specific: frames of one source video must not straddle the split.

    This is the exact mechanism behind implausible 99% deepfake accuracies.
    """
    frames = pd.DataFrame(
        [
            {
                "frame_id": f"{video}-f{i:03d}",
                "video_id": video,
                "label": int(video.endswith("fake")),
            }
            for video in ("v001_real", "v001_fake", "v002_real", "v002_fake")
            for i in range(50)
        ]
    )
    rng = np.random.default_rng(7)
    order = rng.permutation(len(frames))
    train, test = order[:150], order[150:]
    with pytest.raises(S.LeakageError):
        S.assert_no_group_leakage(frames["video_id"], train, test, names=("train", "test"))


# ---------------------------------------------------------------------------
# 2. the splitters don't leak
# ---------------------------------------------------------------------------
def test_group_train_val_test_has_no_group_overlap():
    frame = make_frame()
    work, split = S.group_train_val_test(
        frame, group_col="claim_id", label_col="label", dedupe=False
    )
    groups = work["claim_id"]
    S.assert_no_group_leakage(groups, split.train, split.val, split.test)
    assert len(split.train) and len(split.val) and len(split.test)
    assert len(split.train) + len(split.val) + len(split.test) == len(work)


def test_group_train_val_test_is_deterministic_under_the_seed():
    frame = make_frame()
    _, a = S.group_train_val_test(frame, group_col="claim_id", seed=99, dedupe=False)
    _, b = S.group_train_val_test(frame, group_col="claim_id", seed=99, dedupe=False)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)
    _, c = S.group_train_val_test(frame, group_col="claim_id", seed=100, dedupe=False)
    assert not np.array_equal(a.test, c.test)


def test_split_refuses_a_missing_group_column():
    frame = make_frame().drop(columns=["claim_id"])
    with pytest.raises(KeyError, match="refusing"):
        S.group_train_val_test(frame, group_col="claim_id", dedupe=False)


def test_split_refuses_too_few_groups():
    """Two claims cannot support a 3-way grouped split; say so rather than
    silently producing an empty test set."""
    frame = make_frame(n_groups=2, per_group=50)
    with pytest.raises(ValueError, match="grouped 3-way split is meaningless"):
        S.group_train_val_test(frame, group_col="claim_id", dedupe=False)


def test_grouped_kfold_never_shares_an_account_between_folds():
    accounts = pd.DataFrame(
        [
            {
                "account_id": f"acct-{a}",
                "campaign": f"camp-{a % 5}",
                "label": int(a % 5 == 0),
                "x": a,
            }
            for a in range(60)
            for _ in range(3)
        ]
    )
    work, folds = S.grouped_kfold(accounts, group_col="account_id", label_col="label", n_splits=5)
    assert len(folds) == 5
    for fold in folds:
        S.assert_no_group_leakage(work["account_id"], fold.train, fold.test)
    # Every row is tested exactly once across the folds.
    tested = np.concatenate([f.test for f in folds])
    assert sorted(tested.tolist()) == list(range(len(work)))


def test_grouped_kfold_by_campaign_isolates_botnets():
    """Cresci-2017's campaign structure: grouping by account is not enough when
    a whole botnet shares a content template."""
    accounts = pd.DataFrame(
        [
            {"account_id": f"acct-{c}-{a}", "campaign": f"camp-{c}", "label": int(c < 3), "x": a}
            for c in range(10)
            for a in range(8)
        ]
    )
    work, folds = S.grouped_kfold(accounts, group_col="campaign", label_col="label", n_splits=5)
    for fold in folds:
        S.assert_no_group_leakage(work["campaign"], fold.train, fold.test)


def test_grouped_kfold_refuses_fewer_groups_than_folds():
    frame = pd.DataFrame({"g": ["a", "b", "c"] * 4, "label": [0, 1, 0] * 4})
    with pytest.raises(ValueError, match="too small for grouped CV"):
        S.grouped_kfold(frame, group_col="g", label_col="label", n_splits=5)


def test_domain_holdout_puts_the_whole_domain_in_test():
    frame = make_frame()
    work, split = S.domain_holdout(
        frame, domain_col="outlet", held_out="outlet-2", group_col="claim_id"
    )
    assert set(work["outlet"].iloc[split.test].unique()) == {"outlet-2"}
    assert "outlet-2" not in set(work["outlet"].iloc[split.train].unique())
    assert "outlet-2" not in set(work["outlet"].iloc[split.val].unique())


def test_domain_holdout_rejects_an_absent_domain():
    with pytest.raises(ValueError, match="not present"):
        S.domain_holdout(make_frame(), domain_col="outlet", held_out="outlet-99")


# ---------------------------------------------------------------------------
# dedupe-before-split
# ---------------------------------------------------------------------------
def test_dedupe_removes_exact_duplicates_before_splitting():
    frame = pd.DataFrame(
        {
            "text": ["same story", "same story", "SAME  STORY ", "different"],
            "claim_id": ["a", "b", "c", "d"],
            "simhash": [1, 1, 1, 1 << 60],
        }
    )
    out, dropped = S.dedupe_near_duplicates(frame)
    assert len(out) == 2
    assert dropped["exact_text_duplicate"] == 2


def test_dedupe_removes_near_duplicates_by_simhash():
    base = 0xABCD_0000_0000_0000
    frame = pd.DataFrame(
        {
            "text": [f"variant {i}" for i in range(4)],
            "claim_id": list("abcd"),
            # three within 2 bits of each other, one far away
            "simhash": [base, base ^ 0b11, base ^ 0b101, base ^ 0xFFFF],
        }
    )
    out, dropped = S.dedupe_near_duplicates(frame, hamming_threshold=3)
    assert dropped["near_duplicate_simhash"] == 2
    assert len(out) == 2


def test_dedupe_tolerates_a_missing_simhash_column():
    frame = pd.DataFrame({"text": ["a", "b"], "claim_id": ["x", "y"]})
    out, dropped = S.dedupe_near_duplicates(frame)
    assert len(out) == 2
    assert "near_duplicate_simhash" not in dropped


def test_duplicate_text_does_not_straddle_the_split():
    """The second-order leak: same text, different claim ids.

    A syndicated wire story republished under two outlet ids has two group keys
    and one body. Dedupe before splitting is what stops it.
    """
    frame = pd.DataFrame(
        [
            {"text": f"story {g}", "claim_id": f"claim-{g}-{copy}", "simhash": _spread(g)}
            for g in range(20)
            for copy in range(3)
        ]
    )
    work, split = S.group_train_val_test(frame, group_col="claim_id", dedupe=True)
    report = S.leakage_report(work, split, text_col="text")
    assert report["text_overlap"] == 0
    assert report["group_overlap"] == 0


def test_leakage_report_flags_text_overlap_when_dedupe_is_skipped():
    frame = pd.DataFrame(
        [
            {
                "text": f"story {g}",
                "claim_id": f"claim-{g}-{copy}",
                "simhash": _spread(g * 7 + copy),
            }
            for g in range(20)
            for copy in range(3)
        ]
    )
    work, split = S.group_train_val_test(frame, group_col="claim_id", dedupe=False)
    report = S.leakage_report(work, split, text_col="text")
    assert report["group_overlap"] == 0  # groups are genuinely disjoint...
    assert report["text_overlap"] > 0  # ...and the text still leaks


# ---------------------------------------------------------------------------
# 3. nobody bypasses the splitter
# ---------------------------------------------------------------------------
SPLITTER_PATTERN = re.compile(
    r"\b(train_test_split|KFold|StratifiedKFold|ShuffleSplit|GroupShuffleSplit|"
    r"StratifiedGroupKFold|GroupKFold|TimeSeriesSplit)\b"
)


def test_no_module_outside_splits_imports_a_sklearn_splitter():
    """``datasets/splits.py`` is the only splitter. Enforced, not requested.

    The moment a module calls ``train_test_split`` directly, the group
    discipline is gone and no reviewer will notice from the metrics.
    """
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "modeling").rglob("*.py")):
        if path.name == "splits.py":
            continue
        source = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or "noqa: splitter" in stripped:
                continue
            if "sklearn.model_selection" in stripped or SPLITTER_PATTERN.search(stripped):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {stripped}")
    assert not offenders, (
        "these modules bypass modeling/datasets/splits.py:\n  " + "\n  ".join(offenders)
    )
