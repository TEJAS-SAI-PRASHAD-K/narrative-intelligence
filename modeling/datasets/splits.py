"""The only splitter in this codebase.

Nothing else may import a scikit-learn splitter. ``tests/test_splits.py``
enforces that with a source scan, because the failure this module exists to
prevent is invisible: a random post-level split puts the same story in train and
test, every metric goes up, and nothing looks wrong.

The unit of a split is never a post. It is:

============  =========================================
module        group key
============  =========================================
misinfo       claim / story id, and outlet for domain shift
stance        claim id
bot           account id (and botnet/campaign id where known)
deepfake      source video id
narrative     n/a -- unsupervised, no split
============  =========================================

Two things happen here in a fixed order, and the order matters:

1. **Dedupe, then split.** Near-duplicate rows that straddle the boundary are a
   leak even when the group ids differ, because the model can memorize the text.
2. **Split by group, then verify.** Every public function in this module returns
   a :class:`SplitResult` that has already asserted zero group overlap. A split
   that silently leaks is worse than a crash.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from modeling.config import get_settings

log = logging.getLogger(__name__)


class LeakageError(AssertionError):
    """Raised when a split would put one group on both sides of the boundary.

    Deliberately an ``AssertionError`` subclass: this is a programming error in
    the caller, not a recoverable runtime condition. Nothing should catch it.
    """


@dataclass(frozen=True)
class SplitResult:
    """Positional indices into the frame that was split, plus its provenance.

    Indices, not slices of the frame: the caller usually wants to index several
    parallel arrays (features, labels, groups, texts) with the same split.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    group_col: str
    n_groups: int
    seed: int
    #: Rows dropped as near-duplicates before splitting, by reason.
    dropped: dict[str, int] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Verify at construction. A SplitResult that exists is a SplitResult
        # that has been checked -- callers never have to remember to check.
        overlap_tv = set(self.train.tolist()) & set(self.val.tolist())
        overlap_tt = set(self.train.tolist()) & set(self.test.tolist())
        overlap_vt = set(self.val.tolist()) & set(self.test.tolist())
        if overlap_tv or overlap_tt or overlap_vt:
            raise LeakageError(
                "split index sets overlap: "
                f"train/val={len(overlap_tv)}, train/test={len(overlap_tt)}, "
                f"val/test={len(overlap_vt)}"
            )

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def describe(self) -> str:
        """The phrase that must accompany every reported metric."""
        return (
            f"grouped by {self.group_col} "
            f"({self.n_groups} groups; "
            f"train/val/test = {len(self.train)}/{len(self.val)}/{len(self.test)}; "
            f"seed={self.seed})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_col": self.group_col,
            "n_groups": self.n_groups,
            "seed": self.seed,
            "sizes": self.sizes,
            "dropped": dict(self.dropped),
            "notes": dict(self.notes),
        }


# ---------------------------------------------------------------------------
# leakage verification
# ---------------------------------------------------------------------------
def assert_no_group_leakage(
    groups: Sequence[Any] | pd.Series | np.ndarray,
    *splits: Sequence[int] | np.ndarray,
    names: Sequence[str] | None = None,
) -> None:
    """Raise :class:`LeakageError` if any group appears in two splits.

    This is the assertion the whole project rests on. It is called inside every
    splitter here, and again from ``tests/test_splits.py`` against a deliberately
    leaky post-level split to prove it actually fires.
    """
    group_array = np.asarray(pd.Series(groups).to_numpy(), dtype=object)
    labels = list(names) if names else [f"split{i}" for i in range(len(splits))]
    seen: list[set[Any]] = [set(group_array[np.asarray(idx, dtype=int)].tolist()) for idx in splits]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            shared = seen[i] & seen[j]
            if shared:
                sample = sorted(str(s) for s in list(shared)[:5])
                raise LeakageError(
                    f"{len(shared)} group(s) appear in both {labels[i]} and {labels[j]} "
                    f"(e.g. {sample}). Split by group, not by row."
                )


def leakage_report(
    frame: pd.DataFrame,
    split: SplitResult,
    *,
    text_col: str | None = None,
) -> dict[str, Any]:
    """Post-hoc audit of a split: group overlap and exact-text overlap.

    Exact-text overlap is the second-order leak. Two rows can carry different
    claim ids and still be the same sentence -- a syndicated wire story
    republished under two outlet ids is the canonical case.
    """
    groups = frame[split.group_col]
    report: dict[str, Any] = {"group_col": split.group_col, "group_overlap": 0, "text_overlap": 0}
    try:
        assert_no_group_leakage(groups, split.train, split.val, split.test,
                                names=("train", "val", "test"))
    except LeakageError as exc:
        report["group_overlap"] = str(exc)
    if text_col and text_col in frame.columns:
        norm = frame[text_col].astype(str).str.strip().str.lower()
        train_texts = set(norm.iloc[split.train].tolist())
        held = set(norm.iloc[split.val].tolist()) | set(norm.iloc[split.test].tolist())
        report["text_overlap"] = len(train_texts & held)
    return report


# ---------------------------------------------------------------------------
# dedupe (always before splitting)
# ---------------------------------------------------------------------------
def dedupe_near_duplicates(
    frame: pd.DataFrame,
    *,
    text_col: str = "text",
    simhash_col: str | None = "simhash",
    hamming_threshold: int = 3,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse duplicate and near-duplicate rows *before* splitting.

    Exact normalized-text duplicates go first (cheap, always correct). Then, if
    a simhash column exists, rows within ``hamming_threshold`` bits of an
    already-kept row are dropped.

    The simhash pass is bucketed by 16-bit prefix rather than compared pairwise:
    an all-pairs scan over the corpus is O(n^2) and this function sits on the
    hot path of every training run. Bucketing can miss a near-duplicate pair
    whose prefixes differ; that is a false negative in *dedupe*, which costs a
    little redundancy, not a leak in the split, which would cost the result.

    Returns the surviving frame (index reset) and a count per reason code.
    """
    dropped: dict[str, int] = {}
    work = frame.copy()
    before = len(work)

    if text_col in work.columns:
        norm = (
            work[text_col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        )
        work = work.loc[~norm.duplicated(keep=keep)]
        dropped["exact_text_duplicate"] = before - len(work)

    if simhash_col and simhash_col in work.columns:
        before_sim = len(work)
        keep_mask = np.ones(len(work), dtype=bool)
        values = work[simhash_col].to_numpy()
        buckets: dict[int, list[int]] = {}
        for pos, raw in enumerate(values):
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            value = int(raw)
            # 16-bit prefix bucket: candidates must agree on the high bits.
            buckets.setdefault(value >> 48, []).append(pos)
        for members in buckets.values():
            kept_in_bucket: list[int] = []
            for pos in members:
                value = int(values[pos])
                if any(
                    bin(value ^ int(values[other])).count("1") <= hamming_threshold
                    for other in kept_in_bucket
                ):
                    keep_mask[pos] = False
                else:
                    kept_in_bucket.append(pos)
        work = work.loc[keep_mask]
        dropped["near_duplicate_simhash"] = before_sim - len(work)

    total = before - len(work)
    if total:
        log.info(
            "dedupe removed %d/%d rows before splitting (%s)",
            total,
            before,
            ", ".join(f"{k}={v}" for k, v in dropped.items() if v),
        )
    return work.reset_index(drop=True), dropped


# ---------------------------------------------------------------------------
# the splitters
# ---------------------------------------------------------------------------
def group_train_val_test(
    frame: pd.DataFrame,
    *,
    group_col: str,
    label_col: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int | None = None,
    dedupe: bool = True,
    text_col: str = "text",
) -> tuple[pd.DataFrame, SplitResult]:
    """Fixed grouped train/val/test split. Used by the text and media modules.

    ``val_size`` is a fraction of the *whole* frame, not of the post-test
    remainder -- the arithmetic that surprises people is done here once.

    Returns the (possibly deduped) frame alongside the split, because the split
    indices are positional into *that* frame, not the caller's original.
    """
    settings = get_settings()
    resolved_seed = seed if seed is not None else settings.seed

    dropped: dict[str, int] = {}
    work = frame.reset_index(drop=True)
    if dedupe:
        work, dropped = dedupe_near_duplicates(work, text_col=text_col)

    if group_col not in work.columns:
        raise KeyError(
            f"group column {group_col!r} not in frame (have: {list(work.columns)[:12]}...). "
            "A split without a group column is a post-level split; refusing."
        )

    groups = work[group_col].astype(str)
    n_groups = groups.nunique()
    if n_groups < 3:
        raise ValueError(
            f"only {n_groups} distinct {group_col} values; a grouped 3-way split is meaningless. "
            "Either widen the data or state honestly that this module was not evaluated."
        )

    indices = np.arange(len(work))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=resolved_seed)
    trainval_idx, test_idx = next(gss.split(indices, groups=groups))

    # val_size is relative to the whole frame; convert to a fraction of trainval.
    remaining = 1.0 - test_size
    val_fraction = min(0.9, max(1e-6, val_size / remaining)) if remaining > 0 else 0.0
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=resolved_seed + 1)
    sub_train, sub_val = next(
        gss_val.split(trainval_idx, groups=groups.iloc[trainval_idx])
    )
    train_idx = trainval_idx[sub_train]
    val_idx = trainval_idx[sub_val]

    assert_no_group_leakage(groups, train_idx, val_idx, test_idx, names=("train", "val", "test"))

    notes: dict[str, Any] = {"strategy": "GroupShuffleSplit x2"}
    if label_col and label_col in work.columns:
        notes["label_balance"] = {
            name: work[label_col].iloc[idx].value_counts(normalize=True).round(3).to_dict()
            for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))
        }

    result = SplitResult(
        train=train_idx,
        val=val_idx,
        test=test_idx,
        group_col=group_col,
        n_groups=int(n_groups),
        seed=resolved_seed,
        dropped=dropped,
        notes=notes,
    )
    log.info("split %s", result.describe())
    return work, result


def grouped_kfold(
    frame: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    n_splits: int = 5,
    seed: int | None = None,
    dedupe: bool = False,
    text_col: str = "text",
) -> tuple[pd.DataFrame, list[SplitResult]]:
    """Stratified *grouped* K-fold. Used by the bot classifier.

    Stratified because the labeled account sets are imbalanced; grouped because
    Cresci-2017's accounts come in campaigns and TwiBot's come in follow
    neighbourhoods, and splitting inside a campaign leaks the campaign's
    signature into the test fold.

    Each fold's ``val`` is empty: with 5-fold CV the fold's test set is the
    held-out estimate, and carving a third slice out of an already-small labeled
    set costs more than it buys. Calibration for CV is fitted out-of-fold; see
    ``modeling/eval/calibrate.py``.
    """
    settings = get_settings()
    resolved_seed = seed if seed is not None else settings.seed

    dropped: dict[str, int] = {}
    work = frame.reset_index(drop=True)
    if dedupe:
        work, dropped = dedupe_near_duplicates(work, text_col=text_col)

    for col in (group_col, label_col):
        if col not in work.columns:
            raise KeyError(f"column {col!r} not in frame; refusing to fold without it")

    groups = work[group_col].astype(str)
    labels = work[label_col]
    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"{n_groups} groups < {n_splits} folds. Reduce n_splits or report that this "
            "dataset is too small for grouped CV -- do not fall back to a row-level fold."
        )

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=resolved_seed)
    folds: list[SplitResult] = []
    indices = np.arange(len(work))
    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(indices, labels, groups=groups)):
        assert_no_group_leakage(
            groups, train_idx, test_idx, names=(f"fold{fold_i}_train", f"fold{fold_i}_test")
        )
        folds.append(
            SplitResult(
                train=train_idx,
                val=np.array([], dtype=int),
                test=test_idx,
                group_col=group_col,
                n_groups=int(n_groups),
                seed=resolved_seed,
                dropped=dropped,
                notes={"strategy": "StratifiedGroupKFold", "fold": fold_i, "n_splits": n_splits},
            )
        )
    log.info(
        "%d-fold grouped CV on %d rows / %d %s groups (seed=%d)",
        n_splits,
        len(work),
        n_groups,
        group_col,
        resolved_seed,
    )
    return work, folds


def domain_holdout(
    frame: pd.DataFrame,
    *,
    domain_col: str,
    held_out: str | Sequence[str],
    group_col: str | None = None,
) -> tuple[pd.DataFrame, SplitResult]:
    """Train on every domain except ``held_out``; test only on ``held_out``.

    This is how the honest numbers get produced: FakeNewsNet PolitiFact ->
    GossipCop, FF++ three manipulation methods -> the fourth. In-domain F1
    flatters; cross-domain F1 is what a reviewer should be shown.

    ``val`` is carved out of the training domains, so the held-out domain stays
    genuinely unseen until test time.
    """
    settings = get_settings()
    work = frame.reset_index(drop=True)
    if domain_col not in work.columns:
        raise KeyError(f"domain column {domain_col!r} not in frame")

    wanted = {held_out} if isinstance(held_out, str) else set(held_out)
    present = set(work[domain_col].astype(str).unique())
    missing = wanted - present
    if missing:
        raise ValueError(
            f"held-out domain(s) {sorted(missing)} not present; have {sorted(present)}"
        )

    is_test = work[domain_col].astype(str).isin(wanted).to_numpy()
    test_idx = np.flatnonzero(is_test)
    trainval_idx = np.flatnonzero(~is_test)
    if len(trainval_idx) == 0:
        raise ValueError("holding out that domain leaves no training data")

    group_key = group_col or domain_col
    key_col = group_key if group_key in work.columns else domain_col
    groups = work[key_col].astype(str)
    sub_groups = groups.iloc[trainval_idx]
    if sub_groups.nunique() >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=settings.seed)
        sub_train, sub_val = next(gss.split(trainval_idx, groups=sub_groups))
        train_idx, val_idx = trainval_idx[sub_train], trainval_idx[sub_val]
    else:  # pragma: no cover - degenerate single-group training domain
        train_idx, val_idx = trainval_idx, np.array([], dtype=int)

    result = SplitResult(
        train=train_idx,
        val=val_idx,
        test=test_idx,
        group_col=domain_col,
        n_groups=int(work[domain_col].nunique()),
        seed=settings.seed,
        notes={"strategy": "domain_holdout", "held_out": sorted(wanted)},
    )
    log.info("domain holdout on %s=%s: %s", domain_col, sorted(wanted), result.sizes)
    return work, result


def iter_folds(
    frame: pd.DataFrame, folds: list[SplitResult]
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, SplitResult]]:
    """Convenience: yield ``(train_frame, test_frame, split)`` per fold."""
    for split in folds:
        yield frame.iloc[split.train], frame.iloc[split.test], split
