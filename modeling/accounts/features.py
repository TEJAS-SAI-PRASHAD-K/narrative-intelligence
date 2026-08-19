"""Account-level features. Pure functions, unit-tested, individually explainable.

Every feature has a docstring saying what signal it is supposed to capture. That
is not documentation hygiene: the dashboard shows "why is this account flagged"
straight from SHAP contributions over these names, so a feature nobody can
explain becomes an accusation nobody can defend.

**The cross-platform honesty problem, and what is done about it.**

Follower and following counts exist for Mastodon and for the Twitter benchmarks.
They do not exist for ConvoKit Reddit at all. Training a bot classifier on
TwiBot-22's forty Twitter features and then scoring Reddit accounts with twelve
of them present would be a model applied to a distribution it never saw.

So features are declared in tiers:

* ``UNIVERSAL`` -- computable from posts alone, on every platform.
* ``SOCIAL_GRAPH`` -- needs follower/following counts.
* ``THREADING`` -- needs ``parent_id``, so ConvoKit Reddit and YouTube but not
  the Kaggle-flat Reddit dump.

:func:`intersection_features` computes which tier is available on both the
training benchmark and the target corpus, and the classifier trains on that
intersection only. The tier actually used is recorded in the model card. This is
the "build the feature intersection first, then train" discipline, enforced in
code rather than remembered.

**Missingness is a feature, not a fill.** Phase 1's rule holds: `null` means "not
measurable on this platform", `0` means "measured zero". Every feature whose
source can be absent ships an ``*_is_missing`` indicator.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Computable from an author's posts alone. Available on every source.
UNIVERSAL = [
    "posts_per_day",
    "inter_post_mean_seconds",
    "inter_post_std_seconds",
    "inter_post_entropy",
    "hour_entropy",
    "burstiness",
    "longest_streak_days",
    "mean_text_length",
    "type_token_ratio",
    "self_similarity_mean",
    "duplicate_content_rate",
    "url_rate",
    "hashtag_rate",
    "mention_rate",
    "post_count",
    "active_days",
]

#: Needs follower/following counts. Mastodon and the Twitter benchmarks only.
SOCIAL_GRAPH = [
    "followers",
    "following",
    "follower_following_ratio",
    "posts_per_day_per_follower",
    "account_age_days",
    "posts_per_account_day",
    "post_count_is_missing",
    "followers_is_missing",
    "account_age_is_missing",
]

#: Needs threading. ConvoKit Reddit and YouTube comments; not Kaggle-flat Reddit.
THREADING = [
    "distinct_conversations",
    "reply_rate",
    "distinct_reply_targets",
    "reciprocity",
]

TIERS = {"universal": UNIVERSAL, "social_graph": SOCIAL_GRAPH, "threading": THREADING}


@dataclass
class FeatureMatrix:
    """Features plus the names and the tiers they came from."""

    matrix: np.ndarray
    names: list[str]
    account_ids: list[str]
    tiers_used: list[str]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, columns=self.names, index=self.account_ids)

    def subset(self, names: list[str]) -> FeatureMatrix:
        keep = [self.names.index(n) for n in names if n in self.names]
        return FeatureMatrix(
            matrix=self.matrix[:, keep],
            names=[self.names[i] for i in keep],
            account_ids=self.account_ids,
            tiers_used=self.tiers_used,
        )


# ---------------------------------------------------------------------------
# individual features
# ---------------------------------------------------------------------------
def posts_per_day(timestamps: pd.Series) -> float:
    """Raw posting rate. The crudest automation signal and still a useful one."""
    if len(timestamps) < 2:
        return float(len(timestamps))
    span = (timestamps.max() - timestamps.min()).total_seconds() / 86400
    return float(len(timestamps) / max(span, 1 / 24))


def inter_post_intervals(timestamps: pd.Series) -> np.ndarray:
    """Seconds between consecutive posts, ascending."""
    ordered = pd.Series(sorted(pd.to_datetime(timestamps, utc=True)))
    if len(ordered) < 2:
        return np.zeros(0)
    return (ordered.diff().dropna().dt.total_seconds()).to_numpy()


def interval_entropy(intervals: np.ndarray, bins: int = 12) -> float:
    """Shannon entropy of the inter-post interval distribution.

    A scheduler posting every 30 minutes has near-zero entropy; a human's
    intervals are spread across seconds, hours and days. Low entropy at high
    volume is the signature worth catching.
    """
    if len(intervals) < 2:
        return 0.0
    positive = intervals[intervals > 0]
    if not len(positive):
        return 0.0
    counts, _ = np.histogram(np.log1p(positive), bins=bins)
    return _entropy(counts)


def hour_of_day_entropy(timestamps: pd.Series) -> float:
    """Entropy over the 24 posting hours.

    **Bots post uniformly; humans sleep.** A perfectly flat hour distribution is
    the single most interpretable automation signal in this whole set, which is
    also why it is the one most likely to be gamed.
    """
    hours = pd.to_datetime(timestamps, utc=True).dt.hour
    counts = np.bincount(hours.to_numpy(dtype=int), minlength=24)
    return _entropy(counts)


def burstiness(intervals: np.ndarray) -> float:
    """Goh & Barabasi burstiness: ``(std - mean) / (std + mean)``, in [-1, 1].

    +1 is maximally bursty (everything in one clump), -1 is perfectly periodic,
    0 is Poisson. Both extremes are non-human in different ways: campaigns burst,
    schedulers tick.
    """
    if len(intervals) < 2:
        return 0.0
    mean, std = float(intervals.mean()), float(intervals.std())
    if mean + std == 0:
        return 0.0
    return float((std - mean) / (std + mean))


def longest_active_streak_days(timestamps: pd.Series) -> float:
    """Longest run of consecutive calendar days with at least one post.

    Sustained daily activity over weeks without a break is unusual for a person
    and ordinary for a service.
    """
    if not len(timestamps):
        return 0.0
    days = sorted({d.date() for d in pd.to_datetime(timestamps, utc=True)})
    best = run = 1
    # Deliberately offset by one, so strict=True would be wrong here: the two
    # sequences are meant to differ in length.
    for previous, current in zip(days[:-1], days[1:], strict=True):
        run = run + 1 if (current - previous).days == 1 else 1
        best = max(best, run)
    return float(best)


def type_token_ratio(texts: list[str]) -> float:
    """Distinct words over total words across an author's posts.

    Low values mean a small recycled vocabulary — template posting. Length-biased
    (long documents score lower), so it is only comparable between accounts with
    similar volume; the model sees ``post_count`` alongside it.
    """
    tokens = [t for text in texts for t in str(text).lower().split()]
    if not tokens:
        return 0.0
    return float(len(set(tokens)) / len(tokens))


def self_similarity(simhashes: list[int]) -> float:
    """Mean pairwise Hamming similarity among an author's own posts.

    Self-repetition. Capped at 200 posts sampled: this is O(n^2) per author and
    the estimate is stable long before then.
    """
    values = [int(v) for v in simhashes if v is not None and not pd.isna(v)]
    if len(values) < 2:
        return 0.0
    if len(values) > 200:
        values = values[:200]
    total = pairs = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += 64 - bin(values[i] ^ values[j]).count("1")
            pairs += 1
    return float(total / (pairs * 64)) if pairs else 0.0


def duplicate_content_rate(texts: list[str]) -> float:
    """Fraction of an author's posts that repeat an earlier one verbatim."""
    if not texts:
        return 0.0
    normalized = [" ".join(str(t).lower().split()) for t in texts]
    counts = Counter(normalized)
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeats / len(normalized))


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits, 0 when everything is in one bucket."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _rate(values: pd.Series, n: int) -> float:
    if not n:
        return 0.0
    return float(sum(len(v) if v is not None else 0 for v in values) / n)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build_features(
    records: pd.DataFrame,
    authors: pd.DataFrame | None = None,
    *,
    tiers: list[str] | None = None,
    reference_time: pd.Timestamp | None = None,
) -> FeatureMatrix:
    """Compute the requested feature tiers for every author in ``records``.

    Pure given its inputs, which is what makes it unit-testable and what lets
    the same code run over a benchmark and over this project's corpus. Training
    on features computed one way and scoring on features computed another is a
    silent distribution shift, and using one function for both is the only
    reliable guard against it.
    """
    wanted = tiers or ["universal"]
    unknown = set(wanted) - set(TIERS)
    if unknown:
        raise KeyError(f"unknown feature tier(s) {sorted(unknown)}; have {sorted(TIERS)}")

    work = records.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values(["author_id", "timestamp"])
    now = reference_time or work["timestamp"].max()

    author_lookup = (
        authors.set_index("author_id") if authors is not None and len(authors) else pd.DataFrame()
    )

    rows: list[dict[str, float]] = []
    account_ids: list[str] = []

    for author_id, group in work.groupby("author_id", sort=True):
        timestamps = group["timestamp"]
        texts = group["text"].fillna("").astype(str).tolist()
        intervals = inter_post_intervals(timestamps)
        n = len(group)
        feature: dict[str, float] = {}

        if "universal" in wanted:
            feature.update(
                {
                    "posts_per_day": posts_per_day(timestamps),
                    "inter_post_mean_seconds": float(intervals.mean()) if len(intervals) else 0.0,
                    "inter_post_std_seconds": float(intervals.std()) if len(intervals) else 0.0,
                    "inter_post_entropy": interval_entropy(intervals),
                    "hour_entropy": hour_of_day_entropy(timestamps),
                    "burstiness": burstiness(intervals),
                    "longest_streak_days": longest_active_streak_days(timestamps),
                    "mean_text_length": float(np.mean([len(t) for t in texts])) if texts else 0.0,
                    "type_token_ratio": type_token_ratio(texts),
                    "self_similarity_mean": self_similarity(
                        group["simhash"].tolist() if "simhash" in group else []
                    ),
                    "duplicate_content_rate": duplicate_content_rate(texts),
                    "url_rate": _rate(group.get("urls", pd.Series(dtype=object)), n),
                    "hashtag_rate": _rate(group.get("hashtags", pd.Series(dtype=object)), n),
                    "mention_rate": _rate(group.get("mentions", pd.Series(dtype=object)), n),
                    "post_count": float(n),
                    "active_days": float(len({d.date() for d in timestamps})),
                }
            )

        if "social_graph" in wanted:
            profile = (
                author_lookup.loc[author_id].to_dict()
                if author_id in author_lookup.index
                else {}
            )
            followers = _numeric(profile.get("followers"))
            following = _numeric(profile.get("following"))
            created_at = profile.get("created_at")
            age_days = _age_days(created_at, now)
            # The account's *lifetime* post count, not the number of posts this
            # corpus happens to have collected.
            #
            # These two are wildly different numbers and share a name, which is
            # how the mismatch survived review. Phase 1's Author.post_count is
            # "records we ingested for this author" -- median 1 on this corpus.
            # The bot benchmarks' post_count is Twitter's `statuses_count`, a
            # lifetime total with a human median of 6609. Feeding the former
            # where the model expects the latter makes every real account look
            # like a near-dead one, which in Cresci's feature space reads as
            # bot: measured, every Mastodon account scored >= 0.938.
            #
            # An intersection matched on column *name* is not an intersection.
            lifetime_posts = _lifetime_post_count(profile)
            feature.update(
                {
                    "followers": followers if followers is not None else 0.0,
                    "following": following if following is not None else 0.0,
                    # +1 in the denominator: a zero-following account is real and
                    # must not become an infinity that dominates every split.
                    "follower_following_ratio": (
                        (followers or 0.0) / ((following or 0.0) + 1.0)
                    ),
                    "posts_per_day_per_follower": (
                        posts_per_day(timestamps) / ((followers or 0.0) + 1.0)
                    ),
                    "account_age_days": age_days if age_days is not None else 0.0,
                    # Both of these now use the lifetime count, so they mean the
                    # same thing here as they do in the benchmark.
                    "post_count": (
                        lifetime_posts if lifetime_posts is not None else float(n)
                    ),
                    "posts_per_account_day": (
                        (lifetime_posts if lifetime_posts is not None else float(n))
                        / age_days
                        if age_days and age_days > 0
                        else float(lifetime_posts if lifetime_posts is not None else n)
                    ),
                    # Indicators, so "no follower data" never looks like "zero
                    # followers" -- the difference between ConvoKit Reddit and a
                    # brand-new account.
                    "followers_is_missing": 1.0 if followers is None else 0.0,
                    "account_age_is_missing": 1.0 if age_days is None else 0.0,
                    # A platform that cannot report a lifetime total is a
                    # platform this model must not be applied to. Without the
                    # indicator the fallback to observed-post-count would be
                    # invisible, and the score would look valid.
                    "post_count_is_missing": 1.0 if lifetime_posts is None else 0.0,
                }
            )

        if "threading" in wanted:
            parents = group.get("parent_id", pd.Series([None] * n, index=group.index))
            conversations = group.get(
                "conversation_id", pd.Series([None] * n, index=group.index)
            )
            replies = parents.notna()
            feature.update(
                {
                    "distinct_conversations": float(conversations.dropna().nunique()),
                    "reply_rate": float(replies.mean()) if n else 0.0,
                    "distinct_reply_targets": float(parents.dropna().nunique()),
                    "reciprocity": float(
                        parents.dropna().nunique() / max(1, int(replies.sum()))
                    ),
                }
            )

        account_ids.append(str(author_id))
        rows.append(feature)

    names = [name for tier in wanted for name in TIERS[tier]]
    matrix = np.array(
        [[float(row.get(name, 0.0)) for name in names] for row in rows], dtype=float
    )
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    log.info(
        "built %d x %d account features (tiers: %s)",
        len(account_ids),
        len(names),
        ", ".join(wanted),
    )
    return FeatureMatrix(matrix=matrix, names=names, account_ids=account_ids, tiers_used=wanted)


def intersection_features(
    benchmark_tiers: list[str], corpus_tiers: list[str]
) -> list[str]:
    """The tiers computable on *both* sides. Build this before training.

    A model trained on forty Twitter features and scored on twelve is not a
    model, it is a lookup table for a platform we do not have.
    """
    shared = [
        tier for tier in TIERS if tier in benchmark_tiers and tier in corpus_tiers
    ]
    if not shared:
        raise ValueError(
            f"no shared feature tier between benchmark {benchmark_tiers} and corpus "
            f"{corpus_tiers}. Training would produce a model that cannot be applied."
        )
    dropped = (set(benchmark_tiers) | set(corpus_tiers)) - set(shared)
    if dropped:
        log.warning(
            "dropping feature tier(s) %s: not available on both sides. This is recorded "
            "in the model card as a stated limitation, not a silent one.",
            sorted(dropped),
        )
    return shared


def available_tiers(records: pd.DataFrame, authors: pd.DataFrame | None = None) -> list[str]:
    """Which tiers a given corpus can actually support."""
    tiers = ["universal"]
    if (
        authors is not None
        and len(authors)
        and "followers" in authors.columns
        and authors["followers"].notna().any()
    ):
        tiers.append("social_graph")
    if "parent_id" in records.columns and records["parent_id"].notna().any():
        tiers.append("threading")
    return tiers


def _lifetime_post_count(profile: dict) -> float | None:
    """The account's total posts ever, from the platform's own profile payload.

    Phase 1 keeps the untouched provider payload in ``Author.raw``; Mastodon
    reports ``statuses_count`` there, and it is the only field on this corpus
    that means the same thing as the bot benchmarks' ``post_count``. Returns
    ``None`` when the platform does not report one -- ConvoKit Reddit and
    YouTube have no such concept -- so the caller can set the missingness
    indicator rather than substituting a number with different semantics.
    """
    import json

    raw = profile.get("raw")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    for key in ("statuses_count", "tweet_count", "post_count", "videoCount"):
        value = _numeric(raw.get(key))
        if value is not None:
            return float(value)
    return None


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _age_days(created_at: Any, now: pd.Timestamp) -> float | None:
    if created_at is None:
        return None
    try:
        if pd.isna(created_at):
            return None
    except (TypeError, ValueError):
        pass
    try:
        created = pd.to_datetime(created_at, utc=True)
    except (TypeError, ValueError):
        return None
    days = (now - created).total_seconds() / 86400
    return float(days) if math.isfinite(days) and days >= 0 else None
