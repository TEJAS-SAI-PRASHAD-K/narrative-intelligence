"""Post-level behavioural anomaly, unsupervised.

An IsolationForest over per-post behavioural features -- not over text. The
question it answers is "is this post unusual *for this author, on this
platform*", which is a different question from "is this post false" and must not
be conflated with it in the UI.

**What the score is.** ``anomaly_score`` is the within-corpus percentile rank of
the forest's anomaly score: 0.95 means "more anomalous than 95% of the posts in
this scoring run". It is deliberately *not* presented as a probability. There
are no anomaly labels, so there is nothing to calibrate against, and a number
that looks like a probability invites being multiplied by one. Phase 4 should
treat it as a rank.

**How it is evaluated.** By score distribution and a hand-audit of the top 20,
written into ``artifacts/error_analysis/anomaly.md``. There is no F1 here and
there will not be one: reporting a supervised metric for an unsupervised model
against labels that do not exist is the kind of number this project exists to
avoid.

**Null handling.** Phase 1's rule carries through: ``engagement`` values of
``None`` mean "not measurable on this platform" and ``0`` means "measured zero".
Every engagement-derived feature therefore ships with an explicit
``*_is_missing`` indicator column, and the missing value is filled with the
*feature's own median* rather than 0 -- filling with 0 would make an unmeasurable
metric look like a measured absence, which is exactly the conflation Phase 1
refused to make at ingest.

**But the indicators are not fitted.** They are held out of the matrix the
forest sees. An indicator is near-constant within a platform and perfectly
separating between platforms, so isolating on it costs one cut and produces a
"most anomalous" list that just names whichever source reports the least
metadata. The first hand-audit found exactly that: zero of the top 20 were
explicable. They remain available on ``AnomalyFeatures.context`` for anyone
auditing a score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from modeling.aux.base import ScoringOutcome, SkipReason
from modeling.config import ModelingSettings, get_settings, module_config

log = logging.getLogger(__name__)

#: Below this many posts an author has no baseline, so "unusual for this author"
#: is undefined. Those records get null and a reason code rather than a score
#: computed against a sample of one.
MIN_POSTS_FOR_BASELINE = 3


@dataclass
class AnomalyFeatures:
    """The feature matrix plus the names, kept together for the audit trail."""

    matrix: np.ndarray
    names: list[str]
    record_ids: list[str]
    #: Missingness indicators, kept for the audit trail but deliberately not
    #: fitted. See the note in build_features.
    context: pd.DataFrame | None = None

    def as_frame(self) -> pd.DataFrame:
        """The audit view: every feature, including the ones held out of the fit.

        Deliberately wider than ``matrix``. Someone auditing why a record was
        flagged needs to see that its engagement was unmeasurable, even though
        that fact was withheld from the forest -- ``matrix`` is what the model
        saw, this is what a human needs.
        """
        frame = pd.DataFrame(self.matrix, columns=self.names, index=self.record_ids)
        if self.context is not None and len(self.context.columns):
            context = self.context.copy()
            context.index = self.record_ids
            frame = pd.concat([frame, context], axis=1)
        return frame

    @property
    def fitted_names(self) -> list[str]:
        """Only the features the forest actually saw."""
        return list(self.names)


class AnomalyScorer:
    """IsolationForest over per-post behavioural features."""

    module = "anomaly"
    output_columns = ("anomaly_score",)

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.n_estimators = int(self.config.get("n_estimators", 200))
        self.contamination = self.config.get("contamination", 0.05)
        self._features: AnomalyFeatures | None = None

    # --- features --------------------------------------------------------
    def build_features(
        self, records: pd.DataFrame, authors: pd.DataFrame | None = None
    ) -> tuple[AnomalyFeatures, dict[str, str]]:
        """Per-post behavioural features. Pure, given the two frames.

        Every feature has a stated signal. A feature nobody can explain is a
        feature nobody can defend when the dashboard flags an account.
        """
        skipped: dict[str, str] = {}
        work = records.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        work = work.sort_values(["author_id", "timestamp"], kind="stable")

        posts_per_author = work.groupby("author_id")["id"].transform("size")
        too_few = posts_per_author < MIN_POSTS_FOR_BASELINE
        for record_id in work.loc[too_few, "id"]:
            skipped[str(record_id)] = SkipReason.NOT_ENOUGH_HISTORY
        work = work.loc[~too_few]
        if not len(work):
            return AnomalyFeatures(np.empty((0, 0)), [], []), skipped

        features = pd.DataFrame(index=work.index)

        # -- timing ------------------------------------------------------
        # Gap to the author's previous post. Bursts of near-zero gaps are the
        # single strongest cheap signal for automated posting.
        gap = work.groupby("author_id")["timestamp"].diff().dt.total_seconds()
        features["log_gap_seconds"] = np.log1p(gap.fillna(gap.median()).clip(lower=0))
        features["gap_is_missing"] = gap.isna().astype(float)

        # How far this post's hour is from the author's own usual hour, on a
        # circular clock. Humans have a diurnal rhythm; schedulers do not.
        hour = work["timestamp"].dt.hour.astype(float)
        author_hour = work.groupby("author_id")["timestamp"].transform(
            lambda s: s.dt.hour.median()
        )
        delta = (hour - author_hour).abs()
        features["hours_from_own_median"] = np.minimum(delta, 24 - delta)

        # -- engagement --------------------------------------------------
        # null means "not measurable on this platform"; 0 means "measured zero".
        # The indicator preserves that distinction into the model.
        engagement = _engagement_total(work)
        features["engagement_is_missing"] = engagement.isna().astype(float)
        features["log_engagement"] = np.log1p(
            engagement.fillna(
                engagement.median() if engagement.notna().any() else 0.0
            ).clip(lower=0)
        )

        followers = _author_followers(work, authors)
        ratio = engagement / followers.replace(0, np.nan)
        features["engagement_per_follower_is_missing"] = ratio.isna().astype(float)
        features["log_engagement_per_follower"] = np.log1p(
            ratio.fillna(ratio.median() if ratio.notna().any() else 0.0).clip(lower=0)
        )

        # -- content -----------------------------------------------------
        text = work["text"].fillna("").astype(str)
        features["log_text_length"] = np.log1p(text.str.len())
        # Self-repetition: how often this author reposts their own near-identical
        # content. High values are the template-posting signature.
        features["self_duplicate_rate"] = _self_duplicate_rate(work)
        empty = pd.Series([[]] * len(work), index=work.index)
        features["url_count"] = work.get("urls", empty).map(_len_or_zero)
        features["hashtag_count"] = work.get("hashtags", empty).map(_len_or_zero)
        features["mention_count"] = work.get("mentions", empty).map(_len_or_zero)

        # -- author context ----------------------------------------------
        features["log_author_post_count"] = np.log1p(
            work.groupby("author_id")["id"].transform("size").astype(float)
        )

        # Missingness indicators are context, not evidence of anomaly.
        #
        # The hand-audit of the top 20 came back 0 explicable / 10 artefact /
        # 10 uninteresting -- a complete failure, and the indicators were the
        # cause. `gap_is_missing` fires on every author's *first* post and
        # `engagement_is_missing` fires for entire platforms at once, so both
        # are near-constant within a group and perfectly separating between
        # groups. An IsolationForest isolates exactly that: a rare binary column
        # splits off a whole platform in one cut, and the top of the ranking
        # fills with "this record came from a source that reports no
        # engagement" -- true, and not an anomaly.
        #
        # They stay in the frame, because a downstream consumer auditing a score
        # needs to see them, and they stay out of the matrix the forest fits.
        scored_columns = [c for c in features.columns if not c.endswith("_is_missing")]
        excluded = [c for c in features.columns if c.endswith("_is_missing")]
        if excluded:
            log.debug("anomaly: %s held out of the forest as context", ", ".join(excluded))

        matrix = features[scored_columns].to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            AnomalyFeatures(
                matrix=matrix,
                names=list(scored_columns),
                record_ids=[str(v) for v in work["id"].tolist()],
                context=features[excluded].reset_index(drop=True) if excluded else None,
            ),
            skipped,
        )

    # --- scoring ---------------------------------------------------------
    def score(
        self, records: pd.DataFrame, authors: pd.DataFrame | None = None
    ) -> ScoringOutcome:
        outcome = ScoringOutcome()
        if not len(records):
            return outcome

        features, skipped = self.build_features(records, authors)
        outcome.skipped.update(skipped)
        if not len(features.record_ids):
            log.info("anomaly: no record had enough author history to score")
            return outcome

        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:  # pragma: no cover - sklearn is a modeling dep
            for record_id in features.record_ids:
                outcome.skipped[record_id] = SkipReason.MODEL_UNAVAILABLE
            return outcome

        forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.settings.seed,
            n_jobs=1,  # determinism over speed; the corpus is small
        )
        forest.fit(features.matrix)
        # score_samples: lower = more anomalous. Negate so higher = more unusual.
        raw = -forest.score_samples(features.matrix)

        # Percentile rank within this run. See the module docstring: this is a
        # rank, not a probability, and it is labelled as one everywhere.
        ranks = pd.Series(raw).rank(method="average", pct=True).to_numpy()

        for record_id, value in zip(features.record_ids, ranks, strict=True):
            outcome.values[record_id] = round(float(value), 6)

        self._features = features
        log.info(
            "anomaly: scored %d records over %d features (top decile threshold=%.3f)",
            len(outcome.values),
            len(features.names),
            float(np.quantile(ranks, 0.9)) if len(ranks) else float("nan"),
        )
        return outcome

    def audit_frame(self, top_n: int = 20) -> pd.DataFrame:
        """The top-N most anomalous posts with their features, for hand audit.

        This is the evaluation for this module. There is no F1 to report.
        """
        if self._features is None:
            return pd.DataFrame()
        frame = self._features.as_frame()
        return frame.head(top_n)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _engagement_total(records: pd.DataFrame) -> pd.Series:
    """Sum of the engagement struct, preserving "not measurable" as NaN.

    A row where every metric is null stays null. A row with likes=0 and the
    rest null sums to 0, which is correct: something was measured.
    """
    if "engagement" not in records.columns:
        return pd.Series(np.nan, index=records.index)

    def total(value) -> float:
        if value is None or not isinstance(value, dict):
            return float("nan")
        present = [v for v in value.values() if v is not None]
        return float(sum(present)) if present else float("nan")

    return records["engagement"].map(total)


def _author_followers(records: pd.DataFrame, authors: pd.DataFrame | None) -> pd.Series:
    """Follower count per record, NaN where the platform does not expose one.

    ConvoKit Reddit has no follower concept at all; Mastodon does. Filling the
    Reddit side with 0 would make every Reddit post look infinitely engaging
    per follower.
    """
    if authors is None or not len(authors) or "followers" not in authors.columns:
        return pd.Series(np.nan, index=records.index)
    lookup = authors.set_index("author_id")["followers"]
    return records["author_id"].map(lookup).astype(float)


def _self_duplicate_rate(records: pd.DataFrame) -> pd.Series:
    """Fraction of an author's own posts sharing this post's simhash prefix.

    A cheap proxy for template posting. Uses the 16-bit prefix bucket rather
    than pairwise Hamming distance: at this granularity it is a feature, not a
    dedupe decision, and the exact bit count is not worth O(n^2) per author.
    """
    if "simhash" not in records.columns:
        return pd.Series(0.0, index=records.index)
    prefix = records["simhash"].map(lambda v: None if pd.isna(v) else int(v) >> 48)
    grouped = records.assign(_prefix=prefix).groupby(["author_id", "_prefix"])["id"]
    counts = grouped.transform("size")
    totals = records.groupby("author_id")["id"].transform("size")
    rate = (counts / totals).astype(float)
    return rate.where(prefix.notna(), 0.0)


def _len_or_zero(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(len(value))
    except TypeError:
        return 0.0
