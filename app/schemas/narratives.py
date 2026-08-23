"""Narratives, the six-field scorecard, and the explainability payload.

The scorecard shape is reused on the feed card, the detail header and the
comparison table, so it is defined once here. If the UI shows a number in three
places, three different queries computing it three slightly different ways is
the failure mode this file exists to prevent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import Camel, GeneratedText, ScoreComponent

Priority = Literal["high", "medium", "low"]
RiskBand = Literal["high", "medium", "low"]


class NarrativeScorecard(Camel):
    """The six fields the UI renders on every narrative surface.

    ``fusion_score`` is 0-100 and ``priority`` is derived from it by thresholds
    in configs/fusion.yaml -- never hand-assigned, so two narratives with the
    same score can never carry different priorities.
    """

    priority: Priority
    bot_like: float | None = Field(default=None, ge=0.0, le=1.0)
    anomalous: float | None = Field(default=None, ge=0.0, le=1.0)
    toxicity: float | None = Field(default=None, ge=0.0, le=1.0)
    compass_risk: RiskBand | None = None
    negative_sentiment: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion_score: float | None = Field(default=None, ge=0.0, le=100.0)
    scoring_version: str | None = None
    computed_at: datetime | None = None
    #: Names of components that could not be computed. The weights are
    #: renormalized over what remains rather than treating a missing input as a
    #: zero, which would systematically under-flag.
    missing: list[str] = Field(default_factory=list)


class NarrativeSummary(Camel):
    """One card in the narrative feed."""

    id: str
    display_id: int = Field(description="The human-facing 'ID: ####' the UI shows.")
    project_id: str
    title: GeneratedText
    summary: GeneratedText
    claim: str | None = Field(
        default=None,
        description="The specific factual assertion the narrative makes, as extracted for Compass.",
    )
    scorecard: NarrativeScorecard
    date_start: datetime | None = None
    date_end: datetime | None = None
    post_count: int = 0
    author_count: int = 0
    engagement_total: int = 0
    platforms: list[str] = Field(default_factory=list)
    top_domains: list[str] = Field(default_factory=list)
    top_hashtags: list[str] = Field(default_factory=list)
    is_manual: bool = False
    cluster_id: int | None = None
    clustering_run_id: str | None = None
    updated_at: datetime | None = None


class RepresentativePost(Camel):
    """A cluster exemplar, for the feed card's preview line."""

    post_id: str
    text: str
    source: str
    author_handle: str | None = None
    timestamp: datetime
    membership_score: float | None = None


class NarrativeDetail(NarrativeSummary):
    representative_posts: list[RepresentativePost] = Field(default_factory=list)
    coherence: float | None = Field(
        default=None,
        description=(
            "Cluster tightness. Low coherence means the label describes the cluster loosely."
        ),
    )
    velocity: float | None = Field(default=None, description="Posts per hour at peak.")
    compass_available: bool = False


class NarrativeCreate(Camel):
    """A hand-built narrative from the Compose/Curate page."""

    project_id: str
    title: str = Field(min_length=1, max_length=512)
    summary: str | None = None
    claim: str | None = None
    post_ids: list[str] = Field(
        default_factory=list, description="Seed membership. May be empty and filled in later."
    )


class NarrativeUpdate(Camel):
    """An analyst edit.

    Any field set here marks the narrative ``edited_by_user``, and a subsequent
    reclustering run must not overwrite it. That rule is enforced in the task,
    not in the UI, and there is a test for it.
    """

    title: str | None = Field(default=None, min_length=1, max_length=512)
    summary: str | None = None
    claim: str | None = None


class TimelinePoint(Camel):
    bucket_start: datetime
    post_count: int
    engagement: int
    author_count: int
    #: Present only when ``cross_platform=true``; the UI stacks these.
    by_platform: dict[str, int] | None = None


class NarrativeTimeline(Camel):
    narrative_id: str
    interval: str
    tz: str = Field(description="Echoed so the UI can label the axis honestly.")
    points: list[TimelinePoint]
    peak_at: datetime | None = None
    #: A burst is what coordination looks like from the outside. Flagged rather
    #: than left for the eye, because a 6-hour spike on a 90-day axis is easy to
    #: miss and is exactly the signal that matters.
    bursts: list[dict[str, Any]] = Field(default_factory=list)


class NarrativeScoreExplanation(Camel):
    """The explainability endpoint's payload.

    This is the product thesis in one response body. A number the UI cannot
    explain in one click is a failed requirement, so the score never travels
    without the weights, the formula version and the inputs that produced it.
    """

    narrative_id: str
    fusion_score: float | None
    priority: Priority
    scoring_version: str
    formula: str = Field(
        description=(
            "The literal formula, e.g. '100 * (w1*severity + w2*coordination + w3*authenticity)'."
        )
    )
    normalization: str = Field(
        description=(
            "How sub-scores were mapped to [0,1], named so scores are comparable across projects."
        )
    )
    components: list[ScoreComponent]
    missing: list[str] = Field(
        default_factory=list,
        description=(
            "Components that could not be computed. Weights are renormalized over the "
            "rest; a missing input is never treated as zero."
        ),
    )
    weights_renormalized: bool = False
    computed_at: datetime | None = None


class NarrativeFreshness(Camel):
    """Backs the UI's 'Generated X ago / Update Now' pattern."""

    project_id: str
    last_run_id: str | None = None
    last_run_at: datetime | None = None
    age_seconds: float | None = None
    age_human: str = Field(description="e.g. '48 days, 0 hrs ago'. Formatted server-side.")
    status: Literal["fresh", "stale", "running", "never_run"]
    algorithm: str | None = None
    embedding_model: str | None = None
    post_count: int | None = None
    narrative_count: int | None = None
    #: Posts ingested since the last clustering run. This, not wall-clock age,
    #: is what actually makes a clustering stale.
    unclustered_post_count: int | None = None


class NarrativeCohortShare(Camel):
    """One bar of the narrative's top-cohorts chart."""

    cohort_id: str
    name: str
    category: str
    author_count: int
    post_count: int
    share_pct: float = Field(description="Share of this narrative's posts, not of the cohort.")
