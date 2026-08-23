"""Authors, cohorts and curated author groups.

Cohorts and author groups are deliberately different tables and different
concepts, and the API keeps them apart:

* a **cohort** is model-derived and multi-label. The UI spec is explicit that an
  author can be `Right Wing` *and* `Crypto Fan` *and* `Micro Influencer` at
  once, so this is a many-to-many with a confidence, not a category column.
* an **author group** is analyst-curated -- "Russian State Affiliated Accounts",
  "Known Hostile Telegram Channels". It is a watchlist. Merging the two would
  make a model's guess indistinguishable from an analyst's assertion, which is
  exactly the distinction an investigation rests on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import Camel, ScoreComponent

CohortCategory = Literal["geopolitical", "political", "interest", "influence_tier"]


class CohortMembership(Camel):
    cohort_id: str
    name: str
    category: CohortCategory
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    assigned_by: Literal["model", "analyst"]


class AuthorOut(Camel):
    author_id: str = Field(description="Namespaced as '<source>:<native_author_id>'.")
    project_id: str
    source: str
    handle: str | None = None
    created_at_source: datetime | None = Field(
        default=None, description="Account creation date at the platform, when exposed."
    )
    followers: int | None = None
    following: int | None = None
    post_count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    is_bot_flagged: bool | None = Field(
        default=None,
        description=(
            "The platform's own bot flag where one exists (Mastodon exposes it). "
            "Distinct from bot_score, which is this system's estimate."
        ),
    )
    bot_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    partisan_lean: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="-1 left, +1 right, 0 centre or undetermined."
    )
    dominant_sentiment: str | None = None
    dominant_emotion: str | None = None
    anomalous_score: float | None = Field(default=None, ge=0.0, le=1.0)
    toxicity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    coordination_score: float | None = Field(default=None, ge=0.0, le=1.0)
    community_id: str | None = Field(
        default=None, description="Louvain community over the co-posting graph."
    )
    community_size: int | None = None
    cohorts: list[CohortMembership] = Field(default_factory=list)
    author_group_ids: list[str] = Field(default_factory=list)
    narratives_touched: list[str] = Field(default_factory=list)
    scoring_version: str | None = None
    skip_reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Why a score is null. News/RSS 'authors' are outlets, not people, so "
            "account-level bot scoring is skipped there rather than fabricated."
        ),
    )


class AuthorScoreExplanation(Camel):
    """Feature-level attribution for the bot score.

    A bot probability with no explanation is the black-box number this product
    exists to argue against, so the top contributing features travel with it.
    """

    author_id: str
    bot_score: float | None
    risk_score: float | None
    model: str
    scoring_version: str
    top_features: list[ScoreComponent] = Field(
        description="Highest-|contribution| account features, signed toward or against 'bot'."
    )
    components: list[ScoreComponent] = Field(
        default_factory=list, description="The risk-score decomposition, when risk was computed."
    )
    missing: list[str] = Field(default_factory=list)
    computed_at: datetime | None = None
    caveat: str | None = Field(
        default=None,
        description=(
            "Model limitations that bear on this specific score, e.g. weak generalisation "
            "to campaigns unlike the training set. Surfaced rather than buried in a report."
        ),
    )


class AuthorTimelinePoint(Camel):
    bucket_start: datetime
    post_count: int
    engagement: int
    mean_toxicity: float | None = None


class AuthorTimeline(Camel):
    author_id: str
    interval: str
    tz: str
    points: list[AuthorTimelinePoint]
    #: Posting-hour histogram in the requested tz. A flat 24-hour profile on a
    #: supposedly-human account is one of the cheapest coordination tells there
    #: is, so it ships with the timeline rather than needing a second call.
    hour_histogram: dict[int, int] = Field(default_factory=dict)


class CohortOut(Camel):
    id: str
    project_id: str
    name: str
    description: str | None = None
    category: CohortCategory
    author_count: int = 0
    post_pct: float | None = Field(
        default=None, description="Share of the project's posts written by this cohort."
    )
    mean_bot_score: float | None = None
    mean_toxicity: float | None = None


class AuthorGroupOut(Camel):
    id: str
    project_id: str
    name: str
    description: str | None = None
    is_curated: bool = True
    member_count: int = 0
    created_at: datetime | None = None


class AuthorGroupCreate(Camel):
    project_id: str
    name: str = Field(min_length=1, max_length=256)
    description: str | None = None


class AuthorGroupMembers(Camel):
    author_ids: list[str] = Field(min_length=1, max_length=5000)


class AuthorGroupMembersResult(Camel):
    group_id: str
    added: int
    already_present: int
    #: Ids that matched no author in this project. Reported rather than silently
    #: dropped: a watchlist that quietly lost half its entries is worse than one
    #: that failed loudly.
    unknown: list[str] = Field(default_factory=list)
