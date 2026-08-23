"""Overview / dashboard payloads.

Every KPI here ships with its own definition string. That is a deliberate
division of labour: the UI spec requires a `(?)` tooltip on every number
explaining exactly how it was computed, and only the backend knows what the
query actually summed, which nulls it skipped and which rows the filter
excluded. A frontend-authored definition is a guess that goes stale the first
time the query changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import Camel, Metric

Interval = Literal["1h", "6h", "1d", "7d"]
GroupBy = Literal["platform", "narrative", "cohort", "source_detail", "none"]
Aggregation = Literal["total", "average", "peak"]


class KpiBundle(Camel):
    """The five headline KPIs, in the fixed order the UI renders them.

    The order is part of the contract. The dashboard's top row is Posts,
    Engagements, Authors, Sentiment, Emotions, and a response that reorders them
    would silently reshuffle the page.
    """

    posts: Metric
    engagements: Metric
    authors: Metric
    sentiment: Metric
    emotions: Metric
    window_start: datetime | None = None
    window_end: datetime | None = None
    tz: str = "UTC"
    comparison_window: str | None = Field(
        default=None, description="The preceding window that delta_pct compares against."
    )


class HierarchyNode(Camel):
    """A node in a drilldown tree (domain -> channel, cohort -> author)."""

    key: str
    label: str
    value: int
    pct: float | None = None
    children: list[HierarchyNode] = Field(default_factory=list)


class PostsBreakdown(Camel):
    total: Metric
    original: Metric
    local_shared: Metric
    anonymous: Metric
    anomalous: Metric
    toxic: Metric
    by_domain: list[HierarchyNode] = Field(default_factory=list)
    by_channel: list[HierarchyNode] = Field(default_factory=list)


class EngagementsBreakdown(Camel):
    total: Metric
    on_high_risk: Metric
    likes: Metric
    global_shares: Metric
    reactions: Metric
    views: Metric


class AuthorsBreakdown(Camel):
    total: Metric
    bot_like: Metric
    author_groups: Metric
    top_cohorts: list[HierarchyNode] = Field(default_factory=list)
    total_cohort_count: int = 0


class EmotionShare(Camel):
    emotion: Literal["fear", "anger", "disgust", "happiness", "surprise", "sadness", "neutral"]
    count: int
    pct: float


class EmotionsBreakdown(Camel):
    items: list[EmotionShare]
    total_scored: int
    #: Posts the emotion model declined to score. Reported separately so the
    #: percentages add to 100 over what was actually measured, not over what
    #: was merely present.
    unscored: int = 0
    definition: str


class SentimentShare(Camel):
    sentiment: Literal["positive", "neutral", "negative"]
    count: int
    pct: float
    mean_score: float | None = None


class SentimentBreakdown(Camel):
    items: list[SentimentShare]
    total_scored: int
    unscored: int = 0
    net_sentiment: float | None = Field(
        default=None, description="positive_pct - negative_pct, in [-100, 100]."
    )
    definition: str


class TimeseriesPoint(Camel):
    bucket_start: datetime
    value: float
    #: Present when group_by is not 'none'. Keyed by platform / narrative id /
    #: cohort id depending on the grouping.
    series: dict[str, float] | None = None


class TimeseriesResponse(Camel):
    metric: Literal["posts", "engagements", "authors"]
    interval: Interval
    group_by: GroupBy
    agg: Aggregation
    normalized: bool = Field(
        description=(
            "True when each series is scaled to its own peak. Normalized series show "
            "shape and cannot be compared for magnitude; the UI must label them as such."
        )
    )
    tz: str
    points: list[TimeseriesPoint]
    series_labels: dict[str, str] = Field(
        default_factory=dict, description="Series key -> display name, so the UI shows no uuids."
    )
    definition: str


class ConceptItem(Camel):
    """An extracted topic or entity for the concept cloud."""

    term: str
    kind: Literal["hashtag", "entity", "keyword", "domain"]
    volume: int
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    dominant_emotion: str | None = None
    narrative_ids: list[str] = Field(default_factory=list)
    delta_pct: float | None = None


class ConceptsResponse(Camel):
    items: list[ConceptItem]
    total_terms: int
    definition: str
