"""Network graph, layouts and saved comparisons.

The single most important field in this module is ``stats.truncated``. A graph
that quietly drops its periphery makes a coordinated cluster look more isolated
than it is -- which is precisely the wrong error for this product to make -- so
truncation is always reported, along with the threshold that was applied and how
many nodes were dropped.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import Camel

EdgeType = Literal["reply", "repost", "mention", "co_post_similarity"]


class GraphNode(Camel):
    id: str = Field(description="author_id.")
    label: str | None = None
    source: str
    degree: int
    #: Precomputed server-side. The browser must never run a force layout over
    #: 20,000 nodes; it will drop frames and the analyst will conclude the tool
    #: is broken rather than that the graph is big.
    x: float | None = None
    y: float | None = None
    community_id: str | None = None
    bot_score: float | None = None
    is_bot_like: bool | None = None
    post_count: int = 0
    cohort_ids: list[str] = Field(default_factory=list)
    author_group_ids: list[str] = Field(default_factory=list)


class GraphEdge(Camel):
    source: str
    target: str
    edge_type: EdgeType
    weight: float
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    bucket_start: datetime | None = Field(
        default=None, description="Which scrubber bucket this edge belongs to."
    )


class GraphBucket(Camel):
    """One stop on the UI's time scrubber."""

    bucket_start: datetime
    bucket_end: datetime
    node_count: int
    edge_count: int


class GraphLegendEntry(Camel):
    key: str
    label: str
    colour_role: str = Field(
        description="Semantic role, not a hex value. Theming belongs to the frontend."
    )
    definition: str


class GraphStats(Camel):
    node_count: int
    edge_count: int
    truncated: bool = False
    #: Populated whenever truncated is true. Never silently truncate.
    truncation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "{'applied_max_nodes', 'dropped_nodes', 'dropped_edges', 'min_degree_kept'}. "
            "A dropped periphery makes a cluster look more isolated than it is, so the "
            "response says exactly what was removed and by what rule."
        ),
    )
    density: float | None = None
    component_count: int | None = None
    largest_component_size: int | None = None
    modularity: float | None = Field(
        default=None, description="Louvain modularity of the community partition, when computed."
    )


class GraphResponse(Camel):
    project_id: str
    narrative_id: str | None = None
    comparison_id: str | None = None
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    buckets: list[GraphBucket] = Field(default_factory=list)
    layout: str = Field(
        description="Layout algorithm, or 'client' when small enough to lay out live."
    )
    legend: list[GraphLegendEntry] = Field(default_factory=list)
    stats: GraphStats
    hide_standalone: bool = False
    bucket_width: str = "12h"


class LayoutRequest(Camel):
    project_id: str
    narrative_id: str | None = None
    comparison_id: str | None = None
    algorithm: Literal["forceatlas2", "fruchterman_reingold", "spring", "kamada_kawai"] = (
        "forceatlas2"
    )
    max_nodes: int | None = None


class ComparisonCreate(Camel):
    project_id: str
    name: str = Field(min_length=1, max_length=256)
    narrative_ids: list[str] = Field(min_length=2, max_length=8)
    platforms: list[str] = Field(default_factory=list)
    entity_type: Literal["author", "hashtag", "domain", "url"] = "author"


class ComparisonMetricRow(Camel):
    """One row of the UI's 'CW v. WM v. RS' table."""

    metric: str
    definition: str
    values: dict[str, float | int | None] = Field(
        description=(
            "narrative_id -> value. Null where the metric was not measured for that narrative."
        )
    )


class ComparisonOut(Camel):
    id: str
    project_id: str
    name: str
    narrative_ids: list[str]
    narrative_titles: dict[str, str] = Field(default_factory=dict)
    platforms: list[str] = Field(default_factory=list)
    entity_type: str
    rows: list[ComparisonMetricRow] = Field(default_factory=list)
    #: Authors/hashtags/domains appearing in more than one of the compared
    #: narratives. This overlap is the whole reason to run a comparison.
    shared_entities: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
