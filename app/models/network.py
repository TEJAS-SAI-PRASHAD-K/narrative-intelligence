"""The materialized interaction graph and its precomputed layouts.

Both tables exist for the same reason: the browser must never be asked to do
this work. Recomputing edges per request would be a multi-second aggregation,
and running a force layout over 20,000 nodes client-side drops frames until the
analyst concludes the tool is broken.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, uuid_pk

EDGE_TYPES = ("reply", "repost", "mention", "co_post_similarity")


class NetworkEdge(Base):
    """One time-bucketed edge.

    Bucketing at write time rather than filtering by timestamp at read time is
    what makes the UI's scrubber instant: each stop is an index range, not a
    range scan plus an aggregation.
    """

    __tablename__ = "network_edges"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    #: Null for project-wide edges. A narrative-scoped copy exists alongside the
    #: global one rather than being derived, because deriving it per request is
    #: a join against narrative_posts on both endpoints.
    narrative_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("narratives.id", ondelete="CASCADE"), nullable=True
    )
    src_author_id: Mapped[str] = mapped_column(String(512), nullable=False)
    dst_author_id: Mapped[str] = mapped_column(String(512), nullable=False)
    edge_type: Mapped[str] = mapped_column(String(24), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    bucket_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    first_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    observations: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    __table_args__ = (
        # The scrubber's query, exactly.
        Index("ix_network_edges_scope_bucket", "project_id", "narrative_id", "bucket_start"),
        Index("ix_network_edges_src", "project_id", "src_author_id"),
        Index("ix_network_edges_dst", "project_id", "dst_author_id"),
        # Idempotency for graph.build_edges: a rerun upserts rather than
        # doubling every weight, which is the classic way a coordination graph
        # quietly becomes wrong after a retry.
        Index(
            "uq_network_edges_identity",
            "project_id",
            "narrative_id",
            "src_author_id",
            "dst_author_id",
            "edge_type",
            "bucket_start",
            unique=True,
            postgresql_nulls_not_distinct=True,
        ),
    )


class NetworkLayout(Base):
    """Precomputed node positions."""

    __tablename__ = "network_layouts"

    id: Mapped[uuid.UUID] = uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    narrative_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("narratives.id", ondelete="CASCADE"), nullable=True
    )
    comparison_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("comparisons.id", ondelete="CASCADE"), nullable=True
    )
    algorithm: Mapped[str] = mapped_column(String(48), nullable=False)
    node_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    edge_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    #: ``{author_id: [x, y]}``. JSONB rather than a child table: it is written
    #: whole, read whole, and never queried by node. At 20k nodes this is a few
    #: hundred KB, comfortably inside JSONB's practical range. The threshold at
    #: which a child table wins is documented in docs/data-model.md.
    positions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=text("now()")
    )

    __table_args__ = (
        Index("ix_network_layouts_scope", "project_id", "narrative_id", "comparison_id"),
    )
