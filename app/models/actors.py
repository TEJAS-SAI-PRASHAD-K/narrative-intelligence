"""Actors: authors, cohorts and curated groups.

The cohort relationship is many-to-many and that is load-bearing. The UI spec is
explicit that an author can be `Right Wing` *and* `Crypto Fan` *and* `Micro
Influencer` at the same time. Modelling it as a single category column would
work right up until the first author who is two things, and then it is a
migration under time pressure.

``author_groups`` is deliberately a different table from ``cohorts``: a cohort
is a model's multi-label guess with a confidence, an author group is an
analyst's assertion. An investigation rests on being able to tell those apart.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, uuid_pk

COHORT_CATEGORIES = ("geopolitical", "political", "interest", "influence_tier")


class Author(Base):
    """Phase 1's ``Author`` roll-up plus Phase 2's account-level scores."""

    __tablename__ = "authors"

    author_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    #: TEXT, matching posts.author_handle. The roll-up carries the most recent
    #: handle forward, and on a news "author" that is an article byline running
    #: to hundreds of characters. A varchar here overflows only when the newest
    #: post for that outlet happens to have a long byline, which makes it a
    #: non-deterministic failure -- the worst kind to leave in a loader.
    handle: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Account creation at the platform, where the platform exposes it. Named
    #: apart from the row's own created_at, which does not exist here: the
    #: roll-up is derived and rebuilt, so its own age means nothing.
    created_at_source: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    followers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    following: Mapped[int | None] = mapped_column(Integer, nullable=True)
    post_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: The *platform's* bot flag (Mastodon exposes one). Distinct from
    #: bot_score, which is this system's estimate. Conflating a self-declared
    #: bot account with a suspected inauthentic one would be a serious error.
    is_bot_flagged: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    bot_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    partisan_lean: Mapped[float | None] = mapped_column(Float, nullable=True)
    dominant_sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)
    dominant_emotion: Mapped[str | None] = mapped_column(String(24), nullable=True)
    anomalous_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    toxicity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    coordination_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    #: Louvain community over the co-posting graph. A string rather than an FK:
    #: communities are recomputed wholesale on every graph refresh and carry no
    #: identity across runs, so a foreign key would be a lie about stability.
    community_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    community_size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    author_group_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("author_groups.id", ondelete="SET NULL"), nullable=True
    )

    #: Why a score is null. News/RSS "authors" are outlets, not people, so
    #: account-level bot scoring is skipped with a code rather than fabricated.
    skip_reasons: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), nullable=False, server_default="{}"
    )
    scoring_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    score_components: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_authors_project_risk", "project_id", text("risk_score DESC NULLS LAST")),
        Index(
            "ix_authors_project_bot",
            "project_id",
            text("bot_score DESC"),
            postgresql_where=text("bot_score IS NOT NULL"),
        ),
        Index(
            "ix_authors_handle_trgm",
            "handle",
            postgresql_using="gin",
            postgresql_ops={"handle": "gin_trgm_ops"},
        ),
        Index("ix_authors_community", "project_id", "community_id"),
    )


class Cohort(Base):
    """A model-derived audience segment."""

    __tablename__ = "cohorts"

    id: Mapped[uuid.UUID] = uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    author_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    post_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_cohorts_project_id_name"),
        Index("ix_cohorts_project_category", "project_id", "category"),
    )


class AuthorCohort(Base):
    """Many-to-many, multi-label, with the confidence and who assigned it.

    ``assigned_by`` matters: an analyst's manual assignment must never be
    overwritten by the next model run, and the task cannot tell the difference
    without this column.
    """

    __tablename__ = "author_cohorts"

    author_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("authors.author_id", ondelete="CASCADE"), primary_key=True
    )
    cohort_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cohorts.id", ondelete="CASCADE"), primary_key=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    assigned_by: Mapped[str] = mapped_column(String(16), nullable=False, server_default="model")
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_author_cohorts_cohort", "cohort_id"),)


class AuthorGroup(Base):
    """An analyst-curated watchlist. Not model-derived, ever."""

    __tablename__ = "author_groups"

    id: Mapped[uuid.UUID] = uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_curated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_author_groups_project_id_name"),
    )


class AuthorGroupMember(Base):
    """Explicit membership table.

    ``authors.author_group_id`` exists too, for the single primary group the UI
    shows on an author card, but membership is genuinely many-to-many -- an
    account can be on two watchlists -- and this is the table that means it.
    """

    __tablename__ = "author_group_members"

    group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("author_groups.id", ondelete="CASCADE"), primary_key=True
    )
    author_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("authors.author_id", ondelete="CASCADE"), primary_key=True
    )
    added_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=text("now()")
    )
    added_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_author_group_members_author", "author_id"),)
