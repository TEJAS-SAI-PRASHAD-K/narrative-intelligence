"""The corpus: posts and their per-post scores.

Column names mirror ``ingest/schema.py`` field-for-field, with two deliberate
divergences that the Parquet on disk forced and that are documented here rather
than discovered later:

1. **``engagement`` is flattened.** Phase 1 stores it as an Arrow struct
   (``likes``/``shares``/``replies``/``views``). Postgres gets four nullable
   columns instead, because a composite type cannot be indexed usefully and
   every aggregate would need to unpack it. The null semantics survive exactly:
   ``NULL`` means the platform does not expose the metric, ``0`` means measured
   zero, and the loader never coalesces.
2. **``simhash`` is stored as a signed BIGINT.** Phase 1 computes an unsigned
   64-bit value and Arrow stores it as ``uint64``; Postgres has no unsigned
   integer type. The loader reinterprets the same 64 bits as signed rather than
   widening to NUMERIC, so Hamming-distance work still runs on a fixed-width
   integer and the bit pattern round-trips exactly. See ``app/etl/parquet_loader``.

``post_scores`` is a separate table from ``posts`` on purpose. The corpus is
immutable; scores are not. Keeping them apart makes a rescore an idempotent
upsert into one table that never touches corpus data, which is what turns
"rerun scoring with new weights" into a five-minute operation instead of a
migration.
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
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

#: Phase 1's five source values. Six adapters converge on them: ConvoKit and
#: Kaggle both emit `reddit`.
SOURCES = ("reddit", "mastodon", "news", "gdelt", "youtube")


class Post(Base):
    """One row per Phase 1 ``Record``."""

    __tablename__ = "posts"

    #: ``<source>:<native_id>``. Namespaced by Phase 1 so ids from two platforms
    #: can never collide when the corpus is unioned.
    #:
    #: 512 rather than 320: news and GDELT records derive their native_id from
    #: the article URL, and the real corpus already contains a 219-character id
    #: (a Guardian liveblog slug). Measured against the Parquet, not guessed.
    id: Mapped[str] = mapped_column(String(512), primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    native_id: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_detail: Mapped[str] = mapped_column(String(256), nullable=False)
    content_type: Mapped[str] = mapped_column(String(32), nullable=False)
    text_: Mapped[str] = mapped_column("text", Text, nullable=False)
    lang: Mapped[str | None] = mapped_column(String(8), nullable=True)

    author_id: Mapped[str] = mapped_column(String(512), nullable=False)
    #: Unbounded TEXT, not a short varchar. On news and GDELT rows Phase 1 puts
    #: the article *byline* here, and a multi-author paper's byline runs to 744
    #: characters in the current corpus. Calling it a "handle" is Phase 1's
    #: naming; the data is whatever the platform calls an author label.
    author_handle: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Always tz-aware UTC. The database's timezone is set to UTC too, so a
    #: client that forgets to say so still gets UTC rather than the container's
    #: guess at a locale.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    parent_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # --- engagement: four nullable columns, never coalesced ---
    likes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shares: Mapped[int | None] = mapped_column(Integer, nullable=True)
    replies: Mapped[int | None] = mapped_column(Integer, nullable=True)
    views: Mapped[int | None] = mapped_column(Integer, nullable=True)

    urls: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    domains: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    media_urls: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    hashtags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    mentions: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")

    #: Signed reinterpretation of Phase 1's unsigned 64-bit simhash. See the
    #: module docstring; the bits are identical, only the sign convention differs.
    simhash: Mapped[int | None] = mapped_column(
        sa_bigint := __import__("sqlalchemy").BigInteger, nullable=True
    )

    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        # The dashboard's default query: one project, newest first. Composite
        # and ordered to match, so the planner reads the index backwards rather
        # than sorting the project's whole corpus.
        Index("ix_posts_project_timestamp", "project_id", text("timestamp DESC")),
        Index("ix_posts_source_timestamp", "source", text("timestamp DESC")),
        Index("ix_posts_author_id", "author_id"),
        Index("ix_posts_conversation_id", "conversation_id"),
        # Near-duplicate detection joins on this. Phase 2 uses it for repost
        # swarm collapse before clustering.
        Index("ix_posts_simhash", "simhash"),
        Index("ix_posts_domains_gin", "domains", postgresql_using="gin"),
        Index("ix_posts_hashtags_gin", "hashtags", postgresql_using="gin"),
        # jsonb_path_ops rather than the default: half the size and faster for
        # the containment queries this column actually gets, at the cost of key-
        # existence operators nothing here uses.
        Index(
            "ix_posts_raw_gin",
            "raw",
            postgresql_using="gin",
            postgresql_ops={"raw": "jsonb_path_ops"},
        ),
        # Full-text search. Expression index rather than a stored tsvector
        # column: the text is immutable once loaded, so there is nothing to keep
        # in sync, and a generated column would add 30% to the table size.
        Index(
            "ix_posts_text_fts",
            text("to_tsvector('english', text)"),
            postgresql_using="gin",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Post {self.id} {self.timestamp:%Y-%m-%d}>"


class PostScore(Base):
    """Per-post manipulation signals, written by the scoring tasks.

    Every column is nullable and that is the contract: a post the scorer
    declined to handle (wrong language, too short) gets a ``skip_reasons`` entry,
    not a fabricated number. The same null-versus-zero discipline the corpus
    enforces on engagement metrics.
    """

    __tablename__ = "post_scores"

    post_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True
    )

    toxicity: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_toxic: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    anomaly: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_anomalous: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    misinfo_likelihood: Mapped[float | None] = mapped_column(Float, nullable=True)
    stance: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stance_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    emotion: Mapped[str | None] = mapped_column(String(24), nullable=True)
    emotion_scores: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    skip_reasons: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), nullable=False, server_default="{}"
    )
    #: The model_versions map Phase 2 already writes into its Parquet. A row
    #: whose versions differ from the current config is stale, and that is how
    #: the resumable scorer decides what to recompute.
    model_versions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    scoring_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # The overview's high-risk panel and the min_toxicity filter both hit
        # these. Partial on NOT NULL because the null rows are never selected by
        # a threshold filter and have no business inflating the index.
        Index(
            "ix_post_scores_misinfo",
            text("misinfo_likelihood DESC"),
            postgresql_where=text("misinfo_likelihood IS NOT NULL"),
        ),
        Index(
            "ix_post_scores_toxicity",
            text("toxicity DESC"),
            postgresql_where=text("toxicity IS NOT NULL"),
        ),
        Index("ix_post_scores_sentiment", "sentiment"),
        Index("ix_post_scores_emotion", "emotion"),
        Index("ix_post_scores_version", "scoring_version"),
    )
