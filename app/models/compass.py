"""Compass Context: the RAG fact-check engine's persisted output.

The schema encodes the generation rules rather than trusting the prompt to
follow them:

* ``citations`` carry a character span into ``context``. A citation that does
  not point at the text it supports is a bibliography, and a bibliography is
  what an ungrounded generation produces when asked to look grounded.
* Regeneration **inserts** and sets ``superseded_by`` on the old row. Nothing is
  ever mutated. Analysts need to be able to say what the system claimed and when.
* A context that failed citation validation is persisted with
  ``verification_status='insufficient_evidence'`` and an empty ``context``. The
  row exists -- so the attempt is auditable -- and the paragraph does not.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, uuid_pk

VERIFICATION_STATUSES = (
    "unverified",
    "partially_substantiated",
    "substantiated",
    "debunked",
    "insufficient_evidence",
)


class CompassContext(Base):
    __tablename__ = "compass_contexts"

    id: Mapped[uuid.UUID] = uuid_pk()
    narrative_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("narratives.id", ondelete="CASCADE"), nullable=False
    )
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    #: Empty string, never null, when validation failed. Null would be
    #: indistinguishable from "not generated yet", and those are different
    #: states an analyst must be able to tell apart.
    context: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    risk: Mapped[str] = mapped_column(String(8), nullable=False, server_default="medium")
    caution_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    #: How many retrieve -> generate -> validate rounds this took. Surfaced in
    #: the API because a context that needed two attempts deserves more scrutiny.
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    retrieved_document_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    #: The validator's own report: which sentences were checked, which citation
    #: covered each, and what the hedging lint found. Kept for the writeup's
    #: error analysis, and because "why did this fail" is otherwise unanswerable.
    validation_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("compass_contexts.id", ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        # The read path wants the one live context per narrative, and there is
        # exactly one: the row nothing supersedes.
        Index(
            "ix_compass_contexts_current",
            "narrative_id",
            postgresql_where=text("superseded_by IS NULL"),
        ),
        Index("ix_compass_contexts_narrative_time", "narrative_id", text("generated_at DESC")),
    )


class CompassCitation(Base):
    """A retrieved source and the span of ``context`` it supports."""

    __tablename__ = "compass_citations"

    id: Mapped[uuid.UUID] = uuid_pk()
    context_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("compass_contexts.id", ondelete="CASCADE"), nullable=False
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    publisher: Mapped[str | None] = mapped_column(String(256), nullable=True)
    domain: Mapped[str | None] = mapped_column(String(320), nullable=True)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: The passage the generator actually saw. Stored so a disputed context can
    #: be audited against its evidence without re-fetching a page that may have
    #: changed or disappeared.
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (Index("ix_compass_citations_context", "context_id"),)


class CompassFeedback(Base):
    """Thumbs up/down, stored for the writeup's error analysis."""

    __tablename__ = "compass_feedback"

    id: Mapped[uuid.UUID] = uuid_pk()
    context_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("compass_contexts.id", ondelete="CASCADE"), nullable=False
    )
    helpful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=text("now()")
    )

    __table_args__ = (Index("ix_compass_feedback_context", "context_id"),)
