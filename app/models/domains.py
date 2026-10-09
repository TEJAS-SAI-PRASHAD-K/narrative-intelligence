"""Domain risk -- the third pillar.

The table is shaped around the fact that enrichment is best-effort. Every WHOIS
/ TLS / hosting column is nullable, ``enrichment_status`` says why, and
``risk_score`` is computed from in-corpus signals that are always available.
A domain page must render, with a defensible risk band, for a domain whose
registrar never answered.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

ENRICHMENT_STATUSES = ("enriched", "pending", "unavailable", "failed", "not_applicable")


class Domain(Base):
    __tablename__ = "domains"

    #: Registrable domain, lowercased. Subdomains roll up to it -- three
    #: subdomains of one disinformation site are one actor, not three, and
    #: keying on the full hostname would split its risk score three ways.
    domain: Mapped[str] = mapped_column(String(320), primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    )

    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_band: Mapped[str | None] = mapped_column(String(8), nullable=True)
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    post_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    author_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    narrative_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # --- enrichment: all nullable, by construction ---
    whois_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    hosting_country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    tls_cert_age_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    registrar: Mapped[str | None] = mapped_column(String(256), nullable=True)
    enrichment_status: Mapped[str] = mapped_column(
        String(24), nullable=False, server_default="pending"
    )
    enriched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enrichment_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Bounded retry counter. A domain whose WHOIS server is permanently down
    #: must stop consuming the enrichment task's rate limit forever.
    enrichment_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # --- in-corpus signals: always computable ---
    link_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharing_author_bot_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    components: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    scoring_version: Mapped[str | None] = mapped_column(String(128), nullable=True)

    __table_args__ = (
        Index("ix_domains_project_risk", "project_id", text("risk_score DESC NULLS LAST")),
        Index(
            "ix_domains_enrichment_pending",
            "enrichment_status",
            postgresql_where=text("enrichment_status = 'pending'"),
        ),
    )
