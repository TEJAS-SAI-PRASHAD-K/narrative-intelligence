"""Domain risk -- the fourth pillar.

Enrichment (WHOIS, TLS, hosting) is best-effort and rate-limited, and this
schema is built around that reality rather than pretending otherwise. An
un-enriched domain still carries a risk score computed from in-corpus signals
alone, with ``enrichment_status='unavailable'`` and the enrichment fields null.
Blocking a page render on a WHOIS lookup would be a design bug; showing a
confident risk band derived from a failed lookup would be worse.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import Camel, ScoreComponent

EnrichmentStatus = Literal["enriched", "pending", "unavailable", "failed", "not_applicable"]


class DomainOut(Camel):
    domain: str = Field(description="Registrable domain, lowercased. Subdomains roll up to it.")
    project_id: str
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    risk_band: Literal["high", "medium", "low"] | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    post_count: int = 0
    author_count: int = 0
    narrative_count: int = 0

    # --- enrichment, all nullable by construction ---
    whois_created_at: datetime | None = None
    domain_age_days: int | None = None
    hosting_country: str | None = None
    tls_cert_age_days: int | None = None
    registrar: str | None = None
    enrichment_status: EnrichmentStatus = "pending"
    enriched_at: datetime | None = None
    enrichment_detail: str | None = Field(
        default=None, description="Why enrichment is unavailable, when it is."
    )

    # --- in-corpus signals, always computable ---
    link_velocity: float | None = Field(
        default=None, description="Posts per hour carrying this domain at peak."
    )
    sharing_author_bot_ratio: float | None = Field(
        default=None, description="Share of the authors linking it that score bot-like."
    )
    scoring_version: str | None = None
    components: list[ScoreComponent] = Field(default_factory=list)


class DomainNarrativeLink(Camel):
    narrative_id: str
    title: str
    post_count: int
    share_pct: float
    first_seen: datetime | None = None
    last_seen: datetime | None = None
