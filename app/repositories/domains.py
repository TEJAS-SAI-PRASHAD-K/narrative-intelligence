"""Domain queries.

Every enrichment field is nullable and the risk score never depends on one. A
domain whose registrar never answered still gets a defensible band from
in-corpus signals, and `enrichment_status` says why the WHOIS fields are empty.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import utcnow
from app.deps import FilterSpec, Page
from app.errors import NotFound
from app.repositories.filters import count_rows, decode_cursor, encode_cursor, resolve_project_id
from app.schemas.common import PageResponse, ScoreComponent
from app.schemas.domains import DomainNarrativeLink, DomainOut

log = logging.getLogger(__name__)


def to_domain_out(row) -> DomainOut:
    age_days = None
    if row.whois_created_at:
        age_days = (utcnow() - row.whois_created_at).days
    components = row.components or {}
    return DomainOut(
        domain=row.domain,
        project_id=str(row.project_id),
        risk_score=row.risk_score,
        risk_band=row.risk_band,
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        post_count=row.post_count or 0,
        author_count=row.author_count or 0,
        narrative_count=row.narrative_count or 0,
        whois_created_at=row.whois_created_at,
        domain_age_days=age_days,
        hosting_country=row.hosting_country,
        tls_cert_age_days=row.tls_cert_age_days,
        registrar=row.registrar,
        enrichment_status=row.enrichment_status,
        enriched_at=row.enriched_at,
        enrichment_detail=row.enrichment_detail,
        link_velocity=row.link_velocity,
        sharing_author_bot_ratio=row.sharing_author_bot_ratio,
        scoring_version=row.scoring_version,
        components=[
            ScoreComponent(
                name=name,
                value=part.get("value"),
                weight=part.get("weight", 0.0),
                contribution=part.get("contribution"),
                definition=part.get("definition", ""),
                inputs=part.get("inputs", {}),
            )
            for name, part in (components.get("components") or {}).items()
        ],
    )


async def list_domains(
    session: AsyncSession, spec: FilterSpec, page: Page, *, sort: str = "risk"
) -> PageResponse[DomainOut]:
    from app.models.domains import Domain

    project_id = await resolve_project_id(session, spec.project_id)
    stmt = select(Domain).where(Domain.project_id == project_id)
    if spec.min_risk is not None:
        stmt = stmt.where(Domain.risk_score >= spec.min_risk)
    if spec.date_from:
        stmt = stmt.where(Domain.last_seen >= spec.date_from)
    if spec.date_to:
        stmt = stmt.where(Domain.first_seen < spec.date_to)

    orderings = {
        # NULLS LAST on every descending sort: an unscored domain belongs at the
        # bottom of a risk list, and NULL sorts first on DESC by default.
        "risk": (Domain.risk_score.desc().nullslast(),),
        "posts": (Domain.post_count.desc(),),
        "recent": (Domain.last_seen.desc().nullslast(),),
        # Newest registration first -- a domain registered last week and already
        # carrying a narrative is the interesting case.
        "age": (Domain.whois_created_at.desc().nullslast(),),
    }
    stmt = stmt.order_by(*orderings.get(sort, orderings["risk"]), Domain.domain)

    offset = decode_cursor(page.cursor).get("o", 0)
    total = await count_rows(session, stmt)
    rows = (await session.execute(stmt.offset(offset).limit(page.limit))).scalars().all()

    return PageResponse[DomainOut](
        items=[to_domain_out(row) for row in rows],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=total,
        filters_applied={**spec.as_response_dict(), "sort": sort},
    )


async def get_domain(session: AsyncSession, project_id: str, domain: str) -> DomainOut:
    from app.models.domains import Domain

    resolved = await resolve_project_id(session, project_id)
    # Lowercased on the way in, matching how the roll-up keys them: three
    # subdomains of one site are one actor, not three.
    row = await session.get(Domain, (domain.lower(), resolved))
    if row is None:
        raise NotFound(f"No domain {domain!r} in this project.", code="domain_not_found")
    return to_domain_out(row)


async def domain_narratives(
    session: AsyncSession, project_id: str, domain: str
) -> list[DomainNarrativeLink]:
    resolved = await resolve_project_id(session, project_id)
    rows = (
        await session.execute(
            text(
                """
                SELECT n.id, n.title, count(*) AS posts,
                       min(p."timestamp") AS first_seen, max(p."timestamp") AS last_seen
                FROM posts p
                CROSS JOIN LATERAL unnest(p.domains) AS d(domain)
                JOIN narrative_posts np ON np.post_id = p.id
                JOIN narratives n ON n.id = np.narrative_id
                WHERE p.project_id = :p AND lower(d.domain) = :d
                GROUP BY n.id, n.title
                ORDER BY count(*) DESC
                """
            ),
            {"p": resolved, "d": domain.lower()},
        )
    ).all()

    total = sum(row.posts for row in rows) or 1
    return [
        DomainNarrativeLink(
            narrative_id=str(row.id),
            title=row.title,
            post_count=row.posts,
            share_pct=round(100 * row.posts / total, 2),
            first_seen=row.first_seen,
            last_seen=row.last_seen,
        )
        for row in rows
    ]
