"""Domain risk -- the third pillar.

Enrichment is a separate, rate-limited, best-effort task. Nothing on this router
ever blocks on a WHOIS lookup: an un-enriched domain still gets a risk score from
in-corpus signals and says `enrichment_status: unavailable`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import Filters, Pagination, RequireRead, RequireWrite, rate_limit
from app.errors import NotFound
from app.mock import responses as mock
from app.schemas.common import JobAccepted, PageResponse
from app.schemas.domains import DomainNarrativeLink, DomainOut

router = APIRouter(prefix="/domains", tags=["domains"], dependencies=[Depends(rate_limit)])


@router.get("", response_model=PageResponse[DomainOut], summary="List domains by risk")
async def list_domains(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
    sort: Annotated[str, Query(pattern="^(risk|posts|recent|age)$")] = "risk",
) -> PageResponse[DomainOut]:
    if mock.demo_mode():
        rows = list(mock.corpus().domains)
        sorters = {
            "risk": lambda d: -d["risk_score"],
            "posts": lambda d: -d["post_count"],
            "recent": lambda d: -d["last_seen"].timestamp(),
            "age": lambda d: (
                d["whois_created_at"].timestamp() if d["whois_created_at"] else float("inf")
            ),
        }
        rows.sort(key=sorters[sort])
        if filters.min_risk is not None:
            rows = [d for d in rows if d["risk_score"] >= filters.min_risk]
        return mock.paginate(
            [mock.to_domain(d) for d in rows], page, {**filters.as_response_dict(), "sort": sort}
        )
    from app.repositories.domains import list_domains as query_domains

    return await query_domains(session, filters, page, sort=sort)


@router.get("/{domain}", response_model=DomainOut, summary="Domain detail")
async def get_domain(
    domain: str,
    principal: RequireRead,
    project_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DomainOut:
    if mock.demo_mode():
        row = mock.corpus().by_domain.get(domain.lower())
        if row is None:
            raise NotFound(f"No domain {domain!r} in this project.", code="domain_not_found")
        return mock.to_domain(row)
    from app.repositories.domains import get_domain as query_domain

    return await query_domain(session, project_id, domain)


@router.get(
    "/{domain}/narratives",
    response_model=list[DomainNarrativeLink],
    summary="Narratives linking this domain",
)
async def domain_narratives(
    domain: str,
    principal: RequireRead,
    project_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[DomainNarrativeLink]:
    if mock.demo_mode():
        c = mock.corpus()
        if domain.lower() not in c.by_domain:
            raise NotFound(f"No domain {domain!r} in this project.", code="domain_not_found")
        counts: dict[str, int] = {}
        for post in c.posts:
            if domain.lower() in post.domains:
                for narrative_id in post.narrative_ids:
                    counts[narrative_id] = counts.get(narrative_id, 0) + 1
        total = sum(counts.values()) or 1
        return [
            DomainNarrativeLink(
                narrative_id=nid,
                title=c.by_narrative[nid].title,
                post_count=count,
                share_pct=round(100 * count / total, 2),
                first_seen=c.by_narrative[nid].date_start,
                last_seen=c.by_narrative[nid].date_end,
            )
            for nid, count in sorted(counts.items(), key=lambda kv: -kv[1])
        ]
    from app.repositories.domains import domain_narratives as query_links

    return await query_links(session, project_id, domain)


@router.post(
    "/{domain}/enrich",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue WHOIS/TLS enrichment",
    description=(
        "Best-effort and rate-limited. A failure marks the domain "
        "`enrichment_status: failed` and leaves its in-corpus risk score intact."
    ),
)
async def enrich_domain(
    domain: str,
    principal: RequireWrite,
    project_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobAccepted:
    if mock.demo_mode():
        job = mock.job("domain.enrich", "pending")
        return JobAccepted(
            job_id=job.id,
            status="pending",
            status_url=f"/api/v1/jobs/{job.id}",
            kind="domain.enrich",
            message="DEMO_MODE: no work was enqueued.",
        )
    from app.services.jobs import enqueue

    return await enqueue(
        session, kind="domain.enrich", project_id=project_id, params={"domain": domain.lower()}
    )
