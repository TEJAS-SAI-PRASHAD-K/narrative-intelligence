"""Network graph, layouts and saved comparisons -- the fourth pillar.

``max_nodes`` has a hard server-side ceiling. Exceeding it truncates by
descending node degree and sets ``truncated: true`` with the applied threshold
and the dropped counts. Never silently: a graph that quietly loses its periphery
makes a coordinated cluster look more isolated than it is, which is precisely
the wrong error for a coordination-detection product to make.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_api_settings
from app.db import get_session
from app.deps import Filters, Pagination, RequireRead, RequireWrite, rate_limit
from app.errors import BadRequest, NotFound
from app.mock import responses as mock
from app.schemas.common import DeletedResponse, JobAccepted, PageResponse
from app.schemas.network import (
    ComparisonCreate,
    ComparisonOut,
    GraphResponse,
    LayoutRequest,
)

router = APIRouter(tags=["network"], dependencies=[Depends(rate_limit)])


@router.get(
    "/network/graph",
    response_model=GraphResponse,
    summary="The interaction graph",
    description=(
        "Nodes carry precomputed positions so the browser never runs a force layout "
        "over 20,000 nodes. `stats.truncated` and `stats.truncation` report exactly "
        "what was dropped and by what rule whenever the ceiling bites."
    ),
)
async def network_graph(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
    narrative_id: str | None = None,
    comparison_id: str | None = None,
    max_nodes: Annotated[int | None, Query(ge=10)] = None,
    bucket: Annotated[str, Query(pattern="^\\d+h$")] = "12h",
    hide_standalone: bool = False,
) -> GraphResponse:
    settings = get_api_settings()
    if narrative_id and comparison_id:
        raise BadRequest(
            "Pass narrative_id or comparison_id, not both.", code="ambiguous_graph_scope"
        )
    # The ceiling is a server-side maximum, not a default the caller can raise.
    # A client asking for 200,000 nodes gets 20,000 and is told so, rather than
    # getting a response that takes forty seconds and locks up their browser.
    effective_max = min(max_nodes or settings.max_graph_nodes, settings.max_graph_nodes)

    if mock.demo_mode():
        if narrative_id and narrative_id not in mock.corpus().by_narrative:
            raise NotFound(f"No narrative with id {narrative_id}.", code="narrative_not_found")
        return mock.network_graph(filters, narrative_id, effective_max, bucket, hide_standalone)

    from app.repositories.network import build_graph

    return await build_graph(
        session,
        filters,
        narrative_id=narrative_id,
        comparison_id=comparison_id,
        max_nodes=effective_max,
        bucket=bucket,
        hide_standalone=hide_standalone,
    )


@router.post(
    "/network/layout",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Precompute a graph layout",
)
async def precompute_layout(
    body: LayoutRequest,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobAccepted:
    if mock.demo_mode():
        job = mock.job("graph.layout", "pending")
        return JobAccepted(
            job_id=job.id,
            status="pending",
            status_url=f"/api/v1/jobs/{job.id}",
            kind="graph.layout",
            message="DEMO_MODE: no work was enqueued.",
        )
    from app.services.jobs import enqueue

    return await enqueue(
        session,
        kind="graph.layout",
        project_id=body.project_id,
        params=body.model_dump(exclude_none=True),
    )


# --- comparisons -----------------------------------------------------------
@router.get("/comparisons", response_model=PageResponse[ComparisonOut], summary="Saved comparisons")
async def list_comparisons(
    principal: RequireRead,
    project_id: str,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[ComparisonOut]:
    if mock.demo_mode():
        return mock.paginate(mock.comparisons_list(), page, {"project_id": project_id})
    from app.repositories.network import list_comparisons as query_comparisons

    return await query_comparisons(session, project_id, page)


@router.post(
    "/comparisons",
    response_model=ComparisonOut,
    status_code=status.HTTP_201_CREATED,
    summary="Save a comparison",
)
async def create_comparison(
    body: ComparisonCreate,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ComparisonOut:
    if mock.demo_mode():
        return mock.comparisons_list()[0]
    from app.repositories.network import create_comparison as save_comparison

    return await save_comparison(session, body)


@router.get("/comparisons/{comparison_id}", response_model=ComparisonOut, summary="A comparison")
async def get_comparison(
    comparison_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ComparisonOut:
    if mock.demo_mode():
        for item in mock.comparisons_list():
            if item.id == comparison_id:
                return item
        raise NotFound(f"No comparison with id {comparison_id}.", code="comparison_not_found")
    from app.repositories.network import get_comparison as query_comparison

    return await query_comparison(session, comparison_id)


@router.delete(
    "/comparisons/{comparison_id}", response_model=DeletedResponse, summary="Delete a comparison"
)
async def delete_comparison(
    comparison_id: str,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DeletedResponse:
    if mock.demo_mode():
        return DeletedResponse(id=comparison_id)
    from app.repositories.network import delete_comparison as remove_comparison

    await remove_comparison(session, comparison_id)
    return DeletedResponse(id=comparison_id)
