"""Ingestion.

These routes are thin wrappers over Phase 1's existing adapters. Phase 4 does
not know how to talk to Mastodon, and if a request body here ever grows a
`subreddit` field then something has gone wrong architecturally.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import Pagination, RequireRead, RequireWrite, rate_limit
from app.errors import NotFound
from app.mock import responses as mock
from app.schemas.common import JobAccepted, PageResponse
from app.schemas.ingest import IngestRequest, IngestRunOut

router = APIRouter(tags=["ingest"], dependencies=[Depends(rate_limit)])


@router.post(
    "/ingest",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Fetch and/or load a corpus",
    description=(
        "Returns in well under 200ms with a job id; the work happens in the worker. "
        "`mode=fetch` runs the Phase 1 adapters to Parquet, `mode=load` bulk-copies "
        "existing Parquet into Postgres, `mode=both` chains them."
    ),
)
async def start_ingest(
    body: IngestRequest,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobAccepted:
    if mock.demo_mode():
        job = mock.job("ingest.load", "pending")
        return JobAccepted(
            job_id=job.id,
            status="pending",
            status_url=f"/api/v1/jobs/{job.id}",
            kind="ingest.load",
            message="DEMO_MODE: no work was enqueued.",
        )
    from app.services.jobs import enqueue

    kind = "ingest.fetch" if body.mode in ("fetch", "both") else "ingest.load"
    return await enqueue(
        session, kind=kind, project_id=body.project_id, params=body.model_dump(mode="json")
    )


@router.get("/ingest/runs", response_model=PageResponse[IngestRunOut], summary="Ingestion history")
async def list_ingest_runs(
    principal: RequireRead,
    project_id: str,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[IngestRunOut]:
    if mock.demo_mode():
        return mock.paginate(mock.ingest_runs(), page, {"project_id": project_id})
    from app.repositories.ingest import list_runs

    return await list_runs(session, project_id, page)


@router.get(
    "/ingest/runs/{run_id}",
    response_model=IngestRunOut,
    summary="One ingestion run",
    description=(
        "`rejection_reasons` accounts for every row that did not make it into "
        "Postgres, by reason code. Silent data loss between Parquet and Postgres "
        "would poison every downstream metric and surface weeks later as a number "
        "nobody can explain."
    ),
)
async def get_ingest_run(
    run_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> IngestRunOut:
    if mock.demo_mode():
        for run in mock.ingest_runs():
            if run.id == run_id:
                return run
        raise NotFound(f"No ingest run with id {run_id}.", code="ingest_run_not_found")
    from app.repositories.ingest import get_run

    return await get_run(session, run_id)
