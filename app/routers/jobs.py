"""Job status.

One endpoint shape for every async operation. The frontend has one polling hook
and one progress component regardless of whether it started an ingest, a
recluster or a deepfake check, which is the entire reason the ``jobs`` table is
unified rather than per-task.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import Pagination, RequireRead, RequireWrite, rate_limit
from app.mock import responses as mock
from app.schemas.common import PageResponse
from app.schemas.jobs import JobCancelled, JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(rate_limit)])


@router.get("", response_model=PageResponse[JobOut], summary="List jobs")
async def list_jobs(
    principal: RequireRead,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
    project_id: str | None = None,
    kind: str | None = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
) -> PageResponse[JobOut]:
    if mock.demo_mode():
        items = mock.jobs_list()
        if kind:
            items = [j for j in items if j.kind == kind]
        if status_filter:
            items = [j for j in items if j.status == status_filter]
        return mock.paginate(
            items, page, {"project_id": project_id, "kind": kind, "status": status_filter}
        )
    from app.repositories.jobs import list_jobs as query_jobs

    return await query_jobs(session, page, project_id=project_id, kind=kind, status=status_filter)


@router.get("/{job_id}", response_model=JobOut, summary="Job status")
async def get_job(
    job_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobOut:
    if mock.demo_mode():
        for job in mock.jobs_list():
            if job.id == job_id:
                return job
        return mock.job("ingest.load", "succeeded", job_id=job_id)
    from app.repositories.jobs import get_job as query_job

    return await query_job(session, job_id)


@router.post(
    "/{job_id}/cancel",
    response_model=JobCancelled,
    summary="Cancel a job",
    description=(
        "Cooperative: a running task stops at its next checkpoint rather than being "
        "killed mid-write, so a cancelled job can legitimately report partial progress "
        "that is already durably committed."
    ),
)
async def cancel_job(
    job_id: str,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobCancelled:
    if mock.demo_mode():
        return JobCancelled(
            id=job_id,
            status="cancelled",
            detail="DEMO_MODE: no work was running.",
        )
    from app.services.jobs import cancel

    return await cancel(session, job_id)
