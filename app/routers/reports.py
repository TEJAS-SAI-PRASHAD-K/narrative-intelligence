"""Report rendering and download."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import Pagination, RequireRead, RequireWrite, rate_limit
from app.mock import responses as mock
from app.schemas.common import JobAccepted, PageResponse
from app.schemas.reports import ReportCreate, ReportOut

router = APIRouter(prefix="/reports", tags=["reports"], dependencies=[Depends(rate_limit)])


@router.post(
    "",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Render a report",
)
async def create_report(
    body: ReportCreate,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobAccepted:
    if mock.demo_mode():
        job = mock.job("reports.render", "pending")
        return JobAccepted(
            job_id=job.id,
            status="pending",
            status_url=f"/api/v1/jobs/{job.id}",
            kind="reports.render",
            message="DEMO_MODE: no work was enqueued.",
        )
    from app.services.jobs import enqueue

    return await enqueue(
        session, kind="reports.render", project_id=body.project_id, params=body.model_dump()
    )


@router.get("", response_model=PageResponse[ReportOut], summary="List rendered reports")
async def list_reports(
    principal: RequireRead,
    project_id: str,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[ReportOut]:
    if mock.demo_mode():
        return mock.paginate(mock.reports_list(), page, {"project_id": project_id})
    from app.repositories.reports import list_reports as query_reports

    return await query_reports(session, project_id, page)


@router.get(
    "/{report_id}/download",
    summary="Download a rendered report",
    response_class=Response,
    responses={
        200: {"content": {"application/pdf": {}, "text/csv": {}}, "description": "The file."},
        404: {"description": "No such report."},
        409: {"description": "The render has not finished."},
    },
)
async def download_report(
    report_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    if mock.demo_mode():
        # A tiny real CSV rather than a stub body: a frontend developer wiring
        # the download button needs the Content-Disposition round-trip to work,
        # and a JSON placeholder would not exercise it.
        body = "narrative_id,title,fusion_score,priority\n" + "\n".join(
            f'{n.id},"{n.title}",{n.scorecard["fusion_score"]},{n.scorecard["priority"]}'
            for n in mock.corpus().narratives
        )
        return Response(
            content=body,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="report-{report_id}.csv"'},
        )
    from app.repositories.reports import report_file

    return await report_file(session, report_id)
