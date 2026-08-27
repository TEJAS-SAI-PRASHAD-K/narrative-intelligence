"""Report queries and the download handler."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import Page
from app.errors import ApiError, NotFound
from app.repositories.filters import decode_cursor, encode_cursor, resolve_project_id
from app.schemas.common import PageResponse
from app.schemas.reports import ReportOut

log = logging.getLogger(__name__)

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "csv": "text/csv",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def to_report_out(row) -> ReportOut:
    return ReportOut(
        id=str(row.id),
        project_id=str(row.project_id),
        template=row.template,
        format=row.format,
        status=row.status,
        params=row.params or {},
        size_bytes=row.size_bytes,
        # Null until the render succeeds, so the UI cannot offer a download
        # button for a file that does not exist yet.
        download_url=(f"/api/v1/reports/{row.id}/download" if row.status == "succeeded" else None),
        error=row.error,
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


async def list_reports(
    session: AsyncSession, project_id: str, page: Page
) -> PageResponse[ReportOut]:
    from app.models.ops import Report

    resolved = await resolve_project_id(session, project_id)
    offset = decode_cursor(page.cursor).get("o", 0)
    rows = (
        (
            await session.execute(
                select(Report)
                .where(Report.project_id == resolved)
                .order_by(Report.created_at.desc())
                .offset(offset)
                .limit(page.limit)
            )
        )
        .scalars()
        .all()
    )
    return PageResponse[ReportOut](
        items=[to_report_out(row) for row in rows],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=None,
        filters_applied={"project_id": project_id},
    )


async def report_file(session: AsyncSession, report_id: str) -> Response:
    """Stream a rendered report.

    Three distinct failures, three distinct codes: no such report is 404, a
    render still running or failed is 409 (the resource exists, its file does
    not yet), and a row whose file has vanished from disk is 410 rather than a
    500 -- the record is intact, the artifact is gone, and saying so tells the
    operator to re-render instead of to check the logs.
    """
    from app.models.ops import Report

    row = await session.get(Report, report_id)
    if row is None:
        raise NotFound(f"No report with id {report_id}.", code="report_not_found")

    if row.status != "succeeded":
        raise ApiError(
            f"Report {report_id} is {row.status!r}; there is no file to download yet."
            + (f" Error: {row.error}" if row.error else ""),
            code="report_not_ready",
            status_code=409,
            detail={"status": row.status, "job_status_url": f"/api/v1/reports?id={report_id}"},
        )

    path = Path(row.file_path) if row.file_path else None
    if path is None or not path.exists():
        raise ApiError(
            "The report record exists but its file is no longer on disk. Re-render it.",
            code="report_file_missing",
            status_code=410,
            detail={"report_id": report_id},
        )

    filename = f"{row.template}-{report_id[:8]}.{row.format}"
    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(row.format, "application/octet-stream"),
        filename=filename,
    )
