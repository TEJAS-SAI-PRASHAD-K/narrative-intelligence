"""Job queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import Page
from app.errors import NotFound
from app.repositories.filters import decode_cursor, encode_cursor, resolve_project_id
from app.schemas.common import PageResponse
from app.schemas.jobs import JobOut
from app.services.jobs import to_job_out


async def list_jobs(
    session: AsyncSession,
    page: Page,
    *,
    project_id: str | None = None,
    kind: str | None = None,
    status: str | None = None,
) -> PageResponse[JobOut]:
    from app.models.ops import Job

    stmt = select(Job).order_by(Job.created_at.desc(), Job.id.desc())
    if project_id:
        stmt = stmt.where(Job.project_id == await resolve_project_id(session, project_id))
    if kind:
        stmt = stmt.where(Job.kind == kind)
    if status:
        stmt = stmt.where(Job.status == status)

    offset = decode_cursor(page.cursor).get("o", 0)
    rows = (await session.execute(stmt.offset(offset).limit(page.limit))).scalars().all()
    return PageResponse[JobOut](
        items=[to_job_out(row) for row in rows],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=None,
        filters_applied={"project_id": project_id, "kind": kind, "status": status},
    )


async def get_job(session: AsyncSession, job_id: str) -> JobOut:
    from app.models.ops import Job

    row = await session.get(Job, job_id)
    if row is None:
        raise NotFound(f"No job with id {job_id}.", code="job_not_found")
    return to_job_out(row)
