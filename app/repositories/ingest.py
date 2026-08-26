"""Ingest-run queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import Page
from app.errors import NotFound
from app.repositories.filters import decode_cursor, encode_cursor, resolve_project_id
from app.schemas.common import PageResponse
from app.schemas.ingest import IngestRunOut


def to_run_out(row) -> IngestRunOut:
    return IngestRunOut(
        id=str(row.id),
        project_id=str(row.project_id),
        source=row.source,
        status=row.status,
        mode=row.mode,
        records_in=row.records_in,
        records_loaded=row.records_loaded,
        records_rejected=row.records_rejected,
        rejection_reasons=row.rejection_reasons or {},
        rejects_path=row.rejects_path,
        manifest_sha=row.manifest_sha,
        detail=row.detail,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


async def list_runs(
    session: AsyncSession, project_id: str, page: Page
) -> PageResponse[IngestRunOut]:
    from app.models.ops import IngestRun

    resolved = await resolve_project_id(session, project_id)
    offset = decode_cursor(page.cursor).get("o", 0)
    rows = (
        (
            await session.execute(
                select(IngestRun)
                .where(IngestRun.project_id == resolved)
                .order_by(IngestRun.started_at.desc().nullslast(), IngestRun.id.desc())
                .offset(offset)
                .limit(page.limit)
            )
        )
        .scalars()
        .all()
    )
    return PageResponse[IngestRunOut](
        items=[to_run_out(row) for row in rows],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=None,
        filters_applied={"project_id": project_id},
    )


async def get_run(session: AsyncSession, run_id: str) -> IngestRunOut:
    from app.models.ops import IngestRun

    row = await session.get(IngestRun, run_id)
    if row is None:
        raise NotFound(f"No ingest run with id {run_id}.", code="ingest_run_not_found")
    return to_run_out(row)
