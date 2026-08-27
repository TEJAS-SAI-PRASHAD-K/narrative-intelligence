"""Deepfake check queries."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_api_settings
from app.deps import Page
from app.errors import NotFound
from app.repositories.filters import decode_cursor, encode_cursor
from app.schemas.common import PageResponse
from app.schemas.media import MediaCheckOut

log = logging.getLogger(__name__)


def to_check_out(row, job=None) -> MediaCheckOut:
    settings = get_api_settings()
    return MediaCheckOut(
        # The job id is what the client was handed by POST /media/check, so it
        # is what they poll with. Falls back to the row id for checks imported
        # from Phase 3, which never had a job.
        job_id=str(row.job_id or row.id),
        status=(job.status if job else ("succeeded" if row.completed_at else "pending")),
        filename=row.filename,
        media_type=row.media_type,
        size_bytes=row.size_bytes,
        submitted_at=row.submitted_at,
        completed_at=row.completed_at,
        verdict=row.verdict,
        confidence=row.confidence,
        manipulation_type=row.manipulation_type,
        frames_analyzed=row.frames_analyzed,
        face_detected=row.face_detected,
        explanation=row.explanation,
        model=row.model,
        model_version=row.model_version,
        limitations=list(row.limitations or ()),
        post_id=row.post_id,
        error=(job.error if job else None),
        retention={
            "deletes_at": row.deletes_at.isoformat() if row.deletes_at else None,
            "retention_hours": settings.media_retention_hours,
            "purged_at": row.purged_at.isoformat() if row.purged_at else None,
            # Checks over corpus media were never uploaded, so there is nothing
            # to purge and the UI should not imply a countdown.
            "file_retained": bool(row.storage_path),
        },
    )


async def list_checks(session: AsyncSession, page: Page) -> PageResponse[MediaCheckOut]:
    from app.models.ops import MediaCheck

    offset = decode_cursor(page.cursor).get("o", 0)
    rows = (
        (
            await session.execute(
                select(MediaCheck)
                .order_by(MediaCheck.submitted_at.desc())
                .offset(offset)
                .limit(page.limit)
            )
        )
        .scalars()
        .all()
    )
    return PageResponse[MediaCheckOut](
        items=[to_check_out(row) for row in rows],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=None,
        filters_applied={},
    )


async def get_check(session: AsyncSession, job_id: str) -> MediaCheckOut:
    """Look up by job id, which is what the 202 handed the caller."""
    from app.models.ops import Job, MediaCheck

    row = (
        await session.execute(select(MediaCheck).where(MediaCheck.job_id == job_id))
    ).scalar_one_or_none()

    if row is None:
        # The job may exist while the task is still running and has not yet
        # written its MediaCheck row. Reporting the job's state is far more
        # useful than a 404 to a client that is polling exactly as told.
        job = await session.get(Job, job_id)
        if job is None or job.kind != "media.deepfake":
            raise NotFound(f"No media check with job id {job_id}.", code="media_check_not_found")
        params = job.params or {}
        settings = get_api_settings()
        return MediaCheckOut(
            job_id=str(job.id),
            status=job.status,
            filename=params.get("filename"),
            media_type=params.get("media_type"),
            size_bytes=params.get("size_bytes"),
            submitted_at=job.created_at,
            completed_at=job.finished_at,
            error=job.error,
            retention={"retention_hours": settings.media_retention_hours},
        )

    job = await session.get(Job, row.job_id) if row.job_id else None
    return to_check_out(row, job)
