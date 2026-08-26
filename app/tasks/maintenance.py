"""Maintenance tasks.

The purge is not housekeeping. Uploaded media is imagery of real people's faces,
and a retention window that is only a config value is a promise nobody can
verify was kept. This task is what makes the deletion time the API reports true.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from app.config import get_api_settings
from app.db import sync_session, utcnow
from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="maint.purge_uploads", base=TrackedTask, bind=True)
def purge_uploads(self, *, job_id: str | None = None, **params: Any) -> dict:
    """Delete uploaded media past its retention deadline.

    The verdict row survives; only the file goes. An analyst needs to be able to
    cite a past check, and the explanation and confidence are the citable part.
    Keeping the video itself indefinitely to support that is not defensible.
    """
    from sqlalchemy import select

    from app.models.ops import MediaCheck

    settings = get_api_settings()
    now = utcnow()
    deleted, freed, missing = 0, 0, 0

    with sync_session() as session:
        rows = (
            session.execute(
                select(MediaCheck).where(
                    MediaCheck.purged_at.is_(None),
                    MediaCheck.storage_path.isnot(None),
                    MediaCheck.deletes_at.isnot(None),
                    MediaCheck.deletes_at <= now,
                )
            )
            .scalars()
            .all()
        )

        for row in rows:
            path = Path(row.storage_path)
            try:
                if path.exists():
                    freed += path.stat().st_size
                    path.unlink()
                    # The upload directory is one per job, so remove the empty
                    # shell too rather than accumulating thousands of them.
                    if path.parent.is_dir() and not any(path.parent.iterdir()):
                        path.parent.rmdir()
                    deleted += 1
                else:
                    missing += 1
            except OSError as exc:
                # A file that cannot be deleted must stay marked unpurged so the
                # next run tries again. Marking it purged would quietly leave
                # imagery on disk while the record claims otherwise.
                log.error("could not purge %s: %s", path, exc)
                continue
            row.purged_at = now
            row.storage_path = None

    # Orphans: directories from uploads that never produced a MediaCheck row,
    # e.g. a request that failed validation after the file was written.
    orphans = _purge_orphan_directories(settings.uploads_dir, settings.media_retention_hours)

    result = {
        "deleted": deleted,
        "already_missing": missing,
        "orphan_directories_removed": orphans,
        "bytes_freed": freed,
        "retention_hours": settings.media_retention_hours,
    }
    if deleted or orphans:
        log.info("purged %d upload(s) and %d orphan dir(s)", deleted, orphans)
    return result


def _purge_orphan_directories(uploads_dir: Path, retention_hours: int) -> int:
    import time

    if not uploads_dir.exists():
        return 0
    cutoff = time.time() - retention_hours * 3600
    removed = 0
    for child in uploads_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
                removed += 1
        except OSError as exc:
            log.warning("could not remove orphan upload dir %s: %s", child, exc)
    return removed
