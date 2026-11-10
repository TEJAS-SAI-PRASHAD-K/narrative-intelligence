"""Maintenance tasks.

The upload sweep exists for one reason. An earlier version of this service accepted
uploaded images of real people and kept them for a limited time. That upload route
is gone, but a deployment can still hold files that it uploaded before the removal.
The sweep clears those files when they pass the retention window. After every
deployment has passed one retention window, this task has nothing left to delete and
can go.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.config import get_api_settings
from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery
from ingest.config import REPO_ROOT

log = logging.getLogger(__name__)

#: Reason codes for an entry that the sweep could not handle. Each one is counted in
#: the task result and logged, and the next run tries the entry again.
REASON_DELETE_FAILED = "delete_failed"
REASON_STAT_FAILED = "stat_failed"
REASON_LIST_FAILED = "uploads_dir_unreadable"

#: Reason codes for a run that the guard refused. A refused run deletes nothing.
REASON_INVALID_RETENTION = "invalid_retention"
REASON_UNSAFE_DIR = "unsafe_uploads_dir"

#: A shorter window would delete files that a client uploaded moments ago.
MIN_RETENTION_HOURS = 1


@celery.task(name="maint.purge_uploads", base=TrackedTask, bind=True)
def purge_uploads(self, *, job_id: str | None = None, **params: Any) -> dict:
    """Delete every entry under ``uploads_dir`` that is older than the retention window.

    The sweep reads only the file system. It uses no database, so it keeps working
    after the table that once recorded each upload deadline is gone.
    """
    settings = get_api_settings()
    result = sweep_uploads(
        settings.uploads_dir,
        settings.media_retention_hours,
        protected_dirs=(settings.data_dir, settings.models_dir, settings.reports_dir),
    )
    result["retention_hours"] = settings.media_retention_hours
    if result["deleted"] or result["failed"]:
        log.info(
            "upload sweep: deleted=%d failed=%d bytes_freed=%d",
            result["deleted"],
            result["failed"],
            result["bytes_freed"],
        )
    return result


def sweep_uploads(
    uploads_dir: Path,
    retention_hours: int,
    *,
    protected_dirs: Iterable[Path] = (),
    now: float | None = None,
) -> dict:
    """Delete the direct children of ``uploads_dir`` whose mtime passed the cutoff.

    A child can be a file or a directory. A symbolic link is removed as a link and
    its target is never followed. An entry that cannot be examined or deleted is
    logged, counted under a reason code and left in place for the next run.

    Before any deletion, a guard refuses a retention below one hour and a root that
    equals or contains the repository, the user home, the file system root or any of
    ``protected_dirs``. A refused run deletes nothing, logs an error and returns the
    reason code. It does not raise.
    """
    refusal = _refusal(uploads_dir, retention_hours, protected_dirs)
    if refusal is not None:
        reason, detail = refusal
        log.error("upload sweep refused: %s: %s", reason, detail)
        return {"deleted": 0, "failed": 1, "reasons": {reason: 1}, "bytes_freed": 0}

    deleted = 0
    bytes_freed = 0
    reasons: dict[str, int] = {}

    def fail(reason: str, path: Path, exc: OSError) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        log.error("upload sweep: %s for %s: %s", reason, path, exc)

    if not uploads_dir.exists():
        return {"deleted": 0, "failed": 0, "reasons": reasons, "bytes_freed": 0}

    cutoff = (time.time() if now is None else now) - retention_hours * 3600
    try:
        children = list(uploads_dir.iterdir())
    except OSError as exc:
        fail(REASON_LIST_FAILED, uploads_dir, exc)
        return {"deleted": 0, "failed": 1, "reasons": reasons, "bytes_freed": 0}

    for child in children:
        try:
            if child.lstat().st_mtime >= cutoff:
                continue
            size = _tree_size(child)
        except OSError as exc:
            fail(REASON_STAT_FAILED, child, exc)
            continue
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            fail(REASON_DELETE_FAILED, child, exc)
            continue
        deleted += 1
        bytes_freed += size

    return {
        "deleted": deleted,
        "failed": sum(reasons.values()),
        "reasons": reasons,
        "bytes_freed": bytes_freed,
    }


def _refusal(
    uploads_dir: Path, retention_hours: int, protected_dirs: Iterable[Path]
) -> tuple[str, str] | None:
    """Return ``(reason, detail)`` when the sweep must not run, else ``None``."""
    if retention_hours < MIN_RETENTION_HOURS:
        return REASON_INVALID_RETENTION, f"retention {retention_hours}h is below 1h"
    root = uploads_dir.resolve()
    if root == Path(root.anchor):
        return REASON_UNSAFE_DIR, f"{root} is the file system root"
    guarded = [REPO_ROOT, Path.home(), *protected_dirs]
    for path in guarded:
        if path.resolve().is_relative_to(root):
            return REASON_UNSAFE_DIR, f"{root} equals or contains {path}"
    return None


def _tree_size(path: Path) -> int:
    """Total size in bytes of a file, or of every file below a directory."""
    if path.is_symlink() or not path.is_dir():
        return path.lstat().st_size
    return sum(
        item.lstat().st_size for item in path.rglob("*") if not item.is_dir() or item.is_symlink()
    )
