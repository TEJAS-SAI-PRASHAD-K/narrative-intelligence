"""The task base class that owns the ``jobs`` row.

Every task in this system reports through one table so the frontend polls one
endpoint shape. Getting that bookkeeping right in each task by hand would mean
getting it wrong in one of them, so it lives here:

* the row moves pending -> running -> succeeded/failed exactly once;
* ``progress`` is written at checkpoints, and a resumed task continues from
  where it stopped rather than restarting at zero -- resetting would make a
  resume indistinguishable from a restart in the UI;
* a cancellation request is *cooperative*: the task checks at its checkpoints
  and stops cleanly, because killing a worker mid-COPY leaves half a batch;
* an exception is recorded on the row before it is re-raised, so a failed job
  explains itself without anybody reading worker logs.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any

from celery import Task

from app.db import sync_session, utcnow

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised at a checkpoint when an operator asked the task to stop."""


class TrackedTask(Task):
    """A Celery task that keeps its ``jobs`` row current."""

    #: Set by the task body via ``self.job_id``; injected by the caller.
    _job_id: uuid.UUID | None = None

    def before_start(self, task_id: str, args, kwargs) -> None:  # noqa: D102
        job_id = kwargs.get("job_id")
        if not job_id:
            return
        self._job_id = uuid.UUID(str(job_id))
        with sync_session() as session:
            job = self._load(session, self._job_id)
            if job is None:
                return
            job.celery_task_id = task_id
            job.status = "running"
            job.started_at = job.started_at or utcnow()
            job.attempts = (job.attempts or 0) + 1
            job.error = None

    def on_success(self, retval, task_id, args, kwargs) -> None:  # noqa: D102
        job_id = kwargs.get("job_id")
        if not job_id:
            return
        with sync_session() as session:
            job = self._load(session, uuid.UUID(str(job_id)))
            if job is None:
                return
            job.status = "succeeded"
            job.progress = 1.0
            job.finished_at = utcnow()
            job.result = retval if isinstance(retval, dict) else {"result": retval}

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:  # noqa: D102
        job_id = kwargs.get("job_id")
        if not job_id:
            return
        cancelled = isinstance(exc, JobCancelled)
        with sync_session() as session:
            job = self._load(session, uuid.UUID(str(job_id)))
            if job is None:
                return
            job.status = "cancelled" if cancelled else "failed"
            job.finished_at = utcnow()
            # The message, not the whole traceback: the traceback is in the
            # worker log with the celery task id, and a multi-kilobyte stack in
            # a JSON field the UI renders is unreadable.
            job.error = (
                "cancelled at a checkpoint" if cancelled else f"{type(exc).__name__}: {exc}"[:2000]
            )
            if not cancelled:
                log.error(
                    "task %s failed job=%s: %s\n%s",
                    self.name,
                    job_id,
                    exc,
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                )

    @staticmethod
    def _load(session, job_id: uuid.UUID):
        from app.models.ops import Job

        return session.get(Job, job_id)


def report_progress(job_id: Any, fraction: float, *, result: dict | None = None) -> None:
    """Write a checkpoint. Also the cancellation check.

    Combining the two is deliberate: every place a task is far enough along to
    report progress is a place it is safe to stop, and a separate
    ``check_cancelled()`` would inevitably be forgotten in one loop.
    """
    if not job_id:
        return
    with sync_session() as session:
        job = TrackedTask._load(session, uuid.UUID(str(job_id)))
        if job is None:
            return
        if job.cancel_requested:
            raise JobCancelled(f"job {job_id} was cancelled by an operator")
        # Monotonic. A resumed task reports where it actually is, and a retry
        # that starts a batch again must not appear to go backwards.
        job.progress = max(job.progress or 0.0, min(1.0, float(fraction)))
        if result:
            job.result = {**(job.result or {}), **result}
