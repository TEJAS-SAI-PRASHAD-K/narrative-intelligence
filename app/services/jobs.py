"""Job lifecycle: create the row, dispatch the task, report status.

The row is created **before** the task is dispatched and committed
independently of it. That ordering matters: if the broker is down, the caller
still gets a job id and a row that says `failed` with the reason, instead of a
202 pointing at a job that never existed.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import utcnow
from app.errors import BadRequest, NotFound
from app.schemas.common import JobAccepted
from app.schemas.jobs import JobCancelled, JobOut

log = logging.getLogger(__name__)

#: Job kind -> the Celery task name that serves it. One dict rather than a
#: convention, so a typo is a KeyError at enqueue time and not a task that sits
#: in the queue forever because nothing is registered to consume it.
TASK_NAMES: dict[str, str] = {
    "ingest.fetch": "ingest.fetch",
    "ingest.load": "ingest.load",
    "nlp.embed": "nlp.embed",
    "nlp.cluster": "nlp.cluster",
    "nlp.summarize": "nlp.summarize",
    "score.posts": "score.posts",
    "score.authors": "score.authors",
    "score.fusion": "score.fusion",
    "graph.build_edges": "graph.build_edges",
    "graph.communities": "graph.communities",
    "graph.layout": "graph.layout",
    "compass.generate": "compass.generate",
    "domain.enrich": "domain.enrich",
    "media.deepfake": "media.deepfake",
    "alerts.evaluate": "alerts.evaluate",
    "reports.render": "reports.render",
    "maint.purge_uploads": "maint.purge_uploads",
}


def to_job_out(row) -> JobOut:
    return JobOut(
        id=str(row.id),
        kind=row.kind,
        status=row.status,
        project_id=str(row.project_id) if row.project_id else None,
        progress=row.progress or 0.0,
        params=row.params or {},
        result=row.result,
        error=row.error,
        celery_task_id=row.celery_task_id,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


async def enqueue(
    session: AsyncSession,
    *,
    kind: str,
    project_id: Any,
    params: dict,
    job_id: str | None = None,
    status_url: str | None = None,
) -> JobAccepted:
    """Create the job row and dispatch the task. Returns the 202 body."""
    from app.models.ops import Job

    if kind not in TASK_NAMES:
        raise BadRequest(f"Unknown job kind {kind!r}.", code="unknown_job_kind")

    resolved_project: uuid.UUID | None = None
    if project_id:
        from app.repositories.filters import resolve_project_id

        resolved_project = await resolve_project_id(session, str(project_id))

    row = Job(
        id=uuid.UUID(job_id) if job_id else uuid.uuid4(),
        kind=kind,
        project_id=resolved_project,
        status="pending",
        params=params,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    message = None
    try:
        from app.tasks.celery_app import celery

        async_result = celery.send_task(
            TASK_NAMES[kind],
            kwargs={**params, "job_id": str(row.id), "project_id": str(resolved_project or "")},
        )
        row.celery_task_id = async_result.id
        await session.commit()
    except Exception as exc:
        # The row survives and records why. A 202 pointing at a job that was
        # never dispatched is worse than an honest failed job the caller can
        # see through the same endpoint they were going to poll anyway.
        log.error("could not dispatch %s: %s", kind, exc)
        row.status = "failed"
        row.error = f"could not reach the task broker: {type(exc).__name__}: {exc}"[:2000]
        row.finished_at = utcnow()
        await session.commit()
        message = "The job was recorded but could not be dispatched; see `error` on the job."

    return JobAccepted(
        job_id=str(row.id),
        status=row.status,
        status_url=status_url or f"/api/v1/jobs/{row.id}",
        kind=kind,
        message=message,
    )


async def cancel(session: AsyncSession, job_id: str) -> JobCancelled:
    """Request cancellation. Cooperative, never a kill.

    Revoking with ``terminate=True`` would SIGTERM the worker mid-write. These
    tasks COPY into Postgres and upsert scores in batches; interrupting one
    between the write and the commit is how a scoring run ends up double-counted
    on the retry. The flag is set, the task notices at its next checkpoint.
    """
    from app.models.ops import Job

    row = await session.get(Job, job_id)
    if row is None:
        raise NotFound(f"No job with id {job_id}.", code="job_not_found")

    if row.status in ("succeeded", "failed", "cancelled"):
        return JobCancelled(
            id=str(row.id),
            status=row.status,
            detail=f"Job already finished with status {row.status!r}; nothing to cancel.",
        )

    row.cancel_requested = True
    if row.status == "pending":
        # Not started yet, so there is no checkpoint to reach. Revoke it from
        # the queue and close the row out here.
        try:
            from app.tasks.celery_app import celery

            if row.celery_task_id:
                celery.control.revoke(row.celery_task_id)
        except Exception as exc:  # pragma: no cover - broker down is not fatal here
            log.warning("could not revoke queued task %s: %s", row.celery_task_id, exc)
        row.status = "cancelled"
        row.finished_at = utcnow()
        detail = "Job was still queued and has been revoked."
    else:
        detail = (
            "Cancellation requested. The task stops at its next checkpoint rather than "
            "being killed mid-write, so work already committed stays committed and the "
            "job may report partial progress."
        )

    await session.commit()
    return JobCancelled(id=str(row.id), status=row.status, detail=detail)
