"""Job lifecycle: create a row, dispatch the task, report status.

Placeholder until build step 6, when the Celery app lands. The 202 contract is
already published; what is missing is the broker behind it.
"""

from __future__ import annotations

from typing import Any

from app.errors import NotImplementedYet


async def enqueue(session, *, kind: str, project_id: Any, params: dict, **extra: Any):
    raise NotImplementedYet(
        f"Cannot enqueue {kind}: the Celery app lands at build step 6.",
        detail={"kind": kind, "lands_at": "build step 6"},
    )


async def cancel(session, job_id: str):
    raise NotImplementedYet(
        "Job cancellation lands at build step 6.",
        detail={"job_id": job_id, "lands_at": "build step 6"},
    )
