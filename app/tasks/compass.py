"""Compass Context generation.

RAG with mandatory citation. The task is thin: it resolves the narrative, calls
the pipeline, and lets `app/compass/generate.py` own the retrieve -> generate ->
validate loop and the decision to persist or refuse.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.tasks.base import TrackedTask, report_progress
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="compass.generate", base=TrackedTask, bind=True)
def generate(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Generate Compass Context for one narrative, or for a whole project.

    Always writes exactly one row per narrative, even when it refuses: a
    refusal that leaves no trace is indistinguishable from never having tried,
    and an analyst needs to know the difference.
    """
    from sqlalchemy import text

    from app.compass.generate import generate_for_narrative
    from app.db import advisory_lock, sync_session
    from app.etl.service import resolve_project

    narrative_id = params.get("narrative_id")

    with sync_session() as session:
        resolved = None
        if project_id:
            resolved = resolve_project(session, project_id)
        elif narrative_id:
            resolved = session.execute(
                text("SELECT project_id FROM narratives WHERE id = :n"),
                {"n": uuid.UUID(narrative_id)},
            ).scalar()

        if resolved is None:
            return {"skipped": True, "reason": "could not resolve the project"}

        targets = (
            [uuid.UUID(narrative_id)]
            if narrative_id
            else [
                row[0]
                for row in session.execute(
                    text(
                        """
                        SELECT n.id FROM narratives n
                        WHERE n.project_id = :p
                          AND NOT EXISTS (
                              SELECT 1 FROM compass_contexts c
                              WHERE c.narrative_id = n.id AND c.superseded_by IS NULL
                          )
                        """
                    ),
                    {"p": resolved},
                ).all()
            ]
        )

        if not targets:
            return {"generated": 0, "note": "every narrative already has a live context"}

        lock_key = f"compass.generate:{narrative_id or project_id}"
        with advisory_lock(session, lock_key) as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another compass run holds the lock"}

            outcomes: list[dict[str, Any]] = []
            for index, target in enumerate(targets):
                outcomes.append(generate_for_narrative(session, resolved, target))
                report_progress(job_id, 0.1 + 0.85 * (index + 1) / len(targets))

    by_status: dict[str, int] = {}
    for outcome in outcomes:
        status = outcome["verification_status"]
        by_status[status] = by_status.get(status, 0) + 1

    result = {
        "generated": len(outcomes),
        "by_status": by_status,
        "cited": sum(1 for o in outcomes if o["citations"]),
    }
    report_progress(job_id, 1.0, result=result)
    return result
