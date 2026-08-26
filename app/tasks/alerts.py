"""Alert tasks.

Rule evaluation with cooldown.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="alerts.evaluate", base=TrackedTask, bind=True)
def evaluate(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Evaluate every enabled rule and fire past the cooldown."""
    raise NotImplementedError(
        "alerts.evaluate lands at build step 13. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )
