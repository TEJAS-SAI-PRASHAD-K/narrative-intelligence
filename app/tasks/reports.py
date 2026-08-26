"""Report tasks.

PDF, PPTX and CSV rendering.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="reports.render", base=TrackedTask, bind=True)
def render(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Render a report to disk and record its path."""
    raise NotImplementedError(
        "reports.render lands at build step 13. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )
