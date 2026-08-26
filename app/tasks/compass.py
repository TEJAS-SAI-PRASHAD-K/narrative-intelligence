"""Compass tasks.

RAG fact-checking with mandatory citation validation.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="compass.generate", base=TrackedTask, bind=True)
def generate(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Retrieve, generate, validate citations, persist or refuse."""
    raise NotImplementedError(
        "compass.generate lands at build step 11. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )
