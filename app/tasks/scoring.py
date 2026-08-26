"""Scoring tasks.

Post scores, author scores and the narrative fusion scorecards.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="score.posts", base=TrackedTask, bind=True)
def score_posts(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Toxicity, anomaly, misinfo, stance, sentiment, emotion."""
    raise NotImplementedError(
        "score.posts lands at build step 9. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )


@celery.task(name="score.authors", base=TrackedTask, bind=True)
def score_authors(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Bot classifier plus feature attribution."""
    raise NotImplementedError(
        "score.authors lands at build step 9. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )


@celery.task(name="score.fusion", base=TrackedTask, bind=True)
def score_fusion(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Narrative scorecards from configs/fusion.yaml."""
    raise NotImplementedError(
        "score.fusion lands at build step 9. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )
