"""Media tasks.

Deepfake inference over uploaded imagery.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="media.deepfake", base=TrackedTask, bind=True)
def deepfake(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Xception over sampled frames."""
    raise NotImplementedError(
        "media.deepfake lands at build step 12. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )
