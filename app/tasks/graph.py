"""Graph tasks.

Edge building, community detection and server-side layout.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="graph.build_edges", base=TrackedTask, bind=True)
def build_edges(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Reply/repost/mention plus co-post similarity edges."""
    raise NotImplementedError(
        "graph.build_edges lands at build step 10. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )


@celery.task(name="graph.communities", base=TrackedTask, bind=True)
def communities(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Louvain over the co-posting graph."""
    raise NotImplementedError(
        "graph.communities lands at build step 10. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )


@celery.task(name="graph.layout", base=TrackedTask, bind=True)
def layout(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Precompute node positions above the layout threshold."""
    raise NotImplementedError(
        "graph.layout lands at build step 10. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )
