"""Domain tasks.

Best-effort WHOIS/TLS enrichment, rate-limited.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="domain.enrich", base=TrackedTask, bind=True)
def enrich(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Enrich one domain, or a rate-limited batch."""
    raise NotImplementedError(
        "domain.enrich lands at build step 13. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )
