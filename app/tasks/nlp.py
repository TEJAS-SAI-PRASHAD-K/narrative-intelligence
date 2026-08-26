"""NLP tasks.

Embedding, clustering and summarization. Every one is resumable: a
killed worker resumes from the rows that already carry the current model
version rather than recomputing the corpus.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tasks.base import TrackedTask
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="nlp.embed", base=TrackedTask, bind=True)
def embed(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Batch-embed unembedded posts."""
    raise NotImplementedError(
        "nlp.embed lands at build step 7. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )


@celery.task(name="nlp.cluster", base=TrackedTask, bind=True)
def cluster(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """HDBSCAN over embeddings. Preserves user-edited titles."""
    raise NotImplementedError(
        "nlp.cluster lands at build step 8. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )


@celery.task(name="nlp.summarize", base=TrackedTask, bind=True)
def summarize(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """One LLM call per cluster, never per post."""
    raise NotImplementedError(
        "nlp.summarize lands at build step 8. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )
