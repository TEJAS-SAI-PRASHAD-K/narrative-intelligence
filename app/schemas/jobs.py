"""The one job shape every async operation reports through.

Every expensive route returns 202 with a ``job_id``, and the frontend polls this
single shape regardless of what it started. That uniformity is the point: one
progress component in the UI, one polling hook, one error surface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import Camel

JobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]

#: Every task kind. Kept as a literal rather than a free string so the frontend
#: can exhaustively switch on it and so a typo in a task name fails a test.
JobKind = Literal[
    "ingest.fetch",
    "ingest.load",
    "nlp.embed",
    "nlp.cluster",
    "nlp.summarize",
    "score.posts",
    "score.authors",
    "score.fusion",
    "graph.build_edges",
    "graph.communities",
    "graph.layout",
    "compass.generate",
    "domain.enrich",
    "media.deepfake",
    "alerts.evaluate",
    "reports.render",
    "maint.purge_uploads",
]


class JobOut(Camel):
    id: str
    kind: JobKind
    status: JobStatus
    project_id: str | None = None
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction complete. Monotonic within a run, but a resumed task starts from "
            "where it left off rather than at zero -- tasks are resumable, and resetting "
            "progress on a retry would make a resume look like a restart."
        ),
    )
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    celery_task_id: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("succeeded", "failed", "cancelled")


class JobCancelled(Camel):
    id: str
    status: JobStatus
    detail: str = Field(
        description=(
            "Cancellation is cooperative: a running task stops at its next checkpoint "
            "rather than being killed mid-write, so a cancelled job may report "
            "partial progress that is already durably committed."
        )
    )
