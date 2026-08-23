"""Ingestion runs.

These routes are thin wrappers over Phase 1's existing adapters. Nothing in
Phase 4 speaks HTTP to a platform; if a request body here ever grows a
`subreddit` field, something has gone wrong architecturally.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from app.schemas.common import Camel
from app.schemas.projects import SourceName

IngestMode = Literal["fetch", "load", "both"]


class IngestRequest(Camel):
    project_id: str
    sources: list[SourceName] = Field(
        default_factory=list, description="Empty means every source with credentials configured."
    )
    date_start: date | None = None
    date_end: date | None = None
    mode: IngestMode = Field(
        default="both",
        description=(
            "'fetch' runs the Phase 1 adapters to Parquet. 'load' bulk-copies existing "
            "Parquet into Postgres. 'both' chains them."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "Ignore Phase 1's per-source checkpoints and refetch the window. Expensive "
            "and quota-consuming; the default resumes."
        ),
    )

    @model_validator(mode="after")
    def _ordered_dates(self) -> IngestRequest:
        if self.date_start and self.date_end and self.date_start > self.date_end:
            raise ValueError("date_start is after date_end")
        return self


class IngestRunOut(Camel):
    id: str
    project_id: str
    source: str
    status: Literal["pending", "running", "succeeded", "failed", "partial", "skipped"]
    mode: IngestMode | None = None
    records_in: int = Field(default=0, description="Rows read from Parquet.")
    records_loaded: int = Field(default=0, description="Rows that landed in Postgres.")
    records_rejected: int = 0
    #: Reason code -> count. Every rejected row is accounted for here; silent
    #: data loss between Parquet and Postgres would poison every downstream
    #: metric and surface three weeks later as an unexplainable number.
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    rejects_path: str | None = Field(
        default=None, description="Parquet file holding the rejected rows for inspection."
    )
    manifest_sha: str | None = None
    detail: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class VerifyReport(Camel):
    """Reconciliation between the manifest, the Parquet corpus and Postgres.

    Three counts that must agree. When they do not, the difference is itemised
    by reason code rather than reported as a single mismatch number.
    """

    project_id: str
    manifest_rows: int | None = None
    parquet_rows: int
    postgres_rows: int
    rejected_rows: int
    reconciled: bool
    discrepancies: list[dict[str, Any]] = Field(default_factory=list)
    by_source: dict[str, dict[str, int]] = Field(default_factory=dict)
    checked_at: datetime
