"""Report rendering."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import Camel

Template = Literal["exec_summary", "full_detail", "raw_data"]
Format = Literal["pdf", "pptx", "csv"]


class ReportCreate(Camel):
    project_id: str
    template: Template
    format: Format
    narrative_ids: list[str] = Field(
        default_factory=list, description="Empty means the whole project under the filter spec."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Template-specific options, e.g. {'include_compass': true}.",
    )


class ReportOut(Camel):
    id: str
    project_id: str
    template: Template
    format: Format
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    params: dict[str, Any] = Field(default_factory=dict)
    size_bytes: int | None = None
    download_url: str | None = Field(default=None, description="Null until the render succeeds.")
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None
