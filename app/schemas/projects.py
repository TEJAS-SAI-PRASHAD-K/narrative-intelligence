"""Projects and per-source pipeline health."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import Camel

#: Phase 1's five source values. Six adapters converge on them (ConvoKit and
#: Kaggle both write `reddit`), which is why this list is shorter than the
#: adapter count in the ingest package.
SourceName = Literal["reddit", "mastodon", "news", "gdelt", "youtube"]


class ProjectCreate(Camel):
    slug: str = Field(
        min_length=2,
        max_length=96,
        pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$",
        description="URL-safe handle. Unique; used by the CLI so nobody pastes uuids around.",
    )
    name: str = Field(min_length=1, max_length=256)
    description: str | None = None
    date_start: datetime | None = Field(
        default=None, description="Start of the corpus window under study, not of the project."
    )
    date_end: datetime | None = None
    seed_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Mirrors configs/topics.yaml. Stored rather than referenced so a project "
            "stays reproducible after somebody edits the YAML."
        ),
    )


class ProjectUpdate(Camel):
    name: str | None = None
    description: str | None = None
    date_start: datetime | None = None
    date_end: datetime | None = None
    seed_config: dict[str, Any] | None = None


class ProjectStats(Camel):
    """Cheap counts for the project card. Never a full scan of the corpus."""

    post_count: int = 0
    author_count: int = 0
    narrative_count: int = 0
    domain_count: int = 0
    first_post_at: datetime | None = None
    last_post_at: datetime | None = None


class ProjectOut(Camel):
    id: str
    slug: str
    name: str
    description: str | None = None
    date_start: datetime | None = None
    date_end: datetime | None = None
    seed_config: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
    stats: ProjectStats = Field(default_factory=ProjectStats)


class SourceHealth(Camel):
    """One row of the setup page's pipeline health table.

    ``credentials`` and ``status`` are separate on purpose. A source with no
    credentials is not broken -- Phase 1 skips it deliberately and says so -- and
    conflating the two would show a red light for a configuration choice.
    """

    source: SourceName
    configured: bool = Field(description="Whether credentials for this source are present.")
    status: Literal["ok", "never_run", "stale", "quota_exhausted", "error", "skipped"]
    detail: str = Field(description="Human-readable explanation of the status.")
    last_sync_at: datetime | None = None
    record_count: int = 0
    #: Phase 1's per-source checkpoint cursor, so the UI can show where a resume
    #: would start from rather than implying every run is a full refetch.
    checkpoint: dict[str, Any] | None = None
    quota_used: int | None = None
    quota_limit: int | None = None
    quota_resets_at: datetime | None = None


class SourceTestResult(Camel):
    """The result of a live credential check. This one route does make a
    network call, which is why it is a POST and not part of the health GET."""

    source: SourceName
    reachable: bool
    authenticated: bool
    detail: str
    latency_ms: float | None = None
    checked_at: datetime
