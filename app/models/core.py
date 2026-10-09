"""Projects -- the top-level case or workspace.

Everything in this schema is scoped to a project. That is not multi-tenancy; it
is the UI's project selector, and it maps onto what Phase 1 called a pipeline
run: one seed config, one date range, one corpus.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = uuid_pk()
    #: The URL-safe handle the UI and the CLI both use. Unique so `--project
    #: election-2026` is unambiguous without anybody copying a uuid around.
    slug: Mapped[str] = mapped_column(String(96), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: The corpus window under study. Distinct from created_at: a project made
    #: today can study last March.
    date_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    date_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Mirrors configs/topics.yaml -- the seed keywords and boolean queries that
    #: define the case. Stored rather than referenced so a project stays
    #: reproducible after somebody edits the YAML.
    seed_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Project {self.slug}>"
