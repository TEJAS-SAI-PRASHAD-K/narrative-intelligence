"""Column conventions shared by every model.

These exist so that "when was this row written" and "what produced this number"
are answerable for every table without remembering which ones happen to have the
columns. The explainability requirement is not a feature of one endpoint; it is
a property of the schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, declarative_mixin, mapped_column

from app.db import Base

__all__ = ["Base", "TimestampMixin", "ScoredMixin", "uuid_pk", "utc_column", "jsonb_column"]


def uuid_pk() -> Mapped[uuid.UUID]:
    """A server-generated uuid primary key.

    ``gen_random_uuid()`` rather than a Python-side default so that a bulk
    ``INSERT ... SELECT`` (which the ETL uses heavily) does not have to round-trip
    to Python to mint ids.
    """
    return mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )


def utc_column(**kwargs: Any) -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), **kwargs)


def jsonb_column(**kwargs: Any) -> Mapped[dict]:
    return mapped_column(JSONB, **kwargs)


@declarative_mixin
class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


@declarative_mixin
class ScoredMixin:
    """Every table that holds a model-produced number carries these two.

    ``scoring_version`` identifies the config that produced the number;
    ``components`` holds the inputs. Together they are what makes the UI's "why
    is this flagged" drilldown possible, and what makes a rescore under new
    weights detectable rather than silent. A score without them is a black box,
    which is the thing this product exists to argue against.
    """

    scoring_version: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    components: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    computed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
