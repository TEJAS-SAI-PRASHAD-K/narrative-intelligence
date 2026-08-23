"""Conversation drivers -- the three tabs of that page.

Authors, hashtags and URLs share one shape so the frontend renders one table
component three times. They differ only in the ``entity`` field's meaning, which
``entity_type`` names.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import Camel

EntityType = Literal["author", "hashtag", "url", "domain"]


class DriverItem(Camel):
    entity: str
    entity_type: EntityType
    label: str | None = Field(
        default=None, description="Display name, e.g. a handle for an author."
    )
    post_count: int
    author_count: int | None = Field(
        default=None, description="Distinct authors using it. Null for the author tab itself."
    )
    engagement_total: int | None = None
    #: A driver that is amplified almost entirely by bot-like accounts is a very
    #: different object from one with organic reach, and the UI colours the row
    #: on this. It is nullable because the bot scorer skips outlet 'authors'.
    bot_like_share: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_toxicity: float | None = None
    dominant_sentiment: str | None = None
    narrative_ids: list[str] = Field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    delta_pct: float | None = None


class DriversResponse(Camel):
    entity_type: EntityType
    items: list[DriverItem]
    definition: str
