"""Alert rules and fired alerts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from app.schemas.common import Camel

Channel = Literal["in_app", "email", "webhook"]
Op = Literal[">", ">=", "<", "<=", "==", "!="]
Scope = Literal["narrative", "author", "domain", "project"]


class AlertCondition(Camel):
    """A single threshold. Deliberately not a query language.

    An expression DSL here would be a small interpreter with an injection
    surface, evaluated against the corpus on a beat schedule. One metric, one
    operator, one value covers every rule the UI spec asks for.
    """

    metric: str = Field(description="e.g. 'fusion_score', 'bot_like_ratio', 'post_velocity'.")
    op: Op
    value: float
    scope: Scope = "narrative"
    #: Optional extra narrowing, e.g. {"platform": ["mastodon"]}.
    filters: dict[str, Any] = Field(default_factory=dict)


class AlertRuleCreate(Camel):
    project_id: str
    name: str = Field(min_length=1, max_length=256)
    condition: AlertCondition
    channels: list[Channel] = Field(default_factory=lambda: ["in_app"])
    enabled: bool = True
    cooldown_minutes: int = Field(
        default=60,
        ge=0,
        le=10080,
        description=(
            "Minimum gap between two alerts from this rule for the same subject. "
            "Without it a narrative hovering on a threshold fires every evaluation "
            "cycle and the analyst stops reading alerts."
        ),
    )

    @field_validator("channels")
    @classmethod
    def _at_least_one(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("a rule with no channels can never notify anybody")
        return v


class AlertRuleUpdate(Camel):
    name: str | None = None
    condition: AlertCondition | None = None
    channels: list[Channel] | None = None
    enabled: bool | None = None
    cooldown_minutes: int | None = Field(default=None, ge=0, le=10080)


class AlertRuleOut(Camel):
    id: str
    project_id: str
    name: str
    condition: AlertCondition
    channels: list[Channel]
    enabled: bool
    cooldown_minutes: int
    created_at: datetime
    last_evaluated_at: datetime | None = None
    last_triggered_at: datetime | None = None
    trigger_count: int = 0


class AlertOut(Camel):
    id: str
    rule_id: str
    rule_name: str
    project_id: str
    narrative_id: str | None = None
    narrative_title: str | None = None
    subject_type: Scope
    subject_id: str | None = None
    triggered_at: datetime
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The metric value and the condition that fired, so the alert is self-explaining."
        ),
    )
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None


class AlertAcknowledged(Camel):
    id: str
    acknowledged_at: datetime
    acknowledged_by: str
