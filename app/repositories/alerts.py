"""Alert and alert-rule queries."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import utcnow
from app.deps import Page
from app.errors import NotFound
from app.repositories.filters import decode_cursor, encode_cursor, resolve_project_id
from app.schemas.alerts import (
    AlertAcknowledged,
    AlertCondition,
    AlertOut,
    AlertRuleCreate,
    AlertRuleOut,
    AlertRuleUpdate,
)
from app.schemas.common import PageResponse

log = logging.getLogger(__name__)


def to_rule_out(row) -> AlertRuleOut:
    return AlertRuleOut(
        id=str(row.id),
        project_id=str(row.project_id),
        name=row.name,
        condition=AlertCondition(**row.condition),
        channels=list(row.channels or ()),
        enabled=row.enabled,
        cooldown_minutes=row.cooldown_minutes,
        created_at=row.created_at,
        last_evaluated_at=row.last_evaluated_at,
        last_triggered_at=row.last_triggered_at,
        trigger_count=row.trigger_count or 0,
    )


async def list_alerts(
    session: AsyncSession, project_id: str, page: Page, *, acknowledged: bool | None = None
) -> PageResponse[AlertOut]:
    from app.models.narratives import Narrative
    from app.models.ops import Alert, AlertRule

    resolved = await resolve_project_id(session, project_id)
    stmt = (
        select(Alert, AlertRule.name, Narrative.title)
        .join(AlertRule, AlertRule.id == Alert.rule_id)
        .outerjoin(Narrative, Narrative.id == Alert.narrative_id)
        .where(Alert.project_id == resolved)
        .order_by(Alert.triggered_at.desc())
    )
    if acknowledged is not None:
        stmt = (
            stmt.where(Alert.acknowledged_at.isnot(None))
            if acknowledged
            else stmt.where(Alert.acknowledged_at.is_(None))
        )

    offset = decode_cursor(page.cursor).get("o", 0)
    rows = (await session.execute(stmt.offset(offset).limit(page.limit))).all()

    return PageResponse[AlertOut](
        items=[
            AlertOut(
                id=str(alert.id),
                rule_id=str(alert.rule_id),
                rule_name=rule_name,
                project_id=str(alert.project_id),
                narrative_id=str(alert.narrative_id) if alert.narrative_id else None,
                narrative_title=title,
                subject_type=alert.subject_type,
                subject_id=alert.subject_id,
                triggered_at=alert.triggered_at,
                payload=alert.payload or {},
                acknowledged_at=alert.acknowledged_at,
                acknowledged_by=alert.acknowledged_by,
            )
            for alert, rule_name, title in rows
        ],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=None,
        filters_applied={"project_id": project_id, "acknowledged": acknowledged},
    )


async def acknowledge(session: AsyncSession, alert_id: str, *, actor: str) -> AlertAcknowledged:
    from app.models.ops import Alert

    row = await session.get(Alert, alert_id)
    if row is None:
        raise NotFound(f"No alert with id {alert_id}.", code="alert_not_found")
    # Idempotent: acknowledging twice keeps the first acknowledgement rather
    # than rewriting who saw it first, which is the fact worth keeping.
    if row.acknowledged_at is None:
        row.acknowledged_at = utcnow()
        row.acknowledged_by = actor
        await session.commit()
    return AlertAcknowledged(
        id=str(row.id), acknowledged_at=row.acknowledged_at, acknowledged_by=row.acknowledged_by
    )


async def list_rules(
    session: AsyncSession, project_id: str, page: Page
) -> PageResponse[AlertRuleOut]:
    from app.models.ops import AlertRule

    resolved = await resolve_project_id(session, project_id)
    rows = (
        (
            await session.execute(
                select(AlertRule)
                .where(AlertRule.project_id == resolved)
                .order_by(AlertRule.created_at.desc())
                .limit(page.limit)
            )
        )
        .scalars()
        .all()
    )
    return PageResponse[AlertRuleOut](
        items=[to_rule_out(row) for row in rows],
        total=len(rows),
        filters_applied={"project_id": project_id},
    )


async def create_rule(session: AsyncSession, body: AlertRuleCreate) -> AlertRuleOut:
    from app.models.ops import AlertRule

    project_id = await resolve_project_id(session, body.project_id)
    row = AlertRule(
        project_id=project_id,
        name=body.name,
        condition=body.condition.model_dump(),
        channels=body.channels,
        enabled=body.enabled,
        cooldown_minutes=body.cooldown_minutes,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return to_rule_out(row)


async def update_rule(session: AsyncSession, rule_id: str, body: AlertRuleUpdate) -> AlertRuleOut:
    from app.models.ops import AlertRule

    row = await session.get(AlertRule, rule_id)
    if row is None:
        raise NotFound(f"No alert rule with id {rule_id}.", code="alert_rule_not_found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(row, field, value.model_dump() if field == "condition" else value)
    await session.commit()
    await session.refresh(row)
    return to_rule_out(row)


async def delete_rule(session: AsyncSession, rule_id: str) -> None:
    """Delete the rule and, by cascade, its fired alerts.

    A hard delete rather than a tombstone: unlike an API key, a rule's alerts
    carry their own payload and triggering condition, so nothing about what
    happened is lost by the rule going away -- and a disabled rule already
    covers the "stop it without losing it" case.
    """
    from app.models.ops import AlertRule

    row = await session.get(AlertRule, rule_id)
    if row is None:
        raise NotFound(f"No alert rule with id {rule_id}.", code="alert_rule_not_found")
    await session.delete(row)
    await session.commit()
