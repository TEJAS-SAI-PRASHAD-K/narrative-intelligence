"""Alerts and alert rules."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session, utcnow
from app.deps import Pagination, RequireRead, RequireWrite, rate_limit
from app.errors import NotFound
from app.mock import responses as mock
from app.schemas.alerts import (
    AlertAcknowledged,
    AlertOut,
    AlertRuleCreate,
    AlertRuleOut,
    AlertRuleUpdate,
)
from app.schemas.common import DeletedResponse, PageResponse

router = APIRouter(tags=["alerts"], dependencies=[Depends(rate_limit)])


@router.get("/alerts", response_model=PageResponse[AlertOut], summary="Fired alerts")
async def list_alerts(
    principal: RequireRead,
    project_id: str,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
    acknowledged: bool | None = None,
) -> PageResponse[AlertOut]:
    if mock.demo_mode():
        items = mock.alerts_list()
        if acknowledged is not None:
            items = [a for a in items if (a.acknowledged_at is not None) is acknowledged]
        return mock.paginate(items, page, {"project_id": project_id, "acknowledged": acknowledged})
    from app.repositories.alerts import list_alerts as query_alerts

    return await query_alerts(session, project_id, page, acknowledged=acknowledged)


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertAcknowledged,
    summary="Acknowledge an alert",
)
async def acknowledge_alert(
    alert_id: str,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AlertAcknowledged:
    if mock.demo_mode():
        return AlertAcknowledged(
            id=alert_id, acknowledged_at=utcnow(), acknowledged_by=principal.name
        )
    from app.repositories.alerts import acknowledge

    return await acknowledge(session, alert_id, actor=principal.name)


@router.get("/alert-rules", response_model=PageResponse[AlertRuleOut], summary="List alert rules")
async def list_alert_rules(
    principal: RequireRead,
    project_id: str,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[AlertRuleOut]:
    if mock.demo_mode():
        return mock.paginate(mock.alert_rules(), page, {"project_id": project_id})
    from app.repositories.alerts import list_rules

    return await list_rules(session, project_id, page)


@router.post(
    "/alert-rules",
    response_model=AlertRuleOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create an alert rule",
)
async def create_alert_rule(
    body: AlertRuleCreate,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AlertRuleOut:
    if mock.demo_mode():
        from datetime import timedelta

        from app.mock.corpus import DEMO_NOW, _stable_uuid

        return AlertRuleOut(
            id=_stable_uuid("alert_rule", body.name),
            project_id=body.project_id,
            name=body.name,
            condition=body.condition,
            channels=body.channels,
            enabled=body.enabled,
            cooldown_minutes=body.cooldown_minutes,
            created_at=DEMO_NOW - timedelta(seconds=1),
            trigger_count=0,
        )
    from app.repositories.alerts import create_rule

    return await create_rule(session, body)


@router.patch("/alert-rules/{rule_id}", response_model=AlertRuleOut, summary="Update an alert rule")
async def update_alert_rule(
    rule_id: str,
    body: AlertRuleUpdate,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AlertRuleOut:
    if mock.demo_mode():
        for rule in mock.alert_rules():
            if rule.id == rule_id:
                return rule.model_copy(update=body.model_dump(exclude_unset=True))
        raise NotFound(f"No alert rule with id {rule_id}.", code="alert_rule_not_found")
    from app.repositories.alerts import update_rule

    return await update_rule(session, rule_id, body)


@router.delete(
    "/alert-rules/{rule_id}", response_model=DeletedResponse, summary="Delete an alert rule"
)
async def delete_alert_rule(
    rule_id: str,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DeletedResponse:
    if mock.demo_mode():
        return DeletedResponse(id=rule_id)
    from app.repositories.alerts import delete_rule

    await delete_rule(session, rule_id)
    return DeletedResponse(id=rule_id)
