"""API key management. Requires the `admin` scope.

Deliberately minimal: mint, list, revoke. No OAuth, no user accounts, no
password reset flow -- three scopes on an opaque bearer token is the whole
security model for a research capstone with one analyst team, and anything more
is scope creep with a security surface attached.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import RequireAdmin, rate_limit
from app.errors import NotFound
from app.schemas.common import PageResponse
from app.schemas.keys import ApiKeyCreate, ApiKeyCreated, ApiKeyOut

log = logging.getLogger(__name__)

router = APIRouter(prefix="/keys", tags=["auth"], dependencies=[Depends(rate_limit)])


def _to_out(row) -> ApiKeyOut:
    return ApiKeyOut(
        id=str(row.id),
        name=row.name,
        prefix=row.prefix,
        scopes=list(row.scopes or ()),
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
        is_active=row.is_active,
    )


@router.post(
    "",
    response_model=ApiKeyCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Mint an API key",
    description=(
        "Returns the plaintext key exactly once. Only a peppered hash is stored, "
        "so a lost key is reissued, never recovered."
    ),
)
async def create_key(
    body: ApiKeyCreate,
    principal: RequireAdmin,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ApiKeyCreated:
    from app.models.ops import ApiKey
    from app.security import mint

    minted = mint()
    row = ApiKey(
        name=body.name,
        key_hash=minted.key_hash,
        prefix=minted.prefix,
        scopes=body.scopes,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    # The prefix, never the key.
    log.info(
        "api key minted id=%s prefix=%s scopes=%s by=%s",
        row.id,
        row.prefix,
        body.scopes,
        principal.key_id,
    )
    return ApiKeyCreated(**_to_out(row).model_dump(), key=minted.plaintext)


@router.get("", response_model=PageResponse[ApiKeyOut], summary="List API keys")
async def list_keys(
    principal: RequireAdmin,
    session: Annotated[AsyncSession, Depends(get_session)],
    include_revoked: bool = False,
) -> PageResponse[ApiKeyOut]:
    from app.models.ops import ApiKey

    stmt = select(ApiKey).order_by(ApiKey.created_at.desc())
    if not include_revoked:
        stmt = stmt.where(ApiKey.revoked_at.is_(None))
    rows = (await session.execute(stmt)).scalars().all()
    return PageResponse[ApiKeyOut](
        items=[_to_out(r) for r in rows],
        total=len(rows),
        filters_applied={"include_revoked": include_revoked},
    )


@router.delete(
    "/{key_id}",
    response_model=ApiKeyOut,
    summary="Revoke an API key",
    description=(
        "Revocation is a tombstone, not a delete: the audit trail of what a key did "
        "has to outlive the key."
    ),
)
async def revoke_key(
    key_id: str,
    principal: RequireAdmin,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ApiKeyOut:
    from app.db import utcnow
    from app.models.ops import ApiKey

    row = await session.get(ApiKey, key_id)
    if row is None:
        raise NotFound(f"No API key with id {key_id}.", code="api_key_not_found")
    if row.revoked_at is None:
        row.revoked_at = utcnow()
        await session.commit()
        await session.refresh(row)
        log.info("api key revoked id=%s prefix=%s by=%s", row.id, row.prefix, principal.key_id)
    return _to_out(row)
