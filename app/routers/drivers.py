"""Conversation drivers -- the three tabs of that page.

One shape, three entity types, one filter spec, one pagination. The UI renders
the same table component three times, which is only possible because the backend
refuses to give each tab a bespoke response.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import Filters, Pagination, RequireRead, rate_limit
from app.mock import responses as mock
from app.schemas.drivers import DriversResponse

router = APIRouter(prefix="/drivers", tags=["drivers"], dependencies=[Depends(rate_limit)])


async def _drivers(session, filters, page, entity_type: str) -> DriversResponse:
    if mock.demo_mode():
        response, paged = mock.drivers(filters, page, entity_type)
        # The DriversResponse carries the definition; the page envelope carries
        # the cursor. Both matter, so the cursor fields are copied onto the
        # response rather than the caller being handed two objects.
        response_dict = response.model_dump()
        response_dict["items"] = [item.model_dump() for item in paged.items]
        return DriversResponse.model_validate(response_dict)
    from app.repositories.drivers import drivers as query_drivers

    return await query_drivers(session, filters, page, entity_type)


@router.get("/authors", response_model=DriversResponse, summary="Top authors driving conversation")
async def driver_authors(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DriversResponse:
    return await _drivers(session, filters, page, "author")


@router.get("/hashtags", response_model=DriversResponse, summary="Top hashtags")
async def driver_hashtags(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DriversResponse:
    return await _drivers(session, filters, page, "hashtag")


@router.get("/urls", response_model=DriversResponse, summary="Top URLs")
async def driver_urls(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DriversResponse:
    return await _drivers(session, filters, page, "url")
