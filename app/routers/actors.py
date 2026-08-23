"""Actors -- the second pillar.

Cohorts and author groups are separate endpoints because they are separate
concepts: a cohort is a model's multi-label guess, an author group is an
analyst's curated watchlist. An investigation rests on being able to tell those
apart, so the API never merges them into one "tags" field.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import Filters, Pagination, RequireRead, RequireWrite, rate_limit
from app.errors import NotFound
from app.mock import responses as mock
from app.schemas.actors import (
    AuthorGroupCreate,
    AuthorGroupMembers,
    AuthorGroupMembersResult,
    AuthorGroupOut,
    AuthorOut,
    AuthorScoreExplanation,
    AuthorTimeline,
    CohortOut,
)
from app.schemas.common import PageResponse
from app.schemas.posts import PostOut

router = APIRouter(tags=["actors"], dependencies=[Depends(rate_limit)])


def _demo_author(author_id: str):
    author = mock.corpus().by_author.get(author_id)
    if author is None:
        raise NotFound(f"No author with id {author_id}.", code="author_not_found")
    return author


@router.get("/authors", response_model=PageResponse[AuthorOut], summary="List authors")
async def list_authors(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
    sort: Annotated[str, Query(pattern="^(risk|bot_score|posts|followers)$")] = "risk",
) -> PageResponse[AuthorOut]:
    if mock.demo_mode():
        return mock.authors_page(filters, page, sort)
    from app.repositories.actors import list_authors as query_authors

    return await query_authors(session, filters, page, sort=sort)


@router.get(
    "/authors/{author_id:path}/score",
    response_model=AuthorScoreExplanation,
    summary="Why this author scores what it scores",
)
async def author_score(
    author_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AuthorScoreExplanation:
    """Feature-level attribution for the bot score.

    Note the ``:path`` converter on every author route: author ids are
    namespaced as ``source:native_id`` and a raw colon is legal in a path
    segment, but handles from some sources contain slashes. Without ``:path``
    those 404 in a way that looks like a missing row rather than a routing bug.
    """
    if mock.demo_mode():
        return mock.author_score(_demo_author(author_id))
    from app.repositories.actors import author_score as query_score

    return await query_score(session, author_id)


@router.get(
    "/authors/{author_id:path}/timeline",
    response_model=AuthorTimeline,
    summary="Author activity over time",
)
async def author_timeline(
    author_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
    interval: Annotated[str, Query(pattern="^(1h|6h|1d|7d)$")] = "1d",
    tz: str = "UTC",
) -> AuthorTimeline:
    if mock.demo_mode():
        return mock.author_timeline(_demo_author(author_id), interval, tz)
    from app.repositories.actors import author_timeline as query_timeline

    return await query_timeline(session, author_id, interval, tz)


@router.get(
    "/authors/{author_id:path}/posts",
    response_model=PageResponse[PostOut],
    summary="An author's posts",
)
async def author_posts(
    author_id: str,
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[PostOut]:
    if mock.demo_mode():
        _demo_author(author_id)
        posts = [p for p in mock.corpus().posts if p.author_id == author_id]
        posts = mock.apply_post_filters(posts, filters)
        return mock.paginate([mock.to_post(p) for p in posts], page, filters.as_response_dict())
    from app.repositories.actors import author_posts as query_posts

    return await query_posts(session, author_id, filters, page)


@router.get("/authors/{author_id:path}", response_model=AuthorOut, summary="Author detail")
async def get_author(
    author_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AuthorOut:
    if mock.demo_mode():
        return mock.to_author(_demo_author(author_id))
    from app.repositories.actors import get_author as query_author

    return await query_author(session, author_id)


# --- cohorts ---------------------------------------------------------------
@router.get("/cohorts", response_model=PageResponse[CohortOut], summary="List cohorts")
async def list_cohorts(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[CohortOut]:
    if mock.demo_mode():
        items = [mock.to_cohort(c) for c in mock.corpus().cohorts]
        items.sort(key=lambda c: -c.author_count)
        return mock.paginate(items, page, filters.as_response_dict())
    from app.repositories.actors import list_cohorts as query_cohorts

    return await query_cohorts(session, filters, page)


@router.get(
    "/cohorts/{cohort_id}/authors",
    response_model=PageResponse[AuthorOut],
    summary="Authors in a cohort",
)
async def cohort_authors(
    cohort_id: str,
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[AuthorOut]:
    if mock.demo_mode():
        if cohort_id not in mock.corpus().by_cohort:
            raise NotFound(f"No cohort with id {cohort_id}.", code="cohort_not_found")
        authors = [a for a in mock.corpus().authors if cohort_id in a.cohort_ids]
        return mock.paginate([mock.to_author(a) for a in authors], page, filters.as_response_dict())
    from app.repositories.actors import cohort_authors as query_cohort_authors

    return await query_cohort_authors(session, cohort_id, filters, page)


# --- author groups (curated watchlists) ------------------------------------
@router.get(
    "/author-groups", response_model=PageResponse[AuthorGroupOut], summary="List watchlists"
)
async def list_author_groups(
    principal: RequireRead,
    project_id: str,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[AuthorGroupOut]:
    if mock.demo_mode():
        from datetime import timedelta

        from app.mock.corpus import DEMO_NOW, _stable_uuid

        items = [
            AuthorGroupOut(
                id=_stable_uuid("author_group", name),
                project_id=mock.corpus().project_id,
                name=name,
                description=description,
                is_curated=True,
                member_count=count,
                created_at=DEMO_NOW - timedelta(days=20),
            )
            for name, description, count in (
                (
                    "Russian State Affiliated Accounts",
                    "Analyst-maintained. Membership is an assertion, not a model output.",
                    14,
                ),
                (
                    "Known Hostile Telegram Channels",
                    "Cross-referenced against published research; reviewed monthly.",
                    9,
                ),
            )
        ]
        return mock.paginate(items, page, {"project_id": project_id})
    from app.repositories.actors import list_author_groups as query_groups

    return await query_groups(session, project_id, page)


@router.post(
    "/author-groups",
    response_model=AuthorGroupOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a watchlist",
)
async def create_author_group(
    body: AuthorGroupCreate,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AuthorGroupOut:
    if mock.demo_mode():
        from app.mock.corpus import DEMO_NOW, _stable_uuid

        return AuthorGroupOut(
            id=_stable_uuid("author_group", body.name),
            project_id=body.project_id,
            name=body.name,
            description=body.description,
            is_curated=True,
            member_count=0,
            created_at=DEMO_NOW,
        )
    from app.repositories.actors import create_author_group as create_group

    return await create_group(session, body)


@router.post(
    "/author-groups/{group_id}/members",
    response_model=AuthorGroupMembersResult,
    summary="Add authors to a watchlist",
    description=(
        "Ids that match no author in this project are returned in `unknown` rather "
        "than silently dropped: a watchlist that quietly lost half its entries is "
        "worse than one that failed loudly."
    ),
)
async def add_author_group_members(
    group_id: str,
    body: AuthorGroupMembers,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AuthorGroupMembersResult:
    if mock.demo_mode():
        known = {a.author_id for a in mock.corpus().authors}
        matched = [a for a in body.author_ids if a in known]
        return AuthorGroupMembersResult(
            group_id=group_id,
            added=len(matched),
            already_present=0,
            unknown=[a for a in body.author_ids if a not in known],
        )
    from app.repositories.actors import add_group_members

    return await add_group_members(session, group_id, body.author_ids)
