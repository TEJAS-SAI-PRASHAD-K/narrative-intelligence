"""Posts, threads, full-text search and vector similarity."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import Filters, Pagination, RequireRead, rate_limit
from app.errors import BadRequest, NotFound
from app.mock import responses as mock
from app.schemas.common import PageResponse
from app.schemas.posts import (
    PostOut,
    SimilarRequest,
    SimilarResponse,
    ThreadResponse,
)

router = APIRouter(prefix="/posts", tags=["posts"], dependencies=[Depends(rate_limit)])


def _demo_post(post_id: str):
    post = mock.corpus().by_post.get(post_id)
    if post is None:
        raise NotFound(f"No post with id {post_id}.", code="post_not_found")
    return post


@router.get("", response_model=PageResponse[PostOut], summary="List posts")
async def list_posts(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
    sort: Annotated[str, Query(pattern="^(recent|oldest|engagement|risk)$")] = "recent",
) -> PageResponse[PostOut]:
    if mock.demo_mode():
        posts = mock.apply_post_filters(mock.corpus().posts, filters)
        sorters = {
            "recent": lambda p: -p.timestamp.timestamp(),
            "oldest": lambda p: p.timestamp.timestamp(),
            "engagement": lambda p: (
                -sum(v for v in (p.likes, p.shares, p.replies) if v is not None)
            ),
            "risk": lambda p: -((p.scores or {}).get("misinfo_likelihood") or -1),
        }
        posts = sorted(posts, key=sorters[sort])
        return mock.paginate(
            [mock.to_post(p) for p in posts], page, {**filters.as_response_dict(), "sort": sort}
        )
    from app.repositories.posts import list_posts as query_posts

    return await query_posts(session, filters, page, sort=sort)


@router.get(
    "/search",
    response_model=PageResponse[PostOut],
    summary="Full-text search",
    description=(
        "Postgres `tsvector` search over post text, ranked by relevance. Declared "
        "before `/{post_id}` because FastAPI matches in declaration order and a "
        "literal segment behind a parameter of the same shape is unreachable."
    ),
)
async def search_posts(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
    q: Annotated[str, Query(min_length=1, max_length=500)] = "",
) -> PageResponse[PostOut]:
    if mock.demo_mode():
        needle = q.lower()
        posts = [
            p
            for p in mock.apply_post_filters(mock.corpus().posts, filters)
            if needle in p.text.lower()
        ]
        return mock.paginate(
            [mock.to_post(p) for p in posts], page, {**filters.as_response_dict(), "q": q}
        )
    from app.repositories.posts import search_posts as query_search

    return await query_search(session, filters, page, q=q)


@router.post(
    "/similar",
    response_model=SimilarResponse,
    summary="Nearest neighbours by embedding",
    description=(
        "pgvector kNN over the HNSW index, cosine distance throughout. Accepts either "
        "a `post_id` (whose vector is already stored) or raw `text` (which needs a "
        "live embedder and 503s when no embedding checkpoint is mounted)."
    ),
)
async def similar_posts(
    body: SimilarRequest,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SimilarResponse:
    if not body.post_id and not body.text:
        raise BadRequest("Provide either post_id or text.", code="missing_query")
    if body.post_id and body.text:
        raise BadRequest(
            "Provide post_id or text, not both. Two query vectors have no defined "
            "combination and silently preferring one would surprise the caller.",
            code="ambiguous_query",
        )

    if mock.demo_mode():
        from app.config import get_api_settings
        from app.schemas.posts import SimilarPost

        settings = get_api_settings()
        source = _demo_post(body.post_id) if body.post_id else None
        pool = [p for p in mock.corpus().posts if not source or p.id != source.id]
        if body.exclude_same_author and source:
            pool = [p for p in pool if p.author_id != source.author_id]
        if source and source.narrative_ids:
            pool.sort(key=lambda p: 0 if set(p.narrative_ids) & set(source.narrative_ids) else 1)
        items = []
        for index, post in enumerate(pool[: body.limit]):
            similarity = round(max(-1.0, 0.94 - 0.011 * index), 4)
            if body.min_similarity is not None and similarity < body.min_similarity:
                break
            items.append(
                SimilarPost(
                    post=mock.to_post(post),
                    similarity=similarity,
                    distance=round(1 - similarity, 4),
                )
            )
        return SimilarResponse(
            query_post_id=body.post_id,
            query_text=body.text,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            items=items,
        )

    from app.repositories.posts import similar_posts as query_similar

    return await query_similar(session, body)


@router.get(
    "/{post_id:path}/thread",
    response_model=ThreadResponse,
    summary="Reconstruct a conversation",
)
async def post_thread(
    post_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ThreadResponse:
    if mock.demo_mode():
        from app.schemas.posts import ThreadNode

        post = _demo_post(post_id)
        members = [p for p in mock.corpus().posts if p.conversation_id == post.conversation_id]
        members.sort(key=lambda p: p.timestamp)
        children: dict[str, list[str]] = {}
        for member in members:
            if member.parent_id:
                children.setdefault(member.parent_id, []).append(member.id)
        roots = [m for m in members if m.parent_id is None]

        def depth_of(member) -> int:
            depth, cursor, seen = 0, member, set()
            while cursor.parent_id and cursor.parent_id not in seen:
                seen.add(cursor.id)
                cursor = mock.corpus().by_post.get(cursor.parent_id)
                if cursor is None:
                    break
                depth += 1
            return depth

        return ThreadResponse(
            conversation_id=post.conversation_id,
            root_id=roots[0].id if roots else None,
            nodes=[
                ThreadNode(post=mock.to_post(m), depth=depth_of(m), children=children.get(m.id, []))
                for m in members
            ],
            truncated=False,
            detail=(
                None
                if roots
                else (
                    "No root post is present in the corpus for this conversation. "
                    "Kaggle-sourced Reddit rows carry no parent_id, and rendering an "
                    "orphan as a root would misstate the thread."
                )
            ),
        )
    from app.repositories.posts import post_thread as query_thread

    return await query_thread(session, post_id)


@router.get(
    "/{post_id:path}",
    response_model=PostOut,
    summary="Post detail",
    description="Backs the 'Original Post' preview. Includes `raw`, which list responses omit.",
)
async def get_post(
    post_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PostOut:
    if mock.demo_mode():
        return mock.to_post(_demo_post(post_id), include_raw=True)
    from app.repositories.posts import get_post as query_post

    return await query_post(session, post_id)
