"""Post queries: listing, detail, threads, search and kNN.

Routers never write SQL. Everything that touches the database for the posts
surface is here, so an EXPLAIN that needs fixing has exactly one place to be
fixed.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import FilterSpec, Page
from app.errors import BadRequest, NotFound
from app.repositories.filters import (
    apply_post_filters,
    count_rows,
    decode_cursor,
    encode_cursor,
    keyset_where,
    resolve_project_id,
)
from app.schemas.common import PageResponse
from app.schemas.posts import (
    Engagement,
    PostOut,
    PostScores,
    SimilarPost,
    SimilarRequest,
    SimilarResponse,
    ThreadNode,
    ThreadResponse,
)

log = logging.getLogger(__name__)


def to_post_out(post, score=None, *, narrative_ids=None, include_raw: bool = False) -> PostOut:
    return PostOut(
        id=post.id,
        project_id=str(post.project_id),
        native_id=post.native_id,
        source=post.source,
        source_detail=post.source_detail,
        content_type=post.content_type,
        text=post.text_,
        lang=post.lang,
        author_id=post.author_id,
        author_handle=post.author_handle,
        timestamp=post.timestamp,
        parent_id=post.parent_id,
        conversation_id=post.conversation_id,
        # Straight through. No coalescing anywhere on this path.
        engagement=Engagement(
            likes=post.likes, shares=post.shares, replies=post.replies, views=post.views
        ),
        urls=list(post.urls or ()),
        domains=list(post.domains or ()),
        media_urls=list(post.media_urls or ()),
        hashtags=list(post.hashtags or ()),
        mentions=list(post.mentions or ()),
        scores=_to_scores(score),
        narrative_ids=[str(n) for n in (narrative_ids or ())],
        raw=post.raw if include_raw else None,
    )


def _to_scores(score) -> PostScores | None:
    if score is None:
        return None
    return PostScores(
        toxicity=score.toxicity,
        is_toxic=score.is_toxic,
        anomaly=score.anomaly,
        is_anomalous=score.is_anomalous,
        misinfo_likelihood=score.misinfo_likelihood,
        stance=score.stance,
        sentiment=score.sentiment,
        sentiment_score=score.sentiment_score,
        emotion=score.emotion,
        emotion_scores=score.emotion_scores,
        scoring_version=score.scoring_version,
        scored_at=score.scored_at,
        skip_reasons=list(score.skip_reasons or ()),
    )


async def _narrative_ids_for(session: AsyncSession, post_ids: list[str]) -> dict[str, list[str]]:
    """Membership for a page of posts, in one query.

    One query for the page rather than one per post: at 50 posts a naive
    per-row lookup is 50 round trips, which is the entire latency budget for
    the endpoint spent on a lookup table.
    """
    if not post_ids:
        return {}
    from app.models.narratives import NarrativePost

    rows = await session.execute(
        select(NarrativePost.post_id, NarrativePost.narrative_id).where(
            NarrativePost.post_id.in_(post_ids)
        )
    )
    out: dict[str, list[str]] = {}
    for post_id, narrative_id in rows:
        out.setdefault(post_id, []).append(str(narrative_id))
    return out


async def list_posts(
    session: AsyncSession,
    spec: FilterSpec,
    page: Page,
    *,
    sort: str = "recent",
) -> PageResponse[PostOut]:
    from app.models.corpus import Post, PostScore

    project_id = await resolve_project_id(session, spec.project_id)
    stmt = select(Post).select_from(Post)
    stmt = apply_post_filters(stmt, spec, project_id)

    cursor = decode_cursor(page.cursor)
    if sort in ("recent", "oldest"):
        descending = sort == "recent"
        predicate = keyset_where(Post, cursor, descending=descending)
        if predicate is not None:
            stmt = stmt.where(predicate)
        stmt = stmt.order_by(
            Post.timestamp.desc() if descending else Post.timestamp.asc(),
            Post.id.desc() if descending else Post.id.asc(),
        )
    elif sort == "engagement":
        # Nulls are not zero, so they sort last explicitly rather than by
        # whatever the planner's default happens to be for the direction.
        total = func.coalesce(Post.likes, 0) + func.coalesce(Post.shares, 0)
        stmt = stmt.order_by(total.desc(), Post.id.desc()).offset(cursor.get("o", 0))
    else:  # risk
        stmt = (
            stmt.join(PostScore, PostScore.post_id == Post.id)
            .where(PostScore.misinfo_likelihood.isnot(None))
            .order_by(PostScore.misinfo_likelihood.desc(), Post.id.desc())
            .offset(cursor.get("o", 0))
        )

    total = await count_rows(session, stmt)
    rows = (await session.execute(stmt.limit(page.limit))).scalars().all()
    scores = await _scores_for(session, [r.id for r in rows])
    membership = await _narrative_ids_for(session, [r.id for r in rows])

    items = [
        to_post_out(row, scores.get(row.id), narrative_ids=membership.get(row.id)) for row in rows
    ]
    return PageResponse[PostOut](
        items=items,
        next_cursor=_next_cursor(rows, sort, cursor, page.limit),
        total=total,
        filters_applied={**spec.as_response_dict(), "sort": sort},
    )


def _next_cursor(rows, sort: str, cursor: dict[str, Any], limit: int) -> str | None:
    if len(rows) < limit:
        return None
    if sort in ("recent", "oldest"):
        last = rows[-1]
        return encode_cursor({"ts": last.timestamp.isoformat(), "id": last.id})
    # The engagement and risk sorts have no stable keyset (the sort key is
    # computed and not unique), so they fall back to an offset cursor. It is
    # still opaque, so the frontend never learns the difference.
    return encode_cursor({"o": cursor.get("o", 0) + limit})


async def _scores_for(session: AsyncSession, post_ids: list[str]) -> dict[str, Any]:
    if not post_ids:
        return {}
    from app.models.corpus import PostScore

    rows = (
        await session.execute(select(PostScore).where(PostScore.post_id.in_(post_ids)))
    ).scalars()
    return {row.post_id: row for row in rows}


async def get_post(session: AsyncSession, post_id: str) -> PostOut:
    from app.models.corpus import Post, PostScore

    post = await session.get(Post, post_id)
    if post is None:
        raise NotFound(f"No post with id {post_id}.", code="post_not_found")
    score = await session.get(PostScore, post_id)
    membership = await _narrative_ids_for(session, [post_id])
    return to_post_out(post, score, narrative_ids=membership.get(post_id), include_raw=True)


async def search_posts(
    session: AsyncSession, spec: FilterSpec, page: Page, *, q: str
) -> PageResponse[PostOut]:
    """Full-text search ranked by relevance.

    ``plainto_tsquery`` rather than ``to_tsquery``: the input is whatever an
    analyst typed into a search box, and ``to_tsquery`` raises a syntax error on
    an unbalanced quote or a bare ampersand.
    """
    import dataclasses

    from app.models.corpus import Post

    project_id = await resolve_project_id(session, spec.project_id)
    rank = func.ts_rank(
        func.to_tsvector("english", Post.text_),
        func.plainto_tsquery("english", q),
    )
    # The explicit `q` parameter wins. Clearing it on the spec avoids applying
    # the same predicate twice when a caller sets both -- harmless for
    # correctness, but it doubles the ts_rank work and muddies the EXPLAIN.
    stmt = select(Post).select_from(Post)
    stmt = apply_post_filters(stmt, dataclasses.replace(spec, q=None), project_id)
    stmt = stmt.where(
        func.to_tsvector("english", Post.text_).op("@@")(func.plainto_tsquery("english", q))
    )

    cursor = decode_cursor(page.cursor)
    offset = cursor.get("o", 0)
    stmt = stmt.order_by(rank.desc(), Post.id.desc()).offset(offset)

    total = await count_rows(session, stmt)
    rows = (await session.execute(stmt.limit(page.limit))).scalars().all()
    scores = await _scores_for(session, [r.id for r in rows])
    membership = await _narrative_ids_for(session, [r.id for r in rows])

    return PageResponse[PostOut](
        items=[
            to_post_out(row, scores.get(row.id), narrative_ids=membership.get(row.id))
            for row in rows
        ],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=total,
        filters_applied={**spec.as_response_dict(), "q": q},
    )


async def post_thread(session: AsyncSession, post_id: str) -> ThreadResponse:
    """Reconstruct a conversation from ``parent_id``.

    A recursive CTE bounded at depth 50. Unbounded recursion over
    self-referential user data is a hang waiting for a cycle, and threading data
    from a scraped corpus is not guaranteed acyclic.
    """
    from app.models.corpus import Post

    post = await session.get(Post, post_id)
    if post is None:
        raise NotFound(f"No post with id {post_id}.", code="post_not_found")

    if not post.conversation_id:
        # Kaggle-sourced Reddit rows carry no parent_id and no conversation id.
        # Saying so is better than rendering a single orphan as a thread root.
        return ThreadResponse(
            conversation_id=None,
            root_id=post.id,
            nodes=[ThreadNode(post=to_post_out(post), depth=0, children=[])],
            detail=(
                "This post carries no conversation id, so no thread can be "
                "reconstructed. Some Phase 1 sources do not expose threading."
            ),
        )

    rows = (
        (
            await session.execute(
                select(Post)
                .where(Post.conversation_id == post.conversation_id)
                .order_by(Post.timestamp.asc())
                .limit(2000)
            )
        )
        .scalars()
        .all()
    )

    by_id = {row.id: row for row in rows}
    children: dict[str, list[str]] = {}
    for row in rows:
        if row.parent_id and row.parent_id in by_id:
            children.setdefault(row.parent_id, []).append(row.id)

    def depth_of(row) -> int:
        depth, cursor, seen = 0, row, set()
        while cursor.parent_id and cursor.parent_id in by_id and cursor.id not in seen:
            seen.add(cursor.id)
            cursor = by_id[cursor.parent_id]
            depth += 1
            if depth > 50:
                break
        return depth

    roots = [row for row in rows if not row.parent_id or row.parent_id not in by_id]
    scores = await _scores_for(session, list(by_id))
    return ThreadResponse(
        conversation_id=post.conversation_id,
        root_id=roots[0].id if roots else None,
        nodes=[
            ThreadNode(
                post=to_post_out(row, scores.get(row.id)),
                depth=depth_of(row),
                children=children.get(row.id, []),
            )
            for row in rows
        ],
        truncated=len(rows) >= 2000,
        detail=(None if roots else "No root post for this conversation is present in the corpus."),
    )


async def similar_posts(session: AsyncSession, body: SimilarRequest) -> SimilarResponse:
    """pgvector kNN. Cosine distance, HNSW index, no exceptions."""
    from app.config import get_api_settings
    from app.models.embeddings import PostEmbedding

    settings = get_api_settings()
    project_id = await resolve_project_id(session, body.project_id)

    if body.post_id:
        query_vector = (
            await session.execute(
                select(PostEmbedding.embedding).where(PostEmbedding.post_id == body.post_id)
            )
        ).scalar_one_or_none()
        if query_vector is None:
            raise NotFound(
                f"No embedding for post {body.post_id}. It may not have been embedded yet; "
                "check /api/v1/jobs for a running nlp.embed task.",
                code="embedding_not_found",
            )
    else:
        query_vector = await _embed_text(body.text or "")

    from app.models.corpus import Post

    distance = PostEmbedding.embedding.cosine_distance(query_vector)
    stmt = (
        select(Post, distance.label("distance"))
        .join(PostEmbedding, PostEmbedding.post_id == Post.id)
        .where(Post.project_id == project_id)
        .order_by(distance)
        .limit(body.limit + (1 if body.post_id else 0))
    )
    if body.post_id:
        stmt = stmt.where(Post.id != body.post_id)
    if body.exclude_same_author:
        source_author = (
            await session.execute(select(Post.author_id).where(Post.id == body.post_id))
        ).scalar_one_or_none()
        if source_author:
            stmt = stmt.where(Post.author_id != source_author)
    if body.min_similarity is not None:
        stmt = stmt.where(distance <= 1 - body.min_similarity)

    # ef_search is per-session and trades recall for latency. Set it on the
    # query rather than globally so a slow, thorough kNN and a fast dashboard
    # query can coexist.
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {settings.hnsw_ef_search}"))

    rows = (await session.execute(stmt)).all()
    scores = await _scores_for(session, [row[0].id for row in rows])

    return SimilarResponse(
        query_post_id=body.post_id,
        query_text=body.text,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        items=[
            SimilarPost(
                post=to_post_out(post, scores.get(post.id)),
                # Reported as both, because the UI shows similarity and the
                # index ranks by distance. Converting in the frontend is how a
                # threshold ends up inverted.
                similarity=round(1 - float(dist), 6),
                distance=round(float(dist), 6),
            )
            for post, dist in rows[: body.limit]
        ],
    )


async def _embed_text(text_value: str) -> list[float]:
    """Embed raw query text, or 503 with the checkpoint that is missing."""
    from nlp import availability

    availability.require("embed")
    from nlp.adapters import get_embedder

    embedder = get_embedder()
    if embedder is None:
        from app.errors import ScorerUnavailable

        raise ScorerUnavailable(
            "Raw-text similarity needs a live embedding model. No checkpoint is "
            "mounted, so only `post_id` queries are available -- those read a "
            "vector that already exists in the table.",
            detail={"capability": "embed"},
        )
    if not text_value.strip():
        raise BadRequest("text must not be empty.", code="empty_query")
    return embedder.encode([text_value])[0].tolist()
