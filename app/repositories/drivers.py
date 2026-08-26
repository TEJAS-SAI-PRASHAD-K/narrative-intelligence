"""Conversation drivers -- authors, hashtags and URLs, one query shape.

The three tabs differ only in *what is counted*, so they share one Core query
parameterised by the entity expression. Three near-identical hand-written
queries is how the tabs end up quietly disagreeing about what a post count
includes.

Built with SQLAlchemy Core rather than a text() template on purpose: the filter
subquery carries user input (``q``), and composing it into a string -- even with
``literal_binds``, which does escape correctly -- means the safety of this query
depends on a compiler flag being right rather than on the value never reaching
the SQL text at all.
"""

from __future__ import annotations

import logging

from sqlalchemy import Float, cast, distinct, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import FilterSpec, Page
from app.mock.responses import DEFINITIONS
from app.repositories.filters import apply_post_filters, decode_cursor, resolve_project_id
from app.schemas.drivers import DriverItem, DriversResponse

log = logging.getLogger(__name__)

#: What counts as bot-like. Mirrors configs/fusion.yaml's threshold; both read
#: the same config in the scoring path, and this is the display-side default.
BOT_THRESHOLD = 0.6


async def drivers(
    session: AsyncSession, spec: FilterSpec, page: Page, entity_type: str
) -> DriversResponse:
    from app.models.actors import Author
    from app.models.corpus import Post, PostScore

    project_id = await resolve_project_id(session, spec.project_id)
    matching = apply_post_filters(select(Post.id).select_from(Post), spec, project_id).subquery()

    if entity_type == "author":
        entity = Post.author_id
        label = func.max(Post.author_handle)
        source = select(Post)
    else:
        column = Post.hashtags if entity_type == "hashtag" else Post.urls
        # LATERAL unnest: one row per entity per post, so a post with three
        # hashtags contributes to three counts and to none of them twice.
        unnested = func.unnest(column).alias("entity_value")
        entity = literal_column("entity_value")
        label = literal_column("entity_value")
        source = select(Post).join(unnested, literal_column("true"))

    scored_authors = func.count(distinct(Author.author_id)).filter(Author.bot_score.isnot(None))
    bot_authors = func.count(distinct(Author.author_id)).filter(Author.bot_score > BOT_THRESHOLD)

    stmt = (
        source.with_only_columns(
            entity.label("entity"),
            label.label("label"),
            func.count().label("post_count"),
            func.count(distinct(Post.author_id)).label("author_count"),
            func.sum(
                func.coalesce(Post.likes, 0)
                + func.coalesce(Post.shares, 0)
                + func.coalesce(Post.replies, 0)
            ).label("engagement"),
            # Over the authors the classifier could score, not over all of them:
            # counting an unscorable news outlet as human deflates the share on
            # exactly the drivers that news carries.
            (cast(bot_authors, Float) / func.nullif(scored_authors, 0)).label("bot_like_share"),
            func.avg(PostScore.toxicity).label("mean_toxicity"),
            func.min(Post.timestamp).label("first_seen"),
            func.max(Post.timestamp).label("last_seen"),
        )
        .outerjoin(PostScore, PostScore.post_id == Post.id)
        .outerjoin(
            Author,
            (Author.author_id == Post.author_id) & (Author.project_id == project_id),
        )
        .where(Post.id.in_(select(matching.c.id)))
        .group_by(entity)
        .order_by(func.count().desc())
    )

    offset = decode_cursor(page.cursor).get("o", 0)
    rows = (await session.execute(stmt.offset(offset).limit(page.limit))).all()

    return DriversResponse(
        entity_type=entity_type,
        items=[
            DriverItem(
                entity=str(row.entity),
                entity_type=entity_type,
                label=row.label,
                post_count=row.post_count,
                # Always 1 on the author tab, so it is omitted there rather than
                # rendered as a meaningless column.
                author_count=None if entity_type == "author" else row.author_count,
                engagement_total=int(row.engagement or 0),
                bot_like_share=(
                    round(float(row.bot_like_share), 4) if row.bot_like_share is not None else None
                ),
                mean_toxicity=(
                    round(float(row.mean_toxicity), 4) if row.mean_toxicity is not None else None
                ),
                first_seen=row.first_seen,
                last_seen=row.last_seen,
            )
            for row in rows
        ],
        definition=DEFINITIONS[f"drivers_{entity_type}s"],
    )
