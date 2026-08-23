"""One query builder for the shared filter spec.

Every list endpoint applies the same predicates through this module. Duplicating
the logic per route is how a codebase like this rots: six months on, two
endpoints disagree about what ``min_toxicity`` includes, both look reasonable,
and nobody can say which is right.

The one rule worth stating loudly: **a NULL score is never swept in by a
threshold filter.** ``min_toxicity=0.0`` means "posts measured at 0.0 or above",
not "every post including the ones the scorer skipped". Treating unmeasured as
zero here would quietly change every count on the page.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import Any

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import FilterSpec
from app.errors import NotFound

log = logging.getLogger(__name__)


async def resolve_project_id(session: AsyncSession, identifier: str) -> uuid.UUID:
    """Slug or uuid to a project uuid. Cached per request by the caller."""
    from app.models.core import Project

    row = (
        await session.execute(select(Project.id).where(Project.slug == identifier))
    ).scalar_one_or_none()
    if row is not None:
        return row
    try:
        parsed = uuid.UUID(identifier)
    except ValueError:
        raise NotFound(
            f"No project with slug or id {identifier!r}.", code="project_not_found"
        ) from None
    exists = (
        await session.execute(select(Project.id).where(Project.id == parsed))
    ).scalar_one_or_none()
    if exists is None:
        raise NotFound(f"No project with id {parsed}.", code="project_not_found")
    return exists


def apply_post_filters(stmt: Select, spec: FilterSpec, project_id: uuid.UUID) -> Select:
    """Apply the shared filter spec to a statement selecting from ``posts``.

    Assumes ``posts`` is already in the FROM clause and joins ``post_scores``
    only when a score predicate is actually requested -- an unconditional LEFT
    JOIN would cost the planner a merge on every unfiltered page view.
    """
    from app.models.corpus import Post, PostScore

    conditions = [Post.project_id == project_id]

    if spec.date_from:
        conditions.append(Post.timestamp >= spec.date_from)
    if spec.date_to:
        conditions.append(Post.timestamp < spec.date_to)
    if spec.platform:
        conditions.append(Post.source.in_(spec.platform))
    if spec.content_type:
        conditions.append(Post.content_type.in_(spec.content_type))
    if spec.lang:
        conditions.append(Post.lang.in_(spec.lang))
    if not spec.include_shared:
        # The UI's "Original data" chip. A repost swarm can otherwise inflate
        # every count on the overview by an order of magnitude.
        conditions.append(Post.parent_id.is_(None))
    if spec.q:
        # Built from SQLAlchemy functions rather than a text() fragment with a
        # named bind. A text() bind does not survive being wrapped in the
        # subquery that count_rows() builds, and the failure is an
        # InvalidRequestError at execute time rather than anything visible here.
        #
        # The expression must match ix_posts_text_fts exactly -- same
        # dictionary, no cast -- or the planner silently drops to a sequential
        # scan over the whole corpus.
        conditions.append(
            func.to_tsvector("english", Post.text_).op("@@")(
                func.plainto_tsquery("english", spec.q)
            )
        )

    needs_scores = any(
        (
            spec.sentiment,
            spec.emotion,
            spec.min_toxicity is not None,
            spec.is_anomalous is not None,
        )
    )
    if needs_scores:
        stmt = stmt.join(PostScore, PostScore.post_id == Post.id)
        if spec.sentiment:
            conditions.append(PostScore.sentiment.in_(spec.sentiment))
        if spec.emotion:
            conditions.append(PostScore.emotion.in_(spec.emotion))
        if spec.min_toxicity is not None:
            # NOT NULL is explicit and load-bearing. `toxicity >= 0.0` would
            # otherwise be true for measured zeros and false for nulls, which is
            # correct SQL and the wrong answer to what the user asked.
            conditions.append(
                and_(PostScore.toxicity.isnot(None), PostScore.toxicity >= spec.min_toxicity)
            )
        if spec.is_anomalous is not None:
            conditions.append(PostScore.is_anomalous.is_(spec.is_anomalous))

    if spec.narrative_id:
        from app.models.narratives import NarrativePost

        conditions.append(
            Post.id.in_(
                select(NarrativePost.post_id).where(
                    NarrativePost.narrative_id.in_([uuid.UUID(n) for n in spec.narrative_id])
                )
            )
        )

    if spec.cohort_id or spec.author_group_id or spec.is_bot_like is not None:
        stmt, conditions = _apply_author_filters(stmt, conditions, spec)

    return stmt.where(and_(*conditions))


def _apply_author_filters(stmt: Select, conditions: list, spec: FilterSpec):
    """Author-scoped predicates, expressed as semi-joins.

    IN (subquery) rather than a JOIN because an author can belong to several
    cohorts: joining would multiply post rows by cohort membership and every
    count on the page would silently inflate. A semi-join cannot do that.
    """
    from app.models.actors import Author, AuthorCohort, AuthorGroupMember
    from app.models.corpus import Post

    if spec.cohort_id:
        conditions.append(
            Post.author_id.in_(
                select(AuthorCohort.author_id).where(
                    AuthorCohort.cohort_id.in_([uuid.UUID(c) for c in spec.cohort_id])
                )
            )
        )
    if spec.author_group_id:
        conditions.append(
            Post.author_id.in_(
                select(AuthorGroupMember.author_id).where(
                    AuthorGroupMember.group_id.in_([uuid.UUID(g) for g in spec.author_group_id])
                )
            )
        )
    if spec.is_bot_like is not None:
        bot_like = select(Author.author_id).where(
            Author.bot_score.isnot(None), Author.bot_score > 0.6
        )
        conditions.append(
            Post.author_id.in_(bot_like) if spec.is_bot_like else Post.author_id.notin_(bot_like)
        )
    return stmt, conditions


# ---------------------------------------------------------------------------
# cursor pagination
# ---------------------------------------------------------------------------
def encode_cursor(payload: dict[str, Any]) -> str:
    """Opaque on purpose.

    A frontend that learns to do arithmetic on a cursor is a frontend that
    breaks when the sort key changes. Base64 of a small JSON object is not
    security, it is a fence.
    """
    return base64.urlsafe_b64encode(json.dumps(payload, default=str).encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> dict[str, Any]:
    if not cursor:
        return {}
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        # A malformed cursor restarts the listing rather than 400-ing. The user
        # pasted a truncated URL; showing them page one is the useful answer.
        log.debug("ignoring malformed cursor %r", cursor[:40])
        return {}


def keyset_where(model, cursor: dict[str, Any], *, descending: bool = True):
    """A keyset predicate over ``(timestamp, id)``.

    The tiebreaker on the primary key is not optional: thousands of posts in
    this corpus share a timestamp to the second, and a keyset on time alone
    either skips rows or repeats them at every page boundary.
    """
    if not cursor or "ts" not in cursor:
        return None
    from datetime import datetime

    ts = datetime.fromisoformat(cursor["ts"])
    last_id = cursor.get("id", "")
    if descending:
        return or_(
            model.timestamp < ts,
            and_(model.timestamp == ts, model.id < last_id),
        )
    return or_(
        model.timestamp > ts,
        and_(model.timestamp == ts, model.id > last_id),
    )


async def count_rows(session: AsyncSession, stmt: Select, *, limit: int = 100_000) -> int | None:
    """An exact count, or None when it would cost more than the page.

    Counting a filtered ten-million-row table to render "1-50 of ..." is a
    sequential scan per page view. Past the ceiling this returns None and the
    response says `total: null`, which the contract explicitly allows -- an
    honest null beats a guessed number.
    """
    counted = await session.execute(select(func.count()).select_from(stmt.limit(limit).subquery()))
    total = counted.scalar() or 0
    return total if total < limit else None
