"""Author, cohort and watchlist queries."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import utcnow
from app.deps import FilterSpec, Page
from app.errors import NotFound
from app.repositories.filters import (
    apply_post_filters,
    count_rows,
    decode_cursor,
    encode_cursor,
    resolve_project_id,
)
from app.schemas.actors import (
    AuthorGroupCreate,
    AuthorGroupMembersResult,
    AuthorGroupOut,
    AuthorOut,
    AuthorScoreExplanation,
    AuthorTimeline,
    AuthorTimelinePoint,
    CohortMembership,
    CohortOut,
)
from app.schemas.common import PageResponse, ScoreComponent
from app.schemas.posts import PostOut

log = logging.getLogger(__name__)


def to_author_out(row, cohorts=None, group_ids=None) -> AuthorOut:
    components = row.score_components or {}
    return AuthorOut(
        author_id=row.author_id,
        project_id=str(row.project_id),
        source=row.source,
        handle=row.handle,
        created_at_source=row.created_at_source,
        followers=row.followers,
        following=row.following,
        post_count=row.post_count or 0,
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        is_bot_flagged=row.is_bot_flagged,
        bot_score=row.bot_score,
        risk_score=row.risk_score,
        partisan_lean=row.partisan_lean,
        dominant_sentiment=row.dominant_sentiment,
        dominant_emotion=row.dominant_emotion,
        anomalous_score=row.anomalous_score,
        toxicity_score=row.toxicity_score,
        coordination_score=row.coordination_score,
        community_id=row.community_id,
        community_size=row.community_size,
        cohorts=cohorts or [],
        author_group_ids=[str(g) for g in (group_ids or ())],
        narratives_touched=list(components.get("narratives_touched", ())),
        scoring_version=row.scoring_version,
        skip_reasons=list(row.skip_reasons or ()),
    )


async def _cohorts_for(session: AsyncSession, author_ids: list[str]) -> dict[str, list]:
    """Memberships for a page of authors in one query.

    An author is routinely in three cohorts, so this cannot be a column and it
    must not be a per-row lookup: at 50 authors that is 50 round trips for what
    one IN clause answers.
    """
    if not author_ids:
        return {}
    rows = (
        await session.execute(
            text(
                """
                SELECT ac.author_id, c.id, c.name, c.category, ac.confidence, ac.assigned_by
                FROM author_cohorts ac JOIN cohorts c ON c.id = ac.cohort_id
                WHERE ac.author_id = ANY(:ids)
                """
            ),
            {"ids": author_ids},
        )
    ).all()
    out: dict[str, list] = {}
    for row in rows:
        out.setdefault(row.author_id, []).append(
            CohortMembership(
                cohort_id=str(row.id),
                name=row.name,
                category=row.category,
                confidence=row.confidence,
                assigned_by=row.assigned_by or "model",
            )
        )
    return out


async def list_authors(
    session: AsyncSession, spec: FilterSpec, page: Page, *, sort: str = "risk"
) -> PageResponse[AuthorOut]:
    from app.models.actors import Author
    from app.models.corpus import Post

    project_id = await resolve_project_id(session, spec.project_id)
    # Authors are selected by the posts that match the filter, so the same
    # filter bar means the same thing on this page as on every other one.
    matching_posts = apply_post_filters(
        select(Post.author_id).select_from(Post), spec, project_id
    ).subquery()

    stmt = select(Author).where(
        Author.project_id == project_id,
        Author.author_id.in_(select(matching_posts.c.author_id)),
    )
    if spec.is_bot_like is not None:
        stmt = (
            stmt.where(Author.bot_score.isnot(None), Author.bot_score > 0.6)
            if spec.is_bot_like
            else stmt.where((Author.bot_score.is_(None)) | (Author.bot_score <= 0.6))
        )
    if spec.cohort_id:
        from app.models.actors import AuthorCohort

        stmt = stmt.where(
            Author.author_id.in_(
                select(AuthorCohort.author_id).where(
                    AuthorCohort.cohort_id.in_([uuid.UUID(c) for c in spec.cohort_id])
                )
            )
        )

    orderings = {
        "risk": (Author.risk_score.desc().nullslast(),),
        "bot_score": (Author.bot_score.desc().nullslast(),),
        "posts": (Author.post_count.desc(),),
        "followers": (Author.followers.desc().nullslast(),),
    }
    stmt = stmt.order_by(*orderings.get(sort, orderings["risk"]), Author.author_id)

    offset = decode_cursor(page.cursor).get("o", 0)
    total = await count_rows(session, stmt)
    rows = (await session.execute(stmt.offset(offset).limit(page.limit))).scalars().all()
    cohorts = await _cohorts_for(session, [r.author_id for r in rows])

    return PageResponse[AuthorOut](
        items=[to_author_out(row, cohorts.get(row.author_id)) for row in rows],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=total,
        filters_applied={**spec.as_response_dict(), "sort": sort},
    )


async def _load_author(session: AsyncSession, author_id: str):
    from app.models.actors import Author

    row = (
        await session.execute(select(Author).where(Author.author_id == author_id))
    ).scalar_one_or_none()
    if row is None:
        raise NotFound(f"No author with id {author_id}.", code="author_not_found")
    return row


async def get_author(session: AsyncSession, author_id: str) -> AuthorOut:
    row = await _load_author(session, author_id)
    cohorts = await _cohorts_for(session, [author_id])
    groups = (
        (
            await session.execute(
                text("SELECT group_id FROM author_group_members WHERE author_id = :a"),
                {"a": author_id},
            )
        )
        .scalars()
        .all()
    )
    return to_author_out(row, cohorts.get(author_id), groups)


async def author_score(session: AsyncSession, author_id: str) -> AuthorScoreExplanation:
    """Feature-level attribution, served from what the scorer stored.

    A bot probability with no explanation is the black box this product exists
    to argue against, so an author the classifier could not score returns the
    reason rather than a number.
    """
    row = await _load_author(session, author_id)
    components = row.score_components or {}
    features = components.get("top_features") or []

    if row.bot_score is None:
        return AuthorScoreExplanation(
            author_id=author_id,
            bot_score=None,
            risk_score=row.risk_score,
            model="xgboost-bot-clf",
            scoring_version=row.scoring_version or "unscored",
            top_features=[],
            missing=["bot_score"],
            caveat=_skip_caveat(row.skip_reasons),
        )

    return AuthorScoreExplanation(
        author_id=author_id,
        bot_score=row.bot_score,
        risk_score=row.risk_score,
        model="xgboost-bot-clf",
        scoring_version=row.scoring_version or "unknown",
        top_features=[
            ScoreComponent(
                name=feature.get("name", "unknown"),
                value=None,
                weight=abs(float(feature.get("contribution") or 0.0)),
                contribution=float(feature.get("contribution") or 0.0),
                definition=(
                    f"SHAP contribution of {feature.get('name')} toward the bot class. "
                    "Positive pushes toward 'bot', negative toward 'human'."
                ),
                inputs={},
            )
            for feature in sorted(
                features, key=lambda f: -abs(float(f.get("contribution") or 0.0))
            )[:8]
        ],
        computed_at=None,
        # Surfaced rather than buried in a report appendix. A single high score
        # on an unfamiliar campaign is a lead, not a finding, and the model card
        # says so.
        caveat=(
            "Bot generalisation to unseen campaigns is weak: per-fold macro-F1 is "
            "0.416 +/- 0.187 against a pooled 0.705. Treat one high score as a lead "
            "for review, not as a determination."
        ),
    )


def _skip_caveat(skip_reasons) -> str:
    reasons = set(skip_reasons or ())
    if "author_is_outlet" in reasons:
        return (
            "This source's 'authors' are outlets, not accounts. Account-level bot "
            "scoring is skipped with a reason code rather than producing a number."
        )
    if reasons:
        return f"Not scored: {', '.join(sorted(reasons))}."
    return "This author has not been scored yet."


async def author_timeline(
    session: AsyncSession, author_id: str, interval: str, tz: str
) -> AuthorTimeline:
    from app.models.corpus import Post, PostScore

    await _load_author(session, author_id)
    width = {"1h": "1 hour", "6h": "6 hours", "1d": "1 day", "7d": "7 days"}[interval]
    bucket = func.date_bin(
        text(f"interval '{width}'"), Post.timestamp, text("timestamp '2000-01-01 00:00+00'")
    ).label("bucket")

    rows = (
        await session.execute(
            select(
                bucket,
                func.count(Post.id),
                func.sum(
                    func.coalesce(Post.likes, 0)
                    + func.coalesce(Post.shares, 0)
                    + func.coalesce(Post.replies, 0)
                ),
                func.avg(PostScore.toxicity),
            )
            .outerjoin(PostScore, PostScore.post_id == Post.id)
            .where(Post.author_id == author_id)
            .group_by(bucket)
            .order_by(bucket)
        )
    ).all()

    # Posting-hour histogram in the requested zone. A flat 24-hour profile on a
    # supposedly-human account is one of the cheapest coordination tells there
    # is, so it ships with the timeline rather than needing a second call.
    histogram_rows = (
        await session.execute(
            text(
                'SELECT extract(hour FROM ("timestamp" AT TIME ZONE :tz))::int AS hour, '
                "count(*) FROM posts WHERE author_id = :a GROUP BY hour ORDER BY hour"
            ),
            {"a": author_id, "tz": tz},
        )
    ).all()

    return AuthorTimeline(
        author_id=author_id,
        interval=interval,
        tz=tz,
        points=[
            AuthorTimelinePoint(
                bucket_start=bucket_start,
                post_count=posts,
                engagement=int(engagement or 0),
                mean_toxicity=round(float(toxicity), 4) if toxicity is not None else None,
            )
            for bucket_start, posts, engagement, toxicity in rows
        ],
        hour_histogram={int(hour): count for hour, count in histogram_rows},
    )


async def author_posts(
    session: AsyncSession, author_id: str, spec: FilterSpec, page: Page
) -> PageResponse[PostOut]:
    from app.models.corpus import Post
    from app.repositories.posts import _narrative_ids_for, _scores_for, to_post_out

    await _load_author(session, author_id)
    project_id = await resolve_project_id(session, spec.project_id)
    stmt = apply_post_filters(select(Post).select_from(Post), spec, project_id).where(
        Post.author_id == author_id
    )
    stmt = stmt.order_by(Post.timestamp.desc(), Post.id.desc())

    offset = decode_cursor(page.cursor).get("o", 0)
    total = await count_rows(session, stmt)
    rows = (await session.execute(stmt.offset(offset).limit(page.limit))).scalars().all()
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
        filters_applied=spec.as_response_dict(),
    )


async def list_cohorts(
    session: AsyncSession, spec: FilterSpec, page: Page
) -> PageResponse[CohortOut]:
    from app.models.actors import Cohort

    project_id = await resolve_project_id(session, spec.project_id)
    rows = (
        (
            await session.execute(
                select(Cohort)
                .where(Cohort.project_id == project_id)
                .order_by(Cohort.author_count.desc(), Cohort.name)
                .limit(page.limit)
            )
        )
        .scalars()
        .all()
    )

    stats = {
        row.cohort_id: row
        for row in (
            await session.execute(
                text(
                    """
                    SELECT ac.cohort_id,
                           avg(a.bot_score) AS mean_bot,
                           avg(a.toxicity_score) AS mean_tox
                    FROM author_cohorts ac
                    JOIN authors a ON a.author_id = ac.author_id
                    WHERE a.project_id = :p
                    GROUP BY ac.cohort_id
                    """
                ),
                {"p": project_id},
            )
        ).all()
    }

    return PageResponse[CohortOut](
        items=[
            CohortOut(
                id=str(row.id),
                project_id=str(row.project_id),
                name=row.name,
                description=row.description,
                category=row.category,
                author_count=row.author_count or 0,
                post_pct=row.post_pct,
                mean_bot_score=(
                    float(stats[row.id].mean_bot)
                    if row.id in stats and stats[row.id].mean_bot is not None
                    else None
                ),
                mean_toxicity=(
                    float(stats[row.id].mean_tox)
                    if row.id in stats and stats[row.id].mean_tox is not None
                    else None
                ),
            )
            for row in rows
        ],
        total=len(rows),
        filters_applied=spec.as_response_dict(),
    )


async def cohort_authors(
    session: AsyncSession, cohort_id: str, spec: FilterSpec, page: Page
) -> PageResponse[AuthorOut]:
    import dataclasses

    from app.models.actors import Cohort

    if (await session.get(Cohort, cohort_id)) is None:
        raise NotFound(f"No cohort with id {cohort_id}.", code="cohort_not_found")
    return await list_authors(session, dataclasses.replace(spec, cohort_id=(cohort_id,)), page)


async def list_author_groups(
    session: AsyncSession, project_id: str, page: Page
) -> PageResponse[AuthorGroupOut]:
    from app.models.actors import AuthorGroup

    resolved = await resolve_project_id(session, project_id)
    rows = (
        (
            await session.execute(
                select(AuthorGroup)
                .where(AuthorGroup.project_id == resolved)
                .order_by(AuthorGroup.name)
                .limit(page.limit)
            )
        )
        .scalars()
        .all()
    )
    counts = dict(
        (
            await session.execute(
                text(
                    "SELECT group_id, count(*) FROM author_group_members "
                    "WHERE group_id = ANY(:ids) GROUP BY group_id"
                ),
                {"ids": [row.id for row in rows] or [uuid.uuid4()]},
            )
        ).all()
    )
    return PageResponse[AuthorGroupOut](
        items=[
            AuthorGroupOut(
                id=str(row.id),
                project_id=str(row.project_id),
                name=row.name,
                description=row.description,
                is_curated=row.is_curated,
                member_count=counts.get(row.id, 0),
                created_at=row.created_at,
            )
            for row in rows
        ],
        total=len(rows),
        filters_applied={"project_id": project_id},
    )


async def create_author_group(session: AsyncSession, body: AuthorGroupCreate) -> AuthorGroupOut:
    from app.models.actors import AuthorGroup

    project_id = await resolve_project_id(session, body.project_id)
    row = AuthorGroup(
        project_id=project_id, name=body.name, description=body.description, is_curated=True
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return AuthorGroupOut(
        id=str(row.id),
        project_id=str(row.project_id),
        name=row.name,
        description=row.description,
        is_curated=True,
        member_count=0,
        created_at=row.created_at,
    )


async def add_group_members(
    session: AsyncSession, group_id: str, author_ids: list[str]
) -> AuthorGroupMembersResult:
    """Add authors to a watchlist, reporting what did not match.

    Unknown ids come back in ``unknown`` rather than being dropped. A watchlist
    that quietly lost half its entries is worse than one that failed loudly:
    the analyst believes the accounts are being watched.
    """
    from app.models.actors import AuthorGroup, AuthorGroupMember

    group = await session.get(AuthorGroup, group_id)
    if group is None:
        raise NotFound(f"No author group with id {group_id}.", code="author_group_not_found")

    known = set(
        (
            await session.execute(
                text(
                    "SELECT author_id FROM authors WHERE author_id = ANY(:ids) AND project_id = :p"
                ),
                {"ids": author_ids, "p": group.project_id},
            )
        ).scalars()
    )
    existing = set(
        (
            await session.execute(
                text(
                    "SELECT author_id FROM author_group_members "
                    "WHERE group_id = :g AND author_id = ANY(:ids)"
                ),
                {"g": group.id, "ids": list(known) or [""]},
            )
        ).scalars()
    )

    added = 0
    for author_id in known - existing:
        session.add(AuthorGroupMember(group_id=group.id, author_id=author_id, added_at=utcnow()))
        added += 1
    await session.commit()

    return AuthorGroupMembersResult(
        group_id=str(group.id),
        added=added,
        already_present=len(existing),
        unknown=[a for a in author_ids if a not in known],
    )
