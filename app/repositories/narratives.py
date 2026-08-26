"""Narrative queries.

The one to read carefully is :func:`update_narrative`. An analyst edit records
*which fields* were edited, not just that an edit happened, so a reclustering
run can refresh a summary nobody touched while leaving a hand-written title
alone. Storing a single boolean would force a choice between overwriting an
analyst's work and freezing a narrative's machine-generated fields forever.
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import utcnow
from app.deps import FilterSpec, Page
from app.errors import BadRequest, NotFound
from app.repositories.filters import (
    count_rows,
    decode_cursor,
    encode_cursor,
    resolve_project_id,
)
from app.schemas.actors import AuthorOut
from app.schemas.common import AiProvenance, GeneratedText, PageResponse, ScoreComponent
from app.schemas.narratives import (
    NarrativeCohortShare,
    NarrativeCreate,
    NarrativeDetail,
    NarrativeFreshness,
    NarrativeScorecard,
    NarrativeScoreExplanation,
    NarrativeSummary,
    NarrativeTimeline,
    NarrativeUpdate,
    RepresentativePost,
    TimelinePoint,
)
from app.schemas.posts import PostOut

log = logging.getLogger(__name__)


def _provenance(generated_by: str, model: str | None, at, edited: bool) -> AiProvenance:
    """Never guessed.

    A centroid-keyword label is `heuristic` and must not get the AI pill;
    claiming a model wrote something it did not is the same class of error as
    hiding that one did.
    """
    kind = generated_by if generated_by in ("ai", "human", "heuristic") else "heuristic"
    return AiProvenance(
        generated_by=kind,
        model=model if kind == "ai" else None,
        generated_at=at if kind == "ai" else None,
        prompt_version=None,
        edited_by_user=edited,
    )


def to_summary(row, scorecard=None) -> NarrativeSummary:
    edited = bool(row.edited_by_user)
    return NarrativeSummary(
        id=str(row.id),
        display_id=row.display_id,
        project_id=str(row.project_id),
        title=GeneratedText(
            text=row.title,
            provenance=_provenance(
                row.title_generated_by,
                row.summary_model,
                row.summary_generated_at,
                edited and "title" in (row.edited_fields or ()),
            ),
        ),
        summary=GeneratedText(
            text=row.summary,
            provenance=_provenance(
                row.summary_generated_by,
                row.summary_model,
                row.summary_generated_at,
                edited and "summary" in (row.edited_fields or ()),
            ),
        ),
        claim=row.claim,
        scorecard=to_scorecard(scorecard),
        date_start=row.date_start,
        date_end=row.date_end,
        post_count=row.post_count or 0,
        author_count=row.author_count or 0,
        engagement_total=row.engagement_total or 0,
        platforms=list(row.platforms or ()),
        top_domains=list(row.top_domains or ()),
        top_hashtags=list(row.top_hashtags or ()),
        is_manual=bool(row.is_manual),
        cluster_id=row.cluster_id,
        clustering_run_id=str(row.clustering_run_id) if row.clustering_run_id else None,
        updated_at=row.updated_at,
    )


def to_scorecard(row) -> NarrativeScorecard:
    """A narrative with no scorecard yet is `low`, with everything null.

    Not an error and not a zero: score.fusion has simply not run over it, and
    the UI renders the empty state rather than a confident-looking zero.
    """
    if row is None:
        return NarrativeScorecard(priority="low", missing=["not_scored"])
    return NarrativeScorecard(
        priority=row.priority or "low",
        bot_like=row.bot_like,
        anomalous=row.anomalous,
        toxicity=row.toxicity,
        compass_risk=row.compass_risk,
        negative_sentiment=row.negative_sentiment,
        fusion_score=row.fusion_score,
        scoring_version=row.scoring_version,
        computed_at=row.computed_at,
        missing=list((row.components or {}).get("missing", ())),
    )


async def list_narratives(
    session: AsyncSession,
    spec: FilterSpec,
    page: Page,
    *,
    sort: str = "priority",
    q: str | None = None,
) -> PageResponse[NarrativeSummary]:
    from app.models.narratives import Narrative
    from app.models.narratives import NarrativeScorecard as Scorecard

    project_id = await resolve_project_id(session, spec.project_id)
    stmt = (
        select(Narrative, Scorecard)
        .outerjoin(Scorecard, Scorecard.narrative_id == Narrative.id)
        .where(Narrative.project_id == project_id)
    )

    if q:
        needle = f"%{q.lower()}%"
        stmt = stmt.where(
            func.lower(func.coalesce(Narrative.title, "")).like(needle)
            | func.lower(func.coalesce(Narrative.summary, "")).like(needle)
            | func.lower(func.coalesce(Narrative.claim, "")).like(needle)
        )
    if spec.platform:
        stmt = stmt.where(Narrative.platforms.overlap(list(spec.platform)))
    if spec.min_risk is not None:
        stmt = stmt.where(Scorecard.fusion_score >= spec.min_risk)
    if spec.date_from:
        stmt = stmt.where(Narrative.date_end >= spec.date_from)
    if spec.date_to:
        stmt = stmt.where(Narrative.date_start < spec.date_to)

    orderings = {
        # NULLS LAST throughout: an unscored narrative sorts to the bottom of a
        # risk-ordered feed rather than to the top, which is where NULL would
        # land by default on a DESC sort and would be actively misleading.
        "priority": (Scorecard.fusion_score.desc().nullslast(), Narrative.id.desc()),
        "date": (Narrative.date_end.desc().nullslast(), Narrative.id.desc()),
        "engagement": (Narrative.engagement_total.desc(), Narrative.id.desc()),
        "posts": (Narrative.post_count.desc(), Narrative.id.desc()),
    }
    stmt = stmt.order_by(*orderings.get(sort, orderings["priority"]))

    offset = decode_cursor(page.cursor).get("o", 0)
    total = await count_rows(session, stmt)
    rows = (await session.execute(stmt.offset(offset).limit(page.limit))).all()

    return PageResponse[NarrativeSummary](
        items=[to_summary(narrative, card) for narrative, card in rows],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=total,
        filters_applied={**spec.as_response_dict(), "sort": sort, "q": q},
    )


async def _load(session: AsyncSession, narrative_id: str):
    from app.models.narratives import Narrative
    from app.models.narratives import NarrativeScorecard as Scorecard

    try:
        parsed = uuid.UUID(narrative_id)
    except ValueError:
        raise NotFound(
            f"No narrative with id {narrative_id}.", code="narrative_not_found"
        ) from None

    row = (
        await session.execute(
            select(Narrative, Scorecard)
            .outerjoin(Scorecard, Scorecard.narrative_id == Narrative.id)
            .where(Narrative.id == parsed)
        )
    ).first()
    if row is None:
        raise NotFound(f"No narrative with id {narrative_id}.", code="narrative_not_found")
    return row


async def get_narrative(session: AsyncSession, narrative_id: str) -> NarrativeDetail:
    from app.models.corpus import Post
    from app.models.narratives import NarrativePost

    narrative, card = await _load(session, narrative_id)
    representatives = (
        await session.execute(
            select(Post, NarrativePost.membership_score)
            .join(NarrativePost, NarrativePost.post_id == Post.id)
            .where(NarrativePost.narrative_id == narrative.id)
            # Representatives first, then by membership confidence. A feed card
            # showing an arbitrary member misrepresents what the cluster is.
            .order_by(
                NarrativePost.is_representative.desc(),
                NarrativePost.membership_score.desc().nullslast(),
            )
            .limit(5)
        )
    ).all()

    has_compass = bool(
        (
            await session.execute(
                text(
                    "SELECT 1 FROM compass_contexts "
                    "WHERE narrative_id = :n AND superseded_by IS NULL LIMIT 1"
                ),
                {"n": narrative.id},
            )
        ).scalar()
    )

    return NarrativeDetail(
        **to_summary(narrative, card).model_dump(),
        representative_posts=[
            RepresentativePost(
                post_id=post.id,
                text=post.text_,
                source=post.source,
                author_handle=post.author_handle,
                timestamp=post.timestamp,
                membership_score=score,
            )
            for post, score in representatives
        ],
        coherence=narrative.coherence,
        velocity=narrative.velocity,
        compass_available=has_compass,
    )


async def create_manual_narrative(session: AsyncSession, body: NarrativeCreate) -> NarrativeDetail:
    from app.models.narratives import Narrative, NarrativePost

    project_id = await resolve_project_id(session, body.project_id)
    row = Narrative(
        project_id=project_id,
        title=body.title,
        # A hand-built narrative is human-authored end to end. Marking it `ai`
        # would put the generated-by pill on an analyst's own words.
        title_generated_by="human",
        summary=body.summary,
        summary_generated_by="human",
        claim=body.claim,
        is_manual=True,
        edited_by_user=True,
        edited_fields=["title", "summary", "claim"],
        edited_at=utcnow(),
    )
    session.add(row)
    await session.flush()

    if body.post_ids:
        known = set(
            (
                await session.execute(
                    text("SELECT id FROM posts WHERE id = ANY(:ids) AND project_id = :p"),
                    {"ids": body.post_ids, "p": project_id},
                )
            ).scalars()
        )
        unknown = [pid for pid in body.post_ids if pid not in known]
        if unknown:
            # Refuse rather than silently build a narrative missing half its
            # evidence: a curated narrative's membership is an assertion.
            raise BadRequest(
                f"{len(unknown)} post id(s) are not in this project.",
                code="unknown_post_ids",
                detail={"unknown": unknown[:20]},
            )
        for post_id in known:
            session.add(NarrativePost(narrative_id=row.id, post_id=post_id, membership_score=1.0))
        row.post_count = len(known)

    await session.commit()
    return await get_narrative(session, str(row.id))


async def update_narrative(
    session: AsyncSession, narrative_id: str, body: NarrativeUpdate, *, actor: str
) -> NarrativeDetail:
    """Record which fields the analyst changed, not merely that they did."""
    narrative, _ = await _load(session, narrative_id)

    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise BadRequest("No fields to update.", code="empty_update")

    edited = set(narrative.edited_fields or ())
    for field, value in changes.items():
        setattr(narrative, field, value)
        edited.add(field)
        # The field is now human-authored. Leaving generated_by as `ai` would
        # keep the pill on text an analyst wrote.
        if field == "title":
            narrative.title_generated_by = "human"
        elif field == "summary":
            narrative.summary_generated_by = "human"

    narrative.edited_by_user = True
    narrative.edited_fields = sorted(edited)
    narrative.edited_by = actor
    narrative.edited_at = utcnow()
    await session.commit()

    log.info("narrative %s edited by %s: fields=%s", narrative.id, actor, sorted(changes))
    return await get_narrative(session, narrative_id)


async def narrative_timeline(
    session: AsyncSession, narrative_id: str, interval: str, tz: str, cross_platform: bool
) -> NarrativeTimeline:
    from app.models.corpus import Post
    from app.models.narratives import NarrativePost

    narrative, _ = await _load(session, narrative_id)
    width = {"1h": "1 hour", "6h": "6 hours", "1d": "1 day", "7d": "7 days"}[interval]
    bucket = func.date_bin(
        text(f"interval '{width}'"), Post.timestamp, text("timestamp '2000-01-01 00:00+00'")
    ).label("bucket")

    rows = (
        await session.execute(
            select(
                bucket,
                Post.source,
                func.count(Post.id),
                func.count(func.distinct(Post.author_id)),
                func.sum(
                    func.coalesce(Post.likes, 0)
                    + func.coalesce(Post.shares, 0)
                    + func.coalesce(Post.replies, 0)
                ),
            )
            .join(NarrativePost, NarrativePost.post_id == Post.id)
            .where(NarrativePost.narrative_id == narrative.id)
            .group_by(bucket, Post.source)
            .order_by(bucket)
        )
    ).all()

    buckets: dict[Any, dict[str, Any]] = {}
    for bucket_start, source, posts, authors, engagement in rows:
        entry = buckets.setdefault(
            bucket_start, {"posts": 0, "authors": 0, "engagement": 0, "platforms": {}}
        )
        entry["posts"] += posts
        entry["authors"] += authors
        entry["engagement"] += int(engagement or 0)
        entry["platforms"][source] = posts

    points = [
        TimelinePoint(
            bucket_start=start,
            post_count=data["posts"],
            engagement=data["engagement"],
            author_count=data["authors"],
            by_platform=data["platforms"] if cross_platform else None,
        )
        for start, data in sorted(buckets.items())
    ]

    peak = max(points, key=lambda p: p.post_count) if points else None
    mean = sum(p.post_count for p in points) / len(points) if points else 0
    # Flagged server-side because a six-hour spike on a sixty-day axis is easy
    # for the eye to miss, and it is the signal the whole product is about.
    bursts = [
        {
            "bucket_start": p.bucket_start.isoformat(),
            "post_count": p.post_count,
            "times_mean": round(p.post_count / mean, 2),
        }
        for p in points
        if mean and p.post_count > 3 * mean
    ]

    return NarrativeTimeline(
        narrative_id=str(narrative.id),
        interval=interval,
        tz=tz,
        points=points,
        peak_at=peak.bucket_start if peak else None,
        bursts=bursts,
    )


async def narrative_posts(
    session: AsyncSession, narrative_id: str, spec: FilterSpec, page: Page
) -> PageResponse[PostOut]:
    import dataclasses

    from app.repositories.posts import list_posts

    narrative, _ = await _load(session, narrative_id)
    # Reuse the shared post query rather than writing a second one: one filter
    # implementation is the whole point of FilterSpec.
    scoped = dataclasses.replace(spec, narrative_id=(str(narrative.id),))
    return await list_posts(session, scoped, page)


async def narrative_authors(
    session: AsyncSession, narrative_id: str, spec: FilterSpec, page: Page
) -> PageResponse[AuthorOut]:
    import dataclasses

    from app.repositories.actors import list_authors

    narrative, _ = await _load(session, narrative_id)
    scoped = dataclasses.replace(spec, narrative_id=(str(narrative.id),))
    return await list_authors(session, scoped, page)


async def narrative_cohorts(session: AsyncSession, narrative_id: str) -> list[NarrativeCohortShare]:
    narrative, _ = await _load(session, narrative_id)
    rows = (
        await session.execute(
            text(
                """
                SELECT c.id, c.name, c.category,
                       count(DISTINCT p.author_id) AS authors,
                       count(*) AS posts
                FROM narrative_posts np
                JOIN posts p ON p.id = np.post_id
                JOIN author_cohorts ac ON ac.author_id = p.author_id
                JOIN cohorts c ON c.id = ac.cohort_id
                WHERE np.narrative_id = :n
                GROUP BY c.id, c.name, c.category
                ORDER BY count(*) DESC
                """
            ),
            {"n": narrative.id},
        )
    ).all()

    total = narrative.post_count or sum(row.posts for row in rows) or 1
    return [
        NarrativeCohortShare(
            cohort_id=str(row.id),
            name=row.name,
            category=row.category,
            author_count=row.authors,
            post_count=row.posts,
            share_pct=round(100 * row.posts / total, 2),
        )
        for row in rows
    ]


async def narrative_score(session: AsyncSession, narrative_id: str) -> NarrativeScoreExplanation:
    """The explainability endpoint. Serves the stored decomposition verbatim.

    Nothing is recomputed here. The components blob was written by the same run
    that produced the score, so what the drilldown shows is provably what
    produced the number -- recomputing on read would let the two drift the
    moment somebody edited the weights.
    """
    narrative, card = await _load(session, narrative_id)
    if card is None or not card.components:
        raise NotFound(
            "This narrative has not been scored yet. Run score.fusion for its project.",
            code="scorecard_not_found",
            detail={"narrative_id": narrative_id},
        )

    blob = card.components
    return NarrativeScoreExplanation(
        narrative_id=str(narrative.id),
        fusion_score=card.fusion_score,
        priority=card.priority or "low",
        scoring_version=card.scoring_version or blob.get("scoring_version", "unknown"),
        formula=blob.get("formula", ""),
        normalization=blob.get("normalization", ""),
        components=[
            ScoreComponent(
                name=component["name"],
                value=component["value"],
                weight=component["weight"],
                contribution=component["contribution"],
                definition=component["definition"],
                inputs={
                    **component.get("inputs", {}),
                    "missing": component.get("missing", []),
                    "configured_weight": component.get("configured_weight"),
                },
            )
            for component in blob.get("components", [])
        ],
        missing=list(blob.get("missing", ())),
        weights_renormalized=bool(blob.get("weights_renormalized")),
        computed_at=card.computed_at,
    )


async def clustering_freshness(session: AsyncSession, project_id: str) -> NarrativeFreshness:
    from app.models.narratives import ClusteringRun

    resolved = await resolve_project_id(session, project_id)
    run = (
        await session.execute(
            select(ClusteringRun)
            .where(ClusteringRun.project_id == resolved)
            .order_by(ClusteringRun.started_at.desc().nullslast())
            .limit(1)
        )
    ).scalar_one_or_none()

    counts = (
        await session.execute(
            text(
                """
                SELECT
                    (SELECT count(*) FROM narratives WHERE project_id = :p) AS narratives,
                    (SELECT count(*) FROM posts WHERE project_id = :p) AS posts,
                    (SELECT count(*) FROM posts p
                     WHERE p.project_id = :p
                       AND NOT EXISTS (
                           SELECT 1 FROM narrative_posts np WHERE np.post_id = p.id
                       )) AS unclustered
                """
            ),
            {"p": resolved},
        )
    ).one()

    if run is None:
        return NarrativeFreshness(
            project_id=str(resolved),
            age_human="never",
            status="never_run",
            post_count=counts.posts,
            narrative_count=counts.narratives,
            unclustered_post_count=counts.unclustered,
        )

    finished = run.finished_at or run.started_at
    age = (utcnow() - finished) if finished else None
    return NarrativeFreshness(
        project_id=str(resolved),
        last_run_id=str(run.id),
        last_run_at=finished,
        age_seconds=age.total_seconds() if age else None,
        age_human=_humanize(age),
        # Wall-clock age is the weaker signal. What actually makes a clustering
        # stale is posts that arrived after it and were never assigned, so that
        # drives the status.
        status=(
            "running"
            if run.status in ("pending", "running")
            else ("stale" if counts.unclustered > 0.05 * max(counts.posts, 1) else "fresh")
        ),
        algorithm=run.algorithm,
        embedding_model=run.embedding_model,
        post_count=counts.posts,
        narrative_count=counts.narratives,
        unclustered_post_count=counts.unclustered,
    )


def _humanize(age: timedelta | None) -> str:
    """Formatted server-side so every surface says it the same way."""
    if age is None:
        return "unknown"
    days, seconds = age.days, age.seconds
    if days:
        return f"{days} day{'s' if days != 1 else ''}, {seconds // 3600} hrs ago"
    if seconds >= 3600:
        return f"{seconds // 3600} hrs, {(seconds % 3600) // 60} min ago"
    if seconds >= 60:
        return f"{seconds // 60} min ago"
    return "just now"
