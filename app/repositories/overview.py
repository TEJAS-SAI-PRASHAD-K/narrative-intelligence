"""Overview aggregations.

Every function here returns the definition text alongside the number. That text
lives next to the query that produced it, deliberately: if the query changes and
the definition does not, the diff shows it.

The engagement definition is the one to read carefully. Views are excluded from
"engagements" because they measure exposure, not engagement, and a corpus with
YouTube in it would otherwise look two orders of magnitude more engaged than the
same corpus without. Nulls contribute nothing rather than zero, which is why
``nullable_fields`` is populated on that metric and not on the post count.
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta

from sqlalchemy import Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import FilterSpec, Page
from app.mock.responses import DEFINITIONS
from app.repositories.filters import apply_post_filters, resolve_project_id
from app.schemas.common import Metric, PageResponse
from app.schemas.overview import (
    ConceptItem,
    ConceptsResponse,
    KpiBundle,
    TimeseriesPoint,
    TimeseriesResponse,
)
from app.schemas.posts import HighRiskPost

log = logging.getLogger(__name__)

_INTERVALS = {"1h": "1 hour", "6h": "6 hours", "1d": "1 day", "7d": "7 days"}

#: Ceiling on emitted buckets. The real corpus spans 2018-2026 because ConvoKit
#: Reddit is historical, so `interval=1h` over the full range is ~70,000 points
#: -- a multi-megabyte response that no chart can draw. Past this the empty-
#: bucket fill is skipped and the response says so in its definition text, which
#: keeps the payload shape identical while making the degradation visible.
MAX_BUCKETS = 1500


def _filtered_post_ids(spec: FilterSpec, project_id: uuid.UUID) -> Select:
    """The id set the whole overview page agrees on.

    Every KPI on the page is computed against this one subquery rather than each
    re-deriving the filter. Two numbers on one screen disagreeing because two
    queries interpreted `include_shared` differently is the exact failure the
    shared FilterSpec exists to prevent.
    """
    from app.models.corpus import Post

    return apply_post_filters(select(Post.id).select_from(Post), spec, project_id)


async def kpi_bundle(session: AsyncSession, spec: FilterSpec) -> KpiBundle:
    from app.models.corpus import Post, PostScore

    project_id = await resolve_project_id(session, spec.project_id)
    ids = _filtered_post_ids(spec, project_id).subquery()

    row = (
        await session.execute(
            select(
                func.count(Post.id),
                func.count(func.distinct(Post.author_id)),
                # SUM over a nullable column already skips nulls in SQL, which
                # is exactly the semantics we want. Spelled out because the
                # instinct to wrap it in COALESCE is what breaks it.
                func.sum(Post.likes),
                func.sum(Post.shares),
                func.sum(Post.replies),
                func.min(Post.timestamp),
                func.max(Post.timestamp),
            ).where(Post.id.in_(select(ids.c.id)))
        )
    ).one()
    posts, authors, likes, shares, replies, first, last = row
    engagement = sum(v for v in (likes, shares, replies) if v is not None)

    sentiment_rows = dict(
        (
            await session.execute(
                select(PostScore.sentiment, func.count())
                .where(PostScore.post_id.in_(select(ids.c.id)), PostScore.sentiment.isnot(None))
                .group_by(PostScore.sentiment)
            )
        ).all()
    )
    scored = sum(sentiment_rows.values())
    net = (
        round(
            # positive - negative, matching /overview/kpis/sentiment exactly. Both
            # numbers appear on the same screen, and a sign disagreement between two
            # figures a user reads together is worse than either being absent.
            100 * (sentiment_rows.get("positive", 0) - sentiment_rows.get("negative", 0)) / scored,
            1,
        )
        if scored
        else None
    )

    emotion_rows = (
        await session.execute(
            select(PostScore.emotion, func.count())
            .where(PostScore.post_id.in_(select(ids.c.id)), PostScore.emotion.isnot(None))
            .group_by(PostScore.emotion)
            .order_by(func.count().desc())
            .limit(1)
        )
    ).first()

    return KpiBundle(
        posts=Metric(
            value=posts, label="Posts", definition=DEFINITIONS["posts"], sample_size=posts
        ),
        engagements=Metric(
            value=engagement,
            label="Engagements",
            definition=DEFINITIONS["engagements"],
            nullable_fields=["views", "likes", "shares", "replies"],
            sample_size=posts,
        ),
        authors=Metric(
            value=authors,
            label="Authors",
            definition=DEFINITIONS["authors"],
            sample_size=posts,
        ),
        sentiment=Metric(
            value=net,
            label="Net sentiment",
            definition=DEFINITIONS["sentiment"],
            unit="pct",
            sample_size=scored,
        ),
        emotions=Metric(
            value=emotion_rows[1] if emotion_rows else None,
            label=f"Top emotion: {emotion_rows[0]}" if emotion_rows else "Top emotion",
            definition=DEFINITIONS["emotions"],
            sample_size=scored,
        ),
        window_start=spec.date_from or first,
        window_end=spec.date_to or last,
        tz=spec.tz,
    )


async def timeseries(
    session: AsyncSession,
    spec: FilterSpec,
    metric: str,
    group_by: str,
    interval: str,
    normalize: bool,
    agg: str,
) -> TimeseriesResponse:
    """Bucketed volume.

    ``date_bin`` with a generated series so empty buckets are emitted as zero.
    A GROUP BY alone omits them, and a line chart with the gaps closed up
    misrepresents a quiet week as a short one.
    """
    from app.models.corpus import Post

    project_id = await resolve_project_id(session, spec.project_id)
    ids = _filtered_post_ids(spec, project_id).subquery()
    width = _INTERVALS[interval]

    value = (
        func.count(Post.id)
        if metric == "posts"
        else (
            func.count(func.distinct(Post.author_id))
            if metric == "authors"
            else func.sum(
                func.coalesce(Post.likes, 0)
                + func.coalesce(Post.shares, 0)
                + func.coalesce(Post.replies, 0)
            )
        )
    )

    group_column = {
        "platform": Post.source,
        "source_detail": Post.source_detail,
    }.get(group_by)

    bucket = func.date_bin(
        text(f"interval '{width}'"), Post.timestamp, text("timestamp '2000-01-01 00:00+00'")
    ).label("bucket")

    columns = [bucket, value.label("value")]
    grouping = [bucket]
    if group_column is not None:
        columns.insert(1, group_column.label("series"))
        grouping.append(group_column)

    stmt = (
        select(*columns).where(Post.id.in_(select(ids.c.id))).group_by(*grouping).order_by(bucket)
    )
    rows = (await session.execute(stmt)).all()

    buckets: dict[object, dict[str, float]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        if group_column is not None:
            bucket_start, series_key, series_value = row
            labels[str(series_key)] = str(series_key)
        else:
            bucket_start, series_value = row
            series_key = "all"
            labels["all"] = "All"
        buckets.setdefault(bucket_start, {})[str(series_key)] = float(series_value or 0)

    filled, capped = _fill_empty_buckets(buckets, interval)

    if normalize:
        peaks: dict[str, float] = {}
        for series in filled.values():
            for key, val in series.items():
                peaks[key] = max(peaks.get(key, 0.0), val)
        for series in filled.values():
            for key in list(series):
                if peaks.get(key):
                    series[key] = round(series[key] / peaks[key], 4)

    return TimeseriesResponse(
        metric=metric,
        interval=interval,
        group_by=group_by,
        agg=agg,
        normalized=normalize,
        tz=spec.tz,
        points=[
            TimeseriesPoint(
                bucket_start=start,
                value=round(sum(series.values()), 4),
                series=series if group_by != "none" else None,
            )
            for start, series in sorted(filled.items())
        ],
        series_labels=labels,
        definition=(
            DEFINITIONS["timeseries"]
            + (
                f" This range exceeds {MAX_BUCKETS} buckets at the requested interval, "
                "so empty buckets are omitted rather than emitted as zero. Narrow the "
                "date range or widen the interval for a continuous axis."
                if capped
                else ""
            )
        ),
    )


def _fill_empty_buckets(buckets: dict, interval: str) -> tuple[dict, bool]:
    """Emit zero for buckets with no rows. Returns (buckets, was_capped).

    Done in Python rather than with generate_series because the range depends on
    the filtered data, and the second query needed to find that range costs more
    than filling a few hundred dict entries.

    A gap in a line chart has to read as a gap: omitting empty buckets makes a
    quiet month look like a short one. But the fill is bounded -- an eight-year
    corpus at hourly resolution is 70,000 points, and returning that is worse
    than not filling at all.
    """
    if not buckets:
        return buckets, False
    hours = {"1h": 1, "6h": 6, "1d": 24, "7d": 168}[interval]
    step = timedelta(hours=hours)
    keys = sorted(buckets)
    span = int((keys[-1] - keys[0]) / step) + 1
    if span > MAX_BUCKETS:
        return buckets, True
    cursor = keys[0]
    while cursor <= keys[-1]:
        buckets.setdefault(cursor, {})
        cursor += step
    return buckets, False


async def concepts(session: AsyncSession, spec: FilterSpec) -> ConceptsResponse:
    """Hashtags and domains by volume, with the mean sentiment of their posts.

    Hashtags and domains only for now: named-entity extraction would mean
    running a model inside a request, which this layer does not do. When the NER
    pass lands it writes to a table and this reads from it.
    """
    from app.models.corpus import Post, PostScore

    project_id = await resolve_project_id(session, spec.project_id)
    ids = _filtered_post_ids(spec, project_id).subquery()

    items: list[ConceptItem] = []
    for column, kind in ((Post.hashtags, "hashtag"), (Post.domains, "domain")):
        term = func.unnest(column).label("term")
        stmt = (
            select(
                term,
                func.count().label("volume"),
                func.avg(PostScore.sentiment_score).label("sentiment"),
            )
            .select_from(Post)
            .outerjoin(PostScore, PostScore.post_id == Post.id)
            .where(Post.id.in_(select(ids.c.id)))
            .group_by(term)
            .order_by(func.count().desc())
            .limit(40)
        )
        for row in (await session.execute(stmt)).all():
            items.append(
                ConceptItem(
                    term=row.term,
                    kind=kind,
                    volume=row.volume,
                    sentiment_score=(
                        round(float(row.sentiment), 3) if row.sentiment is not None else None
                    ),
                )
            )

    items.sort(key=lambda item: -item.volume)
    return ConceptsResponse(items=items, total_terms=len(items), definition=DEFINITIONS["concepts"])


async def high_risk_posts(
    session: AsyncSession, spec: FilterSpec, page: Page
) -> PageResponse[HighRiskPost]:
    from app.models.actors import Author
    from app.models.corpus import Post, PostScore
    from app.models.narratives import Narrative, NarrativePost
    from app.repositories.filters import decode_cursor, encode_cursor
    from app.repositories.posts import to_post_out

    project_id = await resolve_project_id(session, spec.project_id)
    ids = _filtered_post_ids(spec, project_id).subquery()
    offset = decode_cursor(page.cursor).get("o", 0)

    stmt = (
        select(Post, PostScore, Author.bot_score, Narrative.id, Narrative.title)
        .join(PostScore, PostScore.post_id == Post.id)
        .outerjoin(Author, Author.author_id == Post.author_id)
        .outerjoin(NarrativePost, NarrativePost.post_id == Post.id)
        .outerjoin(Narrative, Narrative.id == NarrativePost.narrative_id)
        .where(
            Post.id.in_(select(ids.c.id)),
            PostScore.misinfo_likelihood.isnot(None),
        )
        .order_by(PostScore.misinfo_likelihood.desc(), Post.id.desc())
        .offset(offset)
        .limit(page.limit)
    )

    items: list[HighRiskPost] = []
    for post, score, bot_score, narrative_id, narrative_title in (
        await session.execute(stmt)
    ).all():
        # The reasons are assembled here, not in the frontend, because only this
        # query knows which signals were actually available for this row.
        reasons = [f"misinfo_likelihood={score.misinfo_likelihood:.2f}"]
        if score.is_toxic and score.toxicity is not None:
            reasons.append(f"toxicity={score.toxicity:.2f}")
        if score.is_anomalous and score.anomaly is not None:
            reasons.append(f"anomaly={score.anomaly:.2f}")
        if bot_score is not None and bot_score > 0.6:
            reasons.append(f"author bot_prob={bot_score:.2f}")
        items.append(
            HighRiskPost(
                post=to_post_out(post, score),
                risk_score=round(100 * score.misinfo_likelihood, 1),
                reasons=reasons,
                narrative_id=str(narrative_id) if narrative_id else None,
                narrative_title=narrative_title,
            )
        )

    return PageResponse[HighRiskPost](
        items=items,
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(items) == page.limit else None
        ),
        total=None,
        filters_applied=spec.as_response_dict(),
    )


async def emotions_breakdown(session: AsyncSession, spec: FilterSpec):
    """Emotion distribution over posts the emotion model actually scored.

    Percentages are over `total_scored`, not over the matched posts. A post the
    model skipped is counted in `unscored` rather than folded into `neutral`,
    which would be the convenient lie: neutral is a prediction, not a default.
    """
    from app.models.corpus import PostScore
    from app.schemas.overview import EmotionsBreakdown, EmotionShare

    project_id = await resolve_project_id(session, spec.project_id)
    ids = _filtered_post_ids(spec, project_id).subquery()

    rows = dict(
        (
            await session.execute(
                select(PostScore.emotion, func.count())
                .where(PostScore.post_id.in_(select(ids.c.id)), PostScore.emotion.isnot(None))
                .group_by(PostScore.emotion)
            )
        ).all()
    )
    matched = (await session.execute(select(func.count()).select_from(ids))).scalar() or 0

    # Phase 2's vocabulary says `joy`; the UI spec says `happiness`. Translated
    # here, once, rather than on the client where two clients would disagree.
    counts: dict[str, int] = {}
    for label, count in rows.items():
        counts[("happiness" if label == "joy" else label)] = (
            counts.get("happiness" if label == "joy" else label, 0) + count
        )

    scored = sum(counts.values())
    order = ("fear", "anger", "disgust", "happiness", "surprise", "sadness", "neutral")
    return EmotionsBreakdown(
        items=[
            EmotionShare(
                emotion=name,
                count=counts.get(name, 0),
                pct=round(100 * counts.get(name, 0) / scored, 2) if scored else 0.0,
            )
            for name in order
        ],
        total_scored=scored,
        unscored=max(matched - scored, 0),
        definition=DEFINITIONS["emotions"],
    )


async def sentiment_breakdown(session: AsyncSession, spec: FilterSpec):
    from app.models.corpus import PostScore
    from app.schemas.overview import SentimentBreakdown, SentimentShare

    project_id = await resolve_project_id(session, spec.project_id)
    ids = _filtered_post_ids(spec, project_id).subquery()

    rows = {
        row.sentiment: row
        for row in (
            await session.execute(
                select(
                    PostScore.sentiment,
                    func.count().label("count"),
                    func.avg(PostScore.sentiment_score).label("mean_score"),
                )
                .where(PostScore.post_id.in_(select(ids.c.id)), PostScore.sentiment.isnot(None))
                .group_by(PostScore.sentiment)
            )
        ).all()
    }
    matched = (await session.execute(select(func.count()).select_from(ids))).scalar() or 0

    scored = sum(row.count for row in rows.values())
    items = [
        SentimentShare(
            sentiment=label,
            count=rows[label].count if label in rows else 0,
            pct=round(100 * (rows[label].count if label in rows else 0) / scored, 2)
            if scored
            else 0.0,
            mean_score=(
                round(float(rows[label].mean_score), 3)
                if label in rows and rows[label].mean_score is not None
                else None
            ),
        )
        for label in ("positive", "neutral", "negative")
    ]
    positive = next(i.pct for i in items if i.sentiment == "positive")
    negative = next(i.pct for i in items if i.sentiment == "negative")

    return SentimentBreakdown(
        items=items,
        total_scored=scored,
        unscored=max(matched - scored, 0),
        net_sentiment=round(positive - negative, 2),
        definition=DEFINITIONS["sentiment"],
    )
