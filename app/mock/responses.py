"""Mock corpus -> schema-valid response models.

One function per route. Routers call these behind a single ``if demo_mode()``
guard, so flipping a route onto real data at its build step is a one-line
deletion rather than an untangling.

Every function here returns the *same Pydantic model* the real implementation
returns. That is the entire point of the contract tag: after it, only values
change.
"""

from __future__ import annotations

import base64
import json
from datetime import timedelta
from typing import Any, TypeVar

from app.config import get_api_settings
from app.deps import FilterSpec, Page
from app.mock.corpus import DEMO_NOW, DEMO_PROJECT_SLUG, MockNarrative, MockPost, corpus
from app.schemas.common import AiProvenance, GeneratedText, PageResponse, ScoreComponent

T = TypeVar("T")


def demo_mode() -> bool:
    return get_api_settings().demo_mode


# ---------------------------------------------------------------------------
# cursor helpers
# ---------------------------------------------------------------------------
def encode_cursor(offset: int) -> str:
    """Opaque, so nobody builds a frontend that does arithmetic on it.

    The real implementation encodes a sort key rather than an offset; the
    *opacity* is what has to match, because a frontend that learned to increment
    an integer here would break the day the real cursor lands.
    """
    return base64.urlsafe_b64encode(json.dumps({"o": offset}).encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        return int(json.loads(base64.urlsafe_b64decode(padded))["o"])
    except Exception:
        return 0


def paginate(items: list[T], page: Page, filters: dict[str, Any]) -> PageResponse[T]:
    offset = decode_cursor(page.cursor)
    window = items[offset : offset + page.limit]
    has_more = offset + page.limit < len(items)
    return PageResponse[T](
        items=window,
        next_cursor=encode_cursor(offset + page.limit) if has_more else None,
        total=len(items),
        filters_applied=filters,
    )


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------
def apply_post_filters(posts: list[MockPost], spec: FilterSpec) -> list[MockPost]:
    """A faithful-enough subset of the real filter semantics.

    Not every predicate is implemented -- these are fixtures -- but the ones that
    change row *counts* are, so a frontend developer wiring the filter bar sees
    the numbers move in the right direction.
    """
    out = posts
    if spec.date_from:
        out = [p for p in out if p.timestamp >= spec.date_from]
    if spec.date_to:
        out = [p for p in out if p.timestamp < spec.date_to]
    if spec.platform:
        out = [p for p in out if p.source in spec.platform]
    if spec.narrative_id:
        wanted = set(spec.narrative_id)
        out = [p for p in out if wanted & set(p.narrative_ids)]
    if spec.content_type:
        out = [p for p in out if p.content_type in spec.content_type]
    if spec.lang:
        out = [p for p in out if p.lang in spec.lang]
    if spec.sentiment:
        out = [p for p in out if (p.scores or {}).get("sentiment") in spec.sentiment]
    if spec.emotion:
        out = [p for p in out if (p.scores or {}).get("emotion") in spec.emotion]
    if spec.min_toxicity is not None:
        # A null toxicity is excluded rather than treated as 0. "Not measured"
        # is not "measured as harmless", and a floor of 0.0 must not silently
        # sweep in every post the scorer skipped.
        out = [
            p
            for p in out
            if (p.scores or {}).get("toxicity") is not None
            and p.scores["toxicity"] >= spec.min_toxicity
        ]
    if spec.is_anomalous is not None:
        out = [p for p in out if (p.scores or {}).get("is_anomalous") is spec.is_anomalous]
    if spec.q:
        needle = spec.q.lower()
        out = [p for p in out if needle in p.text.lower()]
    if not spec.include_shared:
        out = [p for p in out if p.parent_id is None]
    return out


# ---------------------------------------------------------------------------
# converters
# ---------------------------------------------------------------------------
def to_post(post: MockPost, *, include_raw: bool = False):
    from app.schemas.posts import Engagement, PostOut, PostScores

    scores = None
    if post.scores:
        scores = PostScores(
            **{k: v for k, v in post.scores.items() if k in PostScores.model_fields}
        )
    return PostOut(
        id=post.id,
        project_id=corpus().project_id,
        native_id=post.native_id,
        source=post.source,
        source_detail=post.source_detail,
        content_type=post.content_type,
        text=post.text,
        lang=post.lang,
        author_id=post.author_id,
        author_handle=post.author_handle,
        timestamp=post.timestamp,
        parent_id=post.parent_id,
        conversation_id=post.conversation_id,
        engagement=Engagement(
            likes=post.likes, shares=post.shares, replies=post.replies, views=post.views
        ),
        urls=post.urls,
        domains=post.domains,
        media_urls=post.media_urls,
        hashtags=post.hashtags,
        mentions=post.mentions,
        scores=scores,
        narrative_ids=post.narrative_ids,
        raw={"demo": True, "note": "raw payloads are omitted in DEMO_MODE"}
        if include_raw
        else None,
    )


def to_scorecard(narrative: MockNarrative):
    from app.schemas.narratives import NarrativeScorecard

    return NarrativeScorecard(**narrative.scorecard)


def _provenance(kind: str, *, edited: bool = False) -> AiProvenance:
    settings = get_api_settings()
    if kind == "human":
        return AiProvenance(generated_by="human", edited_by_user=edited)
    return AiProvenance(
        generated_by="ai",
        model=settings.llm_model,
        generated_at=DEMO_NOW - timedelta(hours=4),
        prompt_version="v1",
        edited_by_user=edited,
    )


def to_narrative_summary(narrative: MockNarrative):
    from app.schemas.narratives import NarrativeSummary

    return NarrativeSummary(
        id=narrative.id,
        display_id=narrative.display_id,
        project_id=corpus().project_id,
        title=GeneratedText(
            text=narrative.title,
            provenance=_provenance(narrative.title_generated_by, edited=narrative.edited_by_user),
        ),
        summary=GeneratedText(
            text=narrative.summary,
            provenance=_provenance(narrative.summary_generated_by, edited=narrative.edited_by_user),
        ),
        claim=narrative.claim,
        scorecard=to_scorecard(narrative),
        date_start=narrative.date_start,
        date_end=narrative.date_end,
        post_count=len(narrative.post_ids),
        author_count=len(narrative.author_ids),
        engagement_total=narrative.engagement_total,
        platforms=narrative.platforms,
        top_domains=narrative.top_domains[:8],
        top_hashtags=narrative.top_hashtags[:8],
        is_manual=narrative.is_manual,
        cluster_id=narrative.cluster_id,
        clustering_run_id=None if narrative.is_manual else _clustering_run_id(),
        updated_at=DEMO_NOW - timedelta(hours=3),
    )


def to_narrative_detail(narrative: MockNarrative):
    from app.schemas.narratives import NarrativeDetail, RepresentativePost

    base = to_narrative_summary(narrative).model_dump()
    members = [corpus().by_post[pid] for pid in narrative.post_ids[:5]]
    return NarrativeDetail(
        **base,
        representative_posts=[
            RepresentativePost(
                post_id=m.id,
                text=m.text,
                source=m.source,
                author_handle=m.author_handle,
                timestamp=m.timestamp,
                membership_score=round(0.95 - 0.06 * index, 3),
            )
            for index, m in enumerate(members)
        ],
        coherence=narrative.coherence,
        velocity=narrative.velocity,
        compass_available=narrative.has_compass,
    )


def to_author(author):
    from app.schemas.actors import AuthorOut, CohortMembership

    by_cohort = corpus().by_cohort
    return AuthorOut(
        author_id=author.author_id,
        project_id=corpus().project_id,
        source=author.source,
        handle=author.handle,
        created_at_source=author.created_at_source,
        followers=author.followers,
        following=author.following,
        post_count=author.post_count,
        first_seen=author.first_seen,
        last_seen=author.last_seen,
        is_bot_flagged=author.is_bot_flagged,
        bot_score=author.bot_score,
        risk_score=author.risk_score,
        partisan_lean=author.partisan_lean,
        dominant_sentiment=author.dominant_sentiment,
        dominant_emotion=author.dominant_emotion,
        anomalous_score=author.anomalous_score,
        toxicity_score=author.toxicity_score,
        coordination_score=author.coordination_score,
        community_id=author.community_id,
        community_size=author.community_size,
        cohorts=[
            CohortMembership(
                cohort_id=cid,
                name=by_cohort[cid]["name"],
                category=by_cohort[cid]["category"],
                confidence=0.82,
                assigned_by="model",
            )
            for cid in author.cohort_ids
        ],
        narratives_touched=author.narrative_ids,
        scoring_version=None if author.skip_reasons else "phase2-v0.1.0",
        skip_reasons=author.skip_reasons,
    )


def to_cohort(cohort: dict[str, Any]):
    from app.schemas.actors import CohortOut

    total_posts = len(corpus().posts) or 1
    return CohortOut(
        id=cohort["id"],
        project_id=corpus().project_id,
        name=cohort["name"],
        description=cohort["description"],
        category=cohort["category"],
        author_count=cohort["author_count"],
        post_pct=round(100 * cohort["post_count"] / total_posts, 2),
        mean_bot_score=0.31,
        mean_toxicity=0.18,
    )


def to_domain(domain: dict[str, Any]):
    from app.schemas.domains import DomainOut

    whois = domain["whois_created_at"]
    return DomainOut(
        domain=domain["domain"],
        project_id=corpus().project_id,
        risk_score=round(domain["risk_score"], 1),
        risk_band=domain["risk_band"],
        first_seen=domain["first_seen"],
        last_seen=domain["last_seen"],
        post_count=domain["post_count"],
        author_count=domain["author_count"],
        narrative_count=domain["narrative_count"],
        whois_created_at=whois,
        domain_age_days=(DEMO_NOW - whois).days if whois else None,
        hosting_country=domain["hosting_country"],
        tls_cert_age_days=domain["tls_cert_age_days"],
        registrar=domain["registrar"],
        enrichment_status=domain["enrichment_status"],
        enriched_at=domain["enriched_at"],
        enrichment_detail=domain["enrichment_detail"],
        link_velocity=domain["link_velocity"],
        sharing_author_bot_ratio=domain["sharing_author_bot_ratio"],
        scoring_version="domain-v1.0.0",
        components=[
            ScoreComponent(
                name="link_velocity",
                value=min(1.0, domain["link_velocity"] / 25),
                weight=0.35,
                contribution=round(min(1.0, domain["link_velocity"] / 25) * 0.35, 4),
                definition=(
                    "Peak posts per hour carrying this domain, percentile-ranked in project."
                ),
                inputs={"peak_per_hour": domain["link_velocity"]},
            ),
            ScoreComponent(
                name="sharing_author_bot_ratio",
                value=domain["sharing_author_bot_ratio"],
                weight=0.4,
                contribution=round(domain["sharing_author_bot_ratio"] * 0.4, 4),
                definition="Share of linking authors scoring above the bot threshold.",
                inputs={"linking_authors": domain["author_count"]},
            ),
            ScoreComponent(
                name="domain_age",
                value=(
                    None if not whois else min(1.0, max(0.0, 1 - (DEMO_NOW - whois).days / 3650))
                ),
                weight=0.25,
                contribution=(
                    None
                    if not whois
                    else round(min(1.0, max(0.0, 1 - (DEMO_NOW - whois).days / 3650)) * 0.25, 4)
                ),
                definition=(
                    "Newer registrations score higher. Null when WHOIS is unavailable; the "
                    "remaining weights renormalize rather than treating the age as zero."
                ),
                inputs={"whois_created_at": whois.isoformat() if whois else None},
            ),
        ],
    )


def _clustering_run_id() -> str:
    from app.mock.corpus import _stable_uuid

    return _stable_uuid("clustering_run", "latest")


# ---------------------------------------------------------------------------
# projects & setup
# ---------------------------------------------------------------------------
def projects_list():
    from app.schemas.projects import ProjectOut, ProjectStats

    c = corpus()
    return [
        ProjectOut(
            id=c.project_id,
            slug=DEMO_PROJECT_SLUG,
            name="Election Integrity 2026 (demo)",
            description=(
                "Fixture project served under DEMO_MODE. Every id here is stable across "
                "restarts so a bookmarked narrative keeps working."
            ),
            date_start=min(p.timestamp for p in c.posts),
            date_end=max(p.timestamp for p in c.posts),
            seed_config={
                "topics": ["election fraud claim", "voter fraud claim", "ballot fraud"],
                "sources": list({p.source for p in c.posts}),
            },
            created_at=DEMO_NOW - timedelta(days=62),
            updated_at=DEMO_NOW - timedelta(hours=3),
            stats=ProjectStats(
                post_count=len(c.posts),
                author_count=len(c.authors),
                narrative_count=len(c.narratives),
                domain_count=len(c.domains),
                first_post_at=min(p.timestamp for p in c.posts),
                last_post_at=max(p.timestamp for p in c.posts),
            ),
        )
    ]


def project_sources():
    from app.schemas.projects import SourceHealth

    c = corpus()
    counts: dict[str, int] = {}
    for post in c.posts:
        counts[post.source] = counts.get(post.source, 0) + 1

    # Two sources are deliberately unhealthy: one never configured, one out of
    # quota. A setup page where every light is green teaches nobody how the
    # unhappy paths render.
    rows = []
    for source in ("reddit", "mastodon", "news", "gdelt", "youtube"):
        if source == "youtube":
            rows.append(
                SourceHealth(
                    source=source,
                    configured=True,
                    status="quota_exhausted",
                    detail=(
                        "Daily quota of 10,000 units consumed. Resets at 00:00 UTC. "
                        "Ingestion resumes from the stored checkpoint."
                    ),
                    last_sync_at=DEMO_NOW - timedelta(hours=14),
                    record_count=counts.get(source, 0),
                    checkpoint={"last_published_at": (DEMO_NOW - timedelta(hours=14)).isoformat()},
                    quota_used=10_000,
                    quota_limit=10_000,
                    quota_resets_at=DEMO_NOW.replace(hour=0, minute=0) + timedelta(days=1),
                )
            )
        elif source == "news":
            rows.append(
                SourceHealth(
                    source=source,
                    configured=False,
                    status="skipped",
                    detail=(
                        "NEWSAPI_KEY is unset, so only the RSS adapter ran. This is a "
                        "configuration choice, not a failure."
                    ),
                    last_sync_at=DEMO_NOW - timedelta(hours=5),
                    record_count=counts.get(source, 0),
                )
            )
        else:
            rows.append(
                SourceHealth(
                    source=source,
                    configured=True,
                    status="ok",
                    detail=f"Last run completed cleanly; {counts.get(source, 0)} records on disk.",
                    last_sync_at=DEMO_NOW - timedelta(hours=3),
                    record_count=counts.get(source, 0),
                    checkpoint={"last_id": f"{source}:001399"},
                )
            )
    return rows


def source_test(source: str):
    from datetime import datetime, timezone

    from app.schemas.projects import SourceTestResult

    return SourceTestResult(
        source=source,  # type: ignore[arg-type]
        reachable=True,
        authenticated=source not in {"news"},
        detail=(
            "DEMO_MODE: no network call was made. In a live deployment this probes the "
            "Phase 1 adapter's credentials and reports what it found."
        ),
        latency_ms=42.0,
        checked_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------
#: Definition text for every KPI. Written once, here, because the definition is
#: a property of the query and has to change with it.
DEFINITIONS = {
    "posts": (
        "Count of posts matching the active filters. Includes comments and replies "
        "unless the 'Original data' chip is on."
    ),
    "engagements": (
        "Sum of likes, shares and replies on matched posts. Excludes views: views "
        "measure exposure, not engagement, and summing them makes a video-heavy "
        "corpus look an order of magnitude more engaged than it is. Posts where a "
        "platform does not expose a metric contribute nothing rather than zero."
    ),
    "authors": (
        "Distinct author ids among matched posts. Authors are namespaced by source, "
        "so the same person on two platforms counts twice -- cross-platform identity "
        "resolution is out of scope."
    ),
    "sentiment": (
        "Net sentiment: the share of scored posts classified positive minus the "
        "share classified negative, in [-100, 100]. Positive means the corpus "
        "leans positive. Posts the sentiment model skipped are excluded from both "
        "the numerator and the denominator, so this is a share of what was "
        "measured rather than of what was collected."
    ),
    "emotions": (
        "The most frequent dominant emotion among scored posts. Each post "
        "contributes its single highest-scoring emotion, not its full distribution."
    ),
    "timeseries": (
        "Bucketed by the requested interval in the requested timezone. Buckets with "
        "no matching posts are emitted as zero rather than omitted, so the axis is "
        "continuous and a gap reads as a gap."
    ),
    "concepts": (
        "Hashtags, named entities and high-TF-IDF keywords over matched post text, "
        "with the mean sentiment of the posts containing them."
    ),
    "drivers_authors": (
        "Authors ranked by matched post count. bot_like_share is the fraction of "
        "this author's amplifiers scoring above the bot threshold."
    ),
    "drivers_hashtags": "Hashtags ranked by matched post count, with distinct-author reach.",
    "drivers_urls": "URLs ranked by share count, rolled up to the registrable domain.",
}


def _metric(name: str, value, label: str, **kwargs):
    from app.schemas.common import Metric

    return Metric(value=value, label=label, definition=DEFINITIONS[name], **kwargs)


def overview_kpis(spec: FilterSpec):
    from app.schemas.overview import KpiBundle

    posts = apply_post_filters(corpus().posts, spec)
    scored = [p for p in posts if p.scores and p.scores.get("sentiment")]
    engagement = sum(v for p in posts for v in (p.likes, p.shares, p.replies) if v is not None)
    negative = sum(1 for p in scored if p.scores["sentiment"] == "negative")
    positive = sum(1 for p in scored if p.scores["sentiment"] == "positive")
    # positive - negative, matching the breakdown endpoint exactly. These two
    # numbers appear on the same screen; opposite sign conventions between them
    # is the kind of disagreement the shared definitions exist to prevent.
    net = round(100 * (positive - negative) / len(scored), 1) if scored else None

    emotion_counts: dict[str, int] = {}
    for post in posts:
        emotion = (post.scores or {}).get("emotion")
        if emotion:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None

    return KpiBundle(
        posts=_metric("posts", len(posts), "Posts", sample_size=len(posts), delta_pct=12.4),
        engagements=_metric(
            "engagements",
            engagement,
            "Engagements",
            nullable_fields=["views", "likes", "shares", "replies"],
            sample_size=len(posts),
            delta_pct=-3.1,
        ),
        authors=_metric(
            "authors",
            len({p.author_id for p in posts}),
            "Authors",
            sample_size=len(posts),
            delta_pct=8.0,
        ),
        sentiment=_metric("sentiment", net, "Net sentiment", unit="pct", sample_size=len(scored)),
        emotions=_metric(
            "emotions",
            emotion_counts.get(dominant) if dominant else None,
            f"Top emotion: {dominant}" if dominant else "Top emotion",
            sample_size=sum(emotion_counts.values()),
        ),
        window_start=spec.date_from or min((p.timestamp for p in posts), default=None),
        window_end=spec.date_to or max((p.timestamp for p in posts), default=None),
        tz=spec.tz,
        comparison_window="preceding period of equal length",
    )


def overview_timeseries(
    spec: FilterSpec, metric: str, group_by: str, interval: str, normalize: bool, agg: str
):
    from app.schemas.overview import TimeseriesPoint, TimeseriesResponse

    posts = apply_post_filters(corpus().posts, spec)
    hours = {"1h": 1, "6h": 6, "1d": 24, "7d": 168}[interval]
    if not posts:
        return TimeseriesResponse(
            metric=metric,
            interval=interval,
            group_by=group_by,
            agg=agg,
            normalized=normalize,
            tz=spec.tz,
            points=[],
            series_labels={},
            definition=DEFINITIONS["timeseries"],
        )

    start = min(p.timestamp for p in posts).replace(minute=0, second=0, microsecond=0)
    end = max(p.timestamp for p in posts)
    buckets: dict[Any, dict[str, float]] = {}
    cursor = start
    while cursor <= end:
        buckets[cursor] = {}
        cursor += timedelta(hours=hours)

    labels: dict[str, str] = {}
    for post in posts:
        offset = int((post.timestamp - start).total_seconds() // (hours * 3600))
        bucket = start + timedelta(hours=hours * offset)
        if bucket not in buckets:
            continue
        if group_by == "platform":
            key = post.source
            labels[key] = key
        elif group_by == "narrative":
            key = post.narrative_ids[0] if post.narrative_ids else "unassigned"
            labels[key] = (
                corpus().by_narrative[key].title if key in corpus().by_narrative else "Unassigned"
            )
        else:
            key = "all"
            labels[key] = "All"
        value = (
            1
            if metric == "posts"
            else sum(v for v in (post.likes, post.shares, post.replies) if v is not None)
        )
        buckets[bucket][key] = buckets[bucket].get(key, 0) + value

    if normalize:
        # Each series scaled to its own peak. Shape becomes comparable and
        # magnitude does not; the response says `normalized: true` so the UI can
        # label the axis and nobody reads a normalized 1.0 as a raw count.
        peaks: dict[str, float] = {}
        for series in buckets.values():
            for key, value in series.items():
                peaks[key] = max(peaks.get(key, 0.0), value)
        for series in buckets.values():
            for key in list(series):
                if peaks.get(key):
                    series[key] = round(series[key] / peaks[key], 4)

    points = [
        TimeseriesPoint(
            bucket_start=bucket,
            value=round(sum(series.values()), 4),
            series=series if group_by != "none" else None,
        )
        for bucket, series in sorted(buckets.items())
    ]
    return TimeseriesResponse(
        metric=metric,
        interval=interval,
        group_by=group_by,
        agg=agg,
        normalized=normalize,
        tz=spec.tz,
        points=points,
        series_labels=labels,
        definition=DEFINITIONS["timeseries"],
    )


def overview_concepts(spec: FilterSpec):
    from app.schemas.overview import ConceptItem, ConceptsResponse

    posts = apply_post_filters(corpus().posts, spec)
    tallies: dict[tuple[str, str], dict[str, Any]] = {}
    for post in posts:
        terms = [(tag, "hashtag") for tag in post.hashtags] + [
            (domain, "domain") for domain in post.domains
        ]
        for term, kind in terms:
            entry = tallies.setdefault((term, kind), {"volume": 0, "sent": [], "narratives": set()})
            entry["volume"] += 1
            score = (post.scores or {}).get("sentiment_score")
            if score is not None:
                entry["sent"].append(score)
            entry["narratives"].update(post.narrative_ids)

    items = [
        ConceptItem(
            term=term,
            kind=kind,
            volume=data["volume"],
            sentiment_score=(
                round(sum(data["sent"]) / len(data["sent"]), 3) if data["sent"] else None
            ),
            narrative_ids=sorted(data["narratives"])[:5],
        )
        for (term, kind), data in tallies.items()
    ]
    items.sort(key=lambda item: item.volume, reverse=True)
    return ConceptsResponse(
        items=items[:60], total_terms=len(items), definition=DEFINITIONS["concepts"]
    )


def overview_high_risk_posts(spec: FilterSpec, page: Page):
    from app.schemas.posts import HighRiskPost

    posts = apply_post_filters(corpus().posts, spec)
    scored = [p for p in posts if p.scores and p.scores.get("misinfo_likelihood") is not None]
    scored.sort(key=lambda p: p.scores["misinfo_likelihood"], reverse=True)

    items = []
    for post in scored[:400]:
        author = corpus().by_author[post.author_id]
        reasons = [f"misinfo_likelihood={post.scores['misinfo_likelihood']:.2f}"]
        if post.scores.get("is_toxic"):
            reasons.append(f"toxicity={post.scores['toxicity']:.2f}")
        if author.bot_score and author.bot_score > 0.6:
            reasons.append(f"author bot_prob={author.bot_score:.2f}")
        narrative = corpus().by_narrative.get(post.narrative_ids[0]) if post.narrative_ids else None
        items.append(
            HighRiskPost(
                post=to_post(post),
                risk_score=round(100 * post.scores["misinfo_likelihood"], 1),
                reasons=reasons,
                narrative_id=narrative.id if narrative else None,
                narrative_title=narrative.title if narrative else None,
            )
        )
    return paginate(items, page, spec.as_response_dict())


def overview_posts_breakdown(spec: FilterSpec):
    from app.schemas.overview import HierarchyNode, PostsBreakdown

    posts = apply_post_filters(corpus().posts, spec)
    total = len(posts) or 1
    original = [p for p in posts if p.parent_id is None]
    anonymous = [p for p in posts if p.author_handle is None or p.author_id.endswith("__deleted__")]
    anomalous = [p for p in posts if (p.scores or {}).get("is_anomalous")]
    toxic = [p for p in posts if (p.scores or {}).get("is_toxic")]

    by_domain: dict[str, int] = {}
    by_channel: dict[str, dict[str, int]] = {}
    for post in posts:
        for domain in post.domains:
            by_domain[domain] = by_domain.get(domain, 0) + 1
        by_channel.setdefault(post.source, {})
        by_channel[post.source][post.source_detail] = (
            by_channel[post.source].get(post.source_detail, 0) + 1
        )

    return PostsBreakdown(
        total=_metric("posts", len(posts), "Total posts", sample_size=len(posts)),
        original=_metric("posts", len(original), "Original posts", sample_size=len(posts)),
        local_shared=_metric(
            "posts", len(posts) - len(original), "Shared / replies", sample_size=len(posts)
        ),
        anonymous=_metric("posts", len(anonymous), "Anonymous authors", sample_size=len(posts)),
        anomalous=_metric("posts", len(anomalous), "Anomalous", sample_size=len(posts)),
        toxic=_metric("posts", len(toxic), "Toxic", sample_size=len(posts)),
        by_domain=[
            HierarchyNode(key=d, label=d, value=v, pct=round(100 * v / total, 2))
            for d, v in sorted(by_domain.items(), key=lambda kv: -kv[1])[:12]
        ],
        by_channel=[
            HierarchyNode(
                key=source,
                label=source,
                value=sum(children.values()),
                pct=round(100 * sum(children.values()) / total, 2),
                children=[
                    HierarchyNode(
                        key=name, label=name, value=count, pct=round(100 * count / total, 2)
                    )
                    for name, count in sorted(children.items(), key=lambda kv: -kv[1])[:8]
                ],
            )
            for source, children in sorted(by_channel.items(), key=lambda kv: -sum(kv[1].values()))
        ],
    )


def overview_engagements_breakdown(spec: FilterSpec):
    from app.schemas.overview import EngagementsBreakdown

    posts = apply_post_filters(corpus().posts, spec)
    high_risk = {
        p.id
        for p in posts
        if (p.scores or {}).get("misinfo_likelihood") is not None
        and p.scores["misinfo_likelihood"] > 0.7
    }

    def total(attr: str, subset=None) -> int:
        rows = subset if subset is not None else posts
        return sum(getattr(p, attr) or 0 for p in rows)

    on_high_risk = [p for p in posts if p.id in high_risk]
    return EngagementsBreakdown(
        total=_metric(
            "engagements",
            total("likes") + total("shares") + total("replies"),
            "Total engagements",
            nullable_fields=["views"],
            sample_size=len(posts),
        ),
        on_high_risk=_metric(
            "engagements",
            total("likes", on_high_risk) + total("shares", on_high_risk),
            "On high-risk posts",
            sample_size=len(on_high_risk),
        ),
        likes=_metric("engagements", total("likes"), "Likes", sample_size=len(posts)),
        global_shares=_metric("engagements", total("shares"), "Shares", sample_size=len(posts)),
        reactions=_metric("engagements", total("replies"), "Replies", sample_size=len(posts)),
        views=_metric(
            "engagements",
            total("views"),
            "Views",
            nullable_fields=["views"],
            sample_size=sum(1 for p in posts if p.views is not None),
        ),
    )


def overview_authors_breakdown(spec: FilterSpec):
    from app.schemas.overview import AuthorsBreakdown, HierarchyNode

    posts = apply_post_filters(corpus().posts, spec)
    author_ids = {p.author_id for p in posts}
    authors = [a for a in corpus().authors if a.author_id in author_ids]
    bot_like = [a for a in authors if a.bot_score is not None and a.bot_score > 0.6]

    cohort_counts: dict[str, int] = {}
    for author in authors:
        for cohort_id in author.cohort_ids:
            cohort_counts[cohort_id] = cohort_counts.get(cohort_id, 0) + 1

    total = len(authors) or 1
    return AuthorsBreakdown(
        total=_metric("authors", len(authors), "Total authors", sample_size=len(authors)),
        bot_like=_metric(
            "authors",
            len(bot_like),
            "Bot-like authors",
            sample_size=sum(1 for a in authors if a.bot_score is not None),
            nullable_fields=["bot_score"],
        ),
        author_groups=_metric("authors", 2, "Watchlisted authors", sample_size=len(authors)),
        top_cohorts=[
            HierarchyNode(
                key=cid,
                label=corpus().by_cohort[cid]["name"],
                value=count,
                pct=round(100 * count / total, 2),
            )
            for cid, count in sorted(cohort_counts.items(), key=lambda kv: -kv[1])[:6]
        ],
        total_cohort_count=len(cohort_counts),
    )


def overview_emotions_breakdown(spec: FilterSpec):
    from app.schemas.overview import EmotionsBreakdown, EmotionShare

    posts = apply_post_filters(corpus().posts, spec)
    counts: dict[str, int] = {}
    unscored = 0
    for post in posts:
        emotion = (post.scores or {}).get("emotion")
        if emotion is None:
            unscored += 1
        else:
            # The corpus's emotion vocabulary uses 'joy'; the UI spec's does not.
            # Mapped in one place rather than in the frontend, because a label
            # translated on the client is a label two clients disagree about.
            emotion = "happiness" if emotion == "joy" else emotion
            counts[emotion] = counts.get(emotion, 0) + 1

    scored = sum(counts.values()) or 1
    order = ("fear", "anger", "disgust", "happiness", "surprise", "sadness", "neutral")
    return EmotionsBreakdown(
        items=[
            EmotionShare(
                emotion=name,
                count=counts.get(name, 0),
                pct=round(100 * counts.get(name, 0) / scored, 2),
            )
            for name in order
        ],
        total_scored=sum(counts.values()),
        unscored=unscored,
        definition=DEFINITIONS["emotions"],
    )


def overview_sentiment_breakdown(spec: FilterSpec):
    from app.schemas.overview import SentimentBreakdown, SentimentShare

    posts = apply_post_filters(corpus().posts, spec)
    counts: dict[str, list[float]] = {"positive": [], "neutral": [], "negative": []}
    unscored = 0
    for post in posts:
        label = (post.scores or {}).get("sentiment")
        if label is None:
            unscored += 1
        else:
            counts[label].append((post.scores or {}).get("sentiment_score") or 0.0)

    scored = sum(len(v) for v in counts.values()) or 1
    items = [
        SentimentShare(
            sentiment=label,
            count=len(scores),
            pct=round(100 * len(scores) / scored, 2),
            mean_score=round(sum(scores) / len(scores), 3) if scores else None,
        )
        for label, scores in counts.items()
    ]
    negative = next(i.pct for i in items if i.sentiment == "negative")
    positive = next(i.pct for i in items if i.sentiment == "positive")
    return SentimentBreakdown(
        items=items,
        total_scored=sum(len(v) for v in counts.values()),
        unscored=unscored,
        net_sentiment=round(positive - negative, 2),
        definition=DEFINITIONS["sentiment"],
    )


# ---------------------------------------------------------------------------
# narratives
# ---------------------------------------------------------------------------
def narratives_page(spec: FilterSpec, page: Page, sort: str, q: str | None):
    narratives = list(corpus().narratives)
    if q:
        needle = q.lower()
        narratives = [
            n
            for n in narratives
            if needle in n.title.lower() or needle in n.summary.lower() or needle in n.claim.lower()
        ]
    if spec.platform:
        narratives = [n for n in narratives if set(spec.platform) & set(n.platforms)]
    if spec.min_risk is not None:
        narratives = [n for n in narratives if (n.scorecard["fusion_score"] or 0) >= spec.min_risk]

    order = {"high": 0, "medium": 1, "low": 2}
    sorters = {
        "priority": lambda n: (order[n.scorecard["priority"]], -(n.scorecard["fusion_score"] or 0)),
        "date": lambda n: -n.date_end.timestamp(),
        "engagement": lambda n: -n.engagement_total,
        "posts": lambda n: -len(n.post_ids),
    }
    narratives.sort(key=sorters.get(sort, sorters["priority"]))
    return paginate(
        [to_narrative_summary(n) for n in narratives],
        page,
        {**spec.as_response_dict(), "sort": sort, "q": q},
    )


def narrative_timeline(narrative: MockNarrative, interval: str, tz: str, cross_platform: bool):
    from app.schemas.narratives import NarrativeTimeline, TimelinePoint

    posts = [corpus().by_post[pid] for pid in narrative.post_ids]
    hours = {"1h": 1, "6h": 6, "1d": 24, "7d": 168}[interval]
    if not posts:
        return NarrativeTimeline(
            narrative_id=narrative.id, interval=interval, tz=tz, points=[], bursts=[]
        )

    start = min(p.timestamp for p in posts).replace(minute=0, second=0, microsecond=0)
    end = max(p.timestamp for p in posts)
    buckets: dict[Any, dict[str, Any]] = {}
    cursor = start
    while cursor <= end:
        buckets[cursor] = {"posts": 0, "engagement": 0, "authors": set(), "platforms": {}}
        cursor += timedelta(hours=hours)

    for post in posts:
        offset = int((post.timestamp - start).total_seconds() // (hours * 3600))
        bucket = buckets.get(start + timedelta(hours=hours * offset))
        if bucket is None:
            continue
        bucket["posts"] += 1
        bucket["engagement"] += sum(
            v for v in (post.likes, post.shares, post.replies) if v is not None
        )
        bucket["authors"].add(post.author_id)
        bucket["platforms"][post.source] = bucket["platforms"].get(post.source, 0) + 1

    points = [
        TimelinePoint(
            bucket_start=bucket,
            post_count=data["posts"],
            engagement=data["engagement"],
            author_count=len(data["authors"]),
            by_platform=data["platforms"] if cross_platform else None,
        )
        for bucket, data in sorted(buckets.items())
    ]
    peak = max(points, key=lambda p: p.post_count) if points else None
    # A burst is what coordination looks like from the outside: a bucket well
    # above the series mean. Flagged server-side because a six-hour spike on a
    # sixty-day axis is easy for the eye to miss and is the signal that matters.
    mean = sum(p.post_count for p in points) / len(points) if points else 0
    bursts = [
        {
            "bucket_start": p.bucket_start.isoformat(),
            "post_count": p.post_count,
            "times_mean": round(p.post_count / mean, 2) if mean else None,
        }
        for p in points
        if mean and p.post_count > 3 * mean
    ]
    return NarrativeTimeline(
        narrative_id=narrative.id,
        interval=interval,
        tz=tz,
        points=points,
        peak_at=peak.bucket_start if peak else None,
        bursts=bursts,
    )


def narrative_score(narrative: MockNarrative):
    from app.schemas.narratives import NarrativeScoreExplanation

    card = narrative.scorecard
    missing = list(card["missing"])
    severity = round((card["toxicity"] or 0.3) * 0.5 + (card["negative_sentiment"] or 0.4) * 0.5, 4)
    coordination = round((card["bot_like"] or 0.2) * 0.6 + (card["anomalous"] or 0.3) * 0.4, 4)
    authenticity = None if "authenticity" in missing else round(0.18 + 0.4 * narrative.coherence, 4)

    raw_weights = {"narrative_severity": 0.45, "coordination": 0.35, "authenticity": 0.20}
    available = {
        "narrative_severity": severity,
        "coordination": coordination,
        "authenticity": authenticity,
    }
    live = {k: w for k, w in raw_weights.items() if available[k] is not None}
    scale = sum(live.values()) or 1.0
    renormalized = len(live) != len(raw_weights)

    components = [
        ScoreComponent(
            name=name,
            value=available[name],
            weight=round(raw_weights[name] / scale, 4) if name in live else raw_weights[name],
            contribution=(
                round(available[name] * raw_weights[name] / scale, 4)
                if available[name] is not None
                else None
            ),
            definition=definition,
            inputs=inputs,
        )
        for name, definition, inputs in (
            (
                "narrative_severity",
                "f(misinfo_likelihood_agg, compass_risk, toxicity, negative_sentiment)",
                {
                    "toxicity": card["toxicity"],
                    "negative_sentiment": card["negative_sentiment"],
                    "compass_risk": card["compass_risk"],
                },
            ),
            (
                "coordination",
                (
                    "g(bot_like_ratio, co_post_similarity_density, temporal_burstiness, "
                    "cohort_concentration)"
                ),
                {"bot_like_ratio": card["bot_like"], "anomalous": card["anomalous"]},
            ),
            (
                "authenticity",
                "h(deepfake_hits, domain_risk_agg, anonymous_author_ratio)",
                (
                    {"reason": "deepfake module did not run over this narrative's media"}
                    if authenticity is None
                    else {"coherence": narrative.coherence}
                ),
            ),
        )
    ]

    return NarrativeScoreExplanation(
        narrative_id=narrative.id,
        fusion_score=card["fusion_score"],
        priority=card["priority"],
        scoring_version=card["scoring_version"],
        formula="100 * (w1*narrative_severity + w2*coordination + w3*authenticity)",
        normalization=(
            "percentile-rank within project. Chosen over min-max against a fixed reference "
            "range because a project's absolute score distribution shifts with its topic, "
            "and a percentile keeps 'high' meaning the same thing across cases."
        ),
        components=components,
        missing=missing,
        weights_renormalized=renormalized,
        computed_at=card["computed_at"],
    )


def narrative_cohorts(narrative: MockNarrative):
    from app.schemas.narratives import NarrativeCohortShare

    counts: dict[str, dict[str, Any]] = {}
    for author_id in narrative.author_ids:
        author = corpus().by_author[author_id]
        posts_here = sum(
            1 for pid in narrative.post_ids if corpus().by_post[pid].author_id == author_id
        )
        for cohort_id in author.cohort_ids:
            entry = counts.setdefault(cohort_id, {"authors": 0, "posts": 0})
            entry["authors"] += 1
            entry["posts"] += posts_here

    total = len(narrative.post_ids) or 1
    rows = [
        NarrativeCohortShare(
            cohort_id=cid,
            name=corpus().by_cohort[cid]["name"],
            category=corpus().by_cohort[cid]["category"],
            author_count=data["authors"],
            post_count=data["posts"],
            share_pct=round(100 * data["posts"] / total, 2),
        )
        for cid, data in counts.items()
    ]
    rows.sort(key=lambda r: -r.post_count)
    return rows


def narrative_freshness():
    from app.schemas.narratives import NarrativeFreshness

    age = timedelta(days=48, hours=0)
    return NarrativeFreshness(
        project_id=corpus().project_id,
        last_run_id=_clustering_run_id(),
        last_run_at=DEMO_NOW - age,
        age_seconds=age.total_seconds(),
        age_human="48 days, 0 hrs ago",
        status="stale",
        algorithm="hdbscan",
        embedding_model=get_api_settings().embedding_model,
        post_count=len(corpus().posts),
        narrative_count=len(corpus().narratives),
        unclustered_post_count=sum(1 for p in corpus().posts if not p.narrative_ids),
    )


def compass_context(narrative: MockNarrative):
    from app.mock.corpus import _stable_uuid
    from app.schemas.compass import Citation, CompassContext

    if not narrative.has_compass:
        return None

    # One fixture narrative deliberately returns insufficient_evidence with an
    # EMPTY context. That is the required behaviour when citation validation
    # fails twice, and the frontend has to render it as "we could not source
    # this" rather than as an empty card or a spinner.
    unsourceable = narrative.cluster_id == 7
    if unsourceable:
        return CompassContext(
            id=_stable_uuid("compass", narrative.id),
            narrative_id=narrative.id,
            claim=narrative.claim,
            context="",
            verification_status="insufficient_evidence",
            risk="medium",
            caution_note=(
                "No retrievable source addressed this claim directly. The absence of "
                "sources is not evidence that the claim is false."
            ),
            citations=[],
            model=get_api_settings().llm_model,
            prompt_version="v1",
            generated_at=DEMO_NOW - timedelta(hours=5),
            attempts=2,
            retrieved_document_count=3,
        )

    return CompassContext(
        id=_stable_uuid("compass", narrative.id),
        narrative_id=narrative.id,
        claim=narrative.claim,
        context=(
            "Reporting from two national outlets indicates that the tabulator log "
            "circulating with this claim does not identify a county, and the state "
            "election office has not confirmed the described duplication. Analysts "
            "should treat the screenshot as unverified rather than as evidence either "
            "way."
        ),
        verification_status="partially_substantiated",
        risk=narrative.scorecard["compass_risk"],
        caution_note=(
            "This note summarises retrieved reporting. It does not establish that the "
            "claim is false, only that the circulating evidence does not support it as "
            "stated."
        ),
        citations=[
            Citation(
                id=_stable_uuid("citation", f"{narrative.id}:0"),
                url="https://apnews.com/article/demo-election-tabulator-log",
                title="Election office says circulating log lacks county identifier",
                publisher="Associated Press",
                domain="apnews.com",
                retrieved_at=DEMO_NOW - timedelta(hours=5),
                snippet=(
                    "The office said the log excerpt does not identify which county produced it."
                ),
                char_start=0,
                char_end=132,
            ),
            Citation(
                id=_stable_uuid("citation", f"{narrative.id}:1"),
                url="https://reuters.com/world/us/demo-ballot-duplication-claim",
                title="No confirmation of duplicate ballot counting in three counties",
                publisher="Reuters",
                domain="reuters.com",
                retrieved_at=DEMO_NOW - timedelta(hours=5),
                snippet="State officials have not confirmed the duplication described online.",
                char_start=132,
                char_end=245,
            ),
        ],
        model=get_api_settings().llm_model,
        prompt_version="v1",
        generated_at=DEMO_NOW - timedelta(hours=5),
        attempts=1,
        retrieved_document_count=9,
    )


# ---------------------------------------------------------------------------
# actors, drivers, network
# ---------------------------------------------------------------------------
def authors_page(spec: FilterSpec, page: Page, sort: str):
    posts = apply_post_filters(corpus().posts, spec)
    author_ids = {p.author_id for p in posts}
    authors = [a for a in corpus().authors if a.author_id in author_ids]
    if spec.is_bot_like is not None:
        authors = [
            a
            for a in authors
            if a.bot_score is not None and (a.bot_score > 0.6) is spec.is_bot_like
        ]
    if spec.cohort_id:
        wanted = set(spec.cohort_id)
        authors = [a for a in authors if wanted & set(a.cohort_ids)]

    sorters = {
        "bot_score": lambda a: -(a.bot_score or -1),
        "risk": lambda a: -(a.risk_score or -1),
        "posts": lambda a: -a.post_count,
        "followers": lambda a: -(a.followers or -1),
    }
    authors.sort(key=sorters.get(sort, sorters["risk"]))
    return paginate(
        [to_author(a) for a in authors], page, {**spec.as_response_dict(), "sort": sort}
    )


def author_score(author):
    from app.schemas.actors import AuthorScoreExplanation

    if author.bot_score is None:
        return AuthorScoreExplanation(
            author_id=author.author_id,
            bot_score=None,
            risk_score=author.risk_score,
            model="xgboost-bot-clf",
            scoring_version="phase2-v0.1.0",
            top_features=[],
            missing=["bot_score"],
            caveat=(
                "This source's 'authors' are outlets, not accounts. Account-level bot "
                "scoring is skipped with a reason code rather than producing a number."
            ),
        )

    age_days = (DEMO_NOW - author.created_at_source).days if author.created_at_source else None
    features = [
        ("account_age_days", -0.31 if (age_days or 999) < 60 else 0.18, {"value": age_days}),
        ("posts_per_active_day", 0.27, {"value": round(author.post_count / 12, 2)}),
        (
            "follower_following_ratio",
            -0.19,
            {"value": round((author.followers or 1) / max(author.following or 1, 1), 2)},
        ),
        ("posting_hour_entropy", 0.22, {"value": 0.41}),
        ("mean_intra_post_similarity", 0.16, {"value": 0.63}),
    ]
    return AuthorScoreExplanation(
        author_id=author.author_id,
        bot_score=author.bot_score,
        risk_score=author.risk_score,
        model="xgboost-bot-clf",
        scoring_version="phase2-v0.1.0",
        top_features=[
            ScoreComponent(
                name=name,
                value=None,
                weight=abs(contribution),
                contribution=contribution,
                definition=f"SHAP contribution of {name} toward the bot class.",
                inputs=inputs,
            )
            for name, contribution, inputs in features
        ],
        computed_at=DEMO_NOW - timedelta(hours=4),
        caveat=(
            "Bot generalisation to unseen campaigns is weak: per-fold macro-F1 is "
            "0.416 +/- 0.187 against a pooled 0.705. Treat a single high score as a "
            "lead, not a finding."
        ),
    )


def author_timeline(author, interval: str, tz: str):
    from app.schemas.actors import AuthorTimeline, AuthorTimelinePoint

    posts = [p for p in corpus().posts if p.author_id == author.author_id]
    hours = {"1h": 1, "6h": 6, "1d": 24, "7d": 168}[interval]
    histogram: dict[int, int] = {}
    buckets: dict[Any, dict[str, Any]] = {}
    for post in posts:
        histogram[post.timestamp.hour] = histogram.get(post.timestamp.hour, 0) + 1
        bucket = post.timestamp.replace(
            hour=(post.timestamp.hour // hours) * hours if hours < 24 else 0,
            minute=0,
            second=0,
            microsecond=0,
        )
        entry = buckets.setdefault(bucket, {"posts": 0, "engagement": 0, "tox": []})
        entry["posts"] += 1
        entry["engagement"] += sum(
            v for v in (post.likes, post.shares, post.replies) if v is not None
        )
        toxicity = (post.scores or {}).get("toxicity")
        if toxicity is not None:
            entry["tox"].append(toxicity)

    return AuthorTimeline(
        author_id=author.author_id,
        interval=interval,
        tz=tz,
        points=[
            AuthorTimelinePoint(
                bucket_start=bucket,
                post_count=data["posts"],
                engagement=data["engagement"],
                mean_toxicity=(
                    round(sum(data["tox"]) / len(data["tox"]), 3) if data["tox"] else None
                ),
            )
            for bucket, data in sorted(buckets.items())
        ],
        hour_histogram=dict(sorted(histogram.items())),
    )


def drivers(spec: FilterSpec, page: Page, entity_type: str):
    from app.schemas.drivers import DriverItem, DriversResponse

    posts = apply_post_filters(corpus().posts, spec)
    tallies: dict[str, dict[str, Any]] = {}

    def bump(key: str, post, label: str | None = None) -> None:
        entry = tallies.setdefault(
            key,
            {
                "label": label,
                "posts": 0,
                "authors": set(),
                "engagement": 0,
                "tox": [],
                "narratives": set(),
                "first": post.timestamp,
                "last": post.timestamp,
                "bots": set(),
            },
        )
        entry["posts"] += 1
        entry["authors"].add(post.author_id)
        entry["engagement"] += sum(
            v for v in (post.likes, post.shares, post.replies) if v is not None
        )
        toxicity = (post.scores or {}).get("toxicity")
        if toxicity is not None:
            entry["tox"].append(toxicity)
        entry["narratives"].update(post.narrative_ids)
        entry["first"] = min(entry["first"], post.timestamp)
        entry["last"] = max(entry["last"], post.timestamp)
        author = corpus().by_author[post.author_id]
        if author.bot_score is not None and author.bot_score > 0.6:
            entry["bots"].add(post.author_id)

    for post in posts:
        if entity_type == "author":
            bump(post.author_id, post, post.author_handle)
        elif entity_type == "hashtag":
            for tag in post.hashtags:
                bump(tag, post, tag)
        else:
            for url in post.urls:
                bump(url, post, url)

    items = [
        DriverItem(
            entity=key,
            entity_type=entity_type,  # type: ignore[arg-type]
            label=data["label"],
            post_count=data["posts"],
            author_count=None if entity_type == "author" else len(data["authors"]),
            engagement_total=data["engagement"],
            bot_like_share=(
                round(len(data["bots"]) / len(data["authors"]), 3) if data["authors"] else None
            ),
            mean_toxicity=round(sum(data["tox"]) / len(data["tox"]), 3) if data["tox"] else None,
            narrative_ids=sorted(data["narratives"])[:5],
            first_seen=data["first"],
            last_seen=data["last"],
        )
        for key, data in tallies.items()
    ]
    items.sort(key=lambda item: -item.post_count)
    paged = paginate(items, page, spec.as_response_dict())
    return DriversResponse(
        entity_type=entity_type,  # type: ignore[arg-type]
        items=paged.items,
        definition=DEFINITIONS[f"drivers_{entity_type}s"],
    ), paged


def network_graph(
    spec: FilterSpec, narrative_id: str | None, max_nodes: int, bucket: str, hide_standalone: bool
):
    from app.schemas.network import (
        GraphBucket,
        GraphEdge,
        GraphLegendEntry,
        GraphNode,
        GraphResponse,
        GraphStats,
    )

    c = corpus()
    edges = c.edges
    if narrative_id:
        members = set(c.by_narrative[narrative_id].author_ids)
        edges = [e for e in edges if e["source"] in members and e["target"] in members]

    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1
    if hide_standalone:
        degree = {k: v for k, v in degree.items() if v > 1}

    ranked = sorted(degree.items(), key=lambda kv: -kv[1])
    truncated = len(ranked) > max_nodes
    kept = dict(ranked[:max_nodes])
    # Never silently truncate. A graph that quietly drops its periphery makes a
    # coordinated cluster look more isolated than it is, which is exactly the
    # wrong error for this product, so the rule and the counts travel with it.
    truncation = (
        {
            "applied_max_nodes": max_nodes,
            "dropped_nodes": len(ranked) - max_nodes,
            "dropped_edges": sum(
                1 for e in edges if e["source"] not in kept or e["target"] not in kept
            ),
            "min_degree_kept": ranked[max_nodes - 1][1] if max_nodes else None,
            "rule": "descending node degree",
        }
        if truncated
        else None
    )
    kept_edges = [e for e in edges if e["source"] in kept and e["target"] in kept]

    # A deterministic circular seed layout. The real implementation reads
    # precomputed positions from network_layouts; the point of putting
    # coordinates here at all is that the browser must never be handed a
    # position-less 20k-node graph and asked to lay it out live.
    import math

    nodes = []
    for index, (author_id, deg) in enumerate(kept.items()):
        author = c.by_author[author_id]
        angle = 2 * math.pi * index / max(len(kept), 1)
        radius = 100 + 400 * (1 - deg / max(kept.values()))
        nodes.append(
            GraphNode(
                id=author_id,
                label=author.handle,
                source=author.source,
                degree=deg,
                x=round(radius * math.cos(angle), 2),
                y=round(radius * math.sin(angle), 2),
                community_id=author.community_id,
                bot_score=author.bot_score,
                is_bot_like=None if author.bot_score is None else author.bot_score > 0.6,
                post_count=author.post_count,
                cohort_ids=author.cohort_ids,
            )
        )

    bucket_starts = sorted({e["bucket_start"] for e in kept_edges})
    hours = int(bucket.rstrip("h")) if bucket.endswith("h") else 12
    return GraphResponse(
        project_id=c.project_id,
        narrative_id=narrative_id,
        nodes=nodes,
        edges=[
            GraphEdge(
                source=e["source"],
                target=e["target"],
                edge_type=e["edge_type"],
                weight=e["weight"],
                first_ts=e["first_ts"],
                last_ts=e["last_ts"],
                bucket_start=e["bucket_start"],
            )
            for e in kept_edges
        ],
        buckets=[
            GraphBucket(
                bucket_start=start,
                bucket_end=start + timedelta(hours=hours),
                node_count=len(
                    {e["source"] for e in kept_edges if e["bucket_start"] == start}
                    | {e["target"] for e in kept_edges if e["bucket_start"] == start}
                ),
                edge_count=sum(1 for e in kept_edges if e["bucket_start"] == start),
            )
            for start in bucket_starts
        ],
        layout="forceatlas2" if len(nodes) > 1000 else "client",
        legend=[
            GraphLegendEntry(
                key="bot_like",
                label="Bot-like account",
                colour_role="danger",
                definition="Bot probability above 0.6 from the XGBoost account classifier.",
            ),
            GraphLegendEntry(
                key="co_post_similarity",
                label="Co-posting similarity",
                colour_role="accent",
                definition=(
                    "Two accounts posted near-duplicate text within the same window. "
                    "Not proof of coordination on its own."
                ),
            ),
            GraphLegendEntry(
                key="reply",
                label="Reply",
                colour_role="neutral",
                definition="A direct reply relationship reconstructed from parent_id.",
            ),
        ],
        stats=GraphStats(
            node_count=len(nodes),
            edge_count=len(kept_edges),
            truncated=truncated,
            truncation=truncation,
            density=(
                round(2 * len(kept_edges) / (len(nodes) * (len(nodes) - 1)), 6)
                if len(nodes) > 1
                else None
            ),
            component_count=1,
            largest_component_size=len(nodes),
            modularity=0.412,
        ),
        hide_standalone=hide_standalone,
        bucket_width=bucket,
    )


def comparisons_list():
    from app.mock.corpus import _stable_uuid
    from app.schemas.network import ComparisonMetricRow, ComparisonOut

    c = corpus()
    picked = c.narratives[:3]
    ids = [n.id for n in picked]

    def row(metric: str, definition: str, fn) -> ComparisonMetricRow:
        return ComparisonMetricRow(
            metric=metric, definition=definition, values={n.id: fn(n) for n in picked}
        )

    # Authors appearing in more than one compared narrative. This overlap is the
    # entire reason to run a comparison, so it ships with the row data rather
    # than needing a second call.
    membership: dict[str, set[str]] = {}
    for narrative in picked:
        for author_id in narrative.author_ids:
            membership.setdefault(author_id, set()).add(narrative.id)
    shared = [
        {"entity": author_id, "entity_type": "author", "narrative_ids": sorted(narratives)}
        for author_id, narratives in membership.items()
        if len(narratives) > 1
    ][:25]

    return [
        ComparisonOut(
            id=_stable_uuid("comparison", "demo"),
            project_id=c.project_id,
            name="CW v. WM v. RS",
            narrative_ids=ids,
            narrative_titles={n.id: n.title for n in picked},
            platforms=sorted({p for n in picked for p in n.platforms}),
            entity_type="author",
            rows=[
                row("posts", "Matched post count.", lambda n: len(n.post_ids)),
                row("authors", "Distinct authors.", lambda n: len(n.author_ids)),
                row("engagement", "Likes + shares + replies.", lambda n: n.engagement_total),
                row(
                    "fusion_score",
                    "0-100 composite. Null where a component could not be computed.",
                    lambda n: n.scorecard["fusion_score"],
                ),
                row(
                    "bot_like",
                    "Share of member authors above the bot threshold.",
                    lambda n: n.scorecard["bot_like"],
                ),
                row("coherence", "Cluster tightness.", lambda n: n.coherence),
            ],
            shared_entities=shared,
            created_at=DEMO_NOW - timedelta(days=2),
        )
    ]


# ---------------------------------------------------------------------------
# jobs, media, alerts, reports, ingest
# ---------------------------------------------------------------------------
def job(kind: str, status: str = "succeeded", job_id: str | None = None):
    from app.mock.corpus import _stable_uuid
    from app.schemas.jobs import JobOut

    identifier = job_id or _stable_uuid("job", kind)
    return JobOut(
        id=identifier,
        kind=kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        project_id=corpus().project_id,
        progress=1.0 if status == "succeeded" else 0.35,
        params={"demo": True},
        result={"note": "DEMO_MODE: no work was performed."} if status == "succeeded" else None,
        error=None,
        celery_task_id=None,
        created_at=DEMO_NOW - timedelta(minutes=12),
        started_at=DEMO_NOW - timedelta(minutes=11),
        finished_at=DEMO_NOW - timedelta(minutes=2) if status == "succeeded" else None,
    )


def jobs_list():
    return [
        job("nlp.cluster", "succeeded"),
        job("score.fusion", "succeeded"),
        job("ingest.load", "running"),
        job("media.deepfake", "failed"),
    ]


def media_check(job_id: str, status: str = "succeeded"):
    from app.schemas.media import MediaCheckOut

    settings = get_api_settings()
    if status != "succeeded":
        return MediaCheckOut(
            job_id=job_id,
            status=status,  # type: ignore[arg-type]
            submitted_at=DEMO_NOW - timedelta(minutes=1),
            retention={
                "deletes_at": (
                    DEMO_NOW + timedelta(hours=settings.media_retention_hours)
                ).isoformat(),
                "retention_hours": settings.media_retention_hours,
            },
        )
    return MediaCheckOut(
        job_id=job_id,
        status="succeeded",
        filename="clip.mp4",
        media_type="video/mp4",
        size_bytes=4_812_004,
        submitted_at=DEMO_NOW - timedelta(minutes=9),
        completed_at=DEMO_NOW - timedelta(minutes=4),
        verdict="possibly_manipulated",
        confidence=0.68,
        manipulation_type="face_swap",
        frames_analyzed=48,
        face_detected=True,
        explanation=(
            "48 frames were sampled at one-second intervals and a face was found in 44 "
            "of them. The classifier scored 31 of those faces above its manipulation "
            "threshold, concentrated in the middle third of the clip where the subject "
            "turns. That pattern is consistent with a face swap, but heavy compression "
            "produces similar artefacts, so this is a lead for manual review rather "
            "than a determination."
        ),
        model="xception-deepfake",
        model_version="v0.1.0",
        limitations=[
            "Trained on FaceForensics++ and DFDC; generators outside those sets are "
            "out of distribution.",
            "Below 480p the false-positive rate rises sharply.",
            "No face detected means no verdict, not an authentic verdict.",
        ],
        retention={
            "deletes_at": (DEMO_NOW + timedelta(hours=settings.media_retention_hours)).isoformat(),
            "retention_hours": settings.media_retention_hours,
        },
    )


def alert_rules():
    from app.mock.corpus import _stable_uuid
    from app.schemas.alerts import AlertCondition, AlertRuleOut

    return [
        AlertRuleOut(
            id=_stable_uuid("alert_rule", "high-fusion"),
            project_id=corpus().project_id,
            name="High fusion score",
            condition=AlertCondition(metric="fusion_score", op=">", value=70, scope="narrative"),
            channels=["in_app"],
            enabled=True,
            cooldown_minutes=120,
            created_at=DEMO_NOW - timedelta(days=30),
            last_evaluated_at=DEMO_NOW - timedelta(minutes=7),
            last_triggered_at=DEMO_NOW - timedelta(hours=4),
            trigger_count=11,
        ),
        AlertRuleOut(
            id=_stable_uuid("alert_rule", "bot-surge"),
            project_id=corpus().project_id,
            name="Bot-like amplification surge",
            condition=AlertCondition(
                metric="bot_like_ratio", op=">=", value=0.4, scope="narrative"
            ),
            channels=["in_app", "email"],
            enabled=True,
            cooldown_minutes=60,
            created_at=DEMO_NOW - timedelta(days=12),
            last_evaluated_at=DEMO_NOW - timedelta(minutes=7),
            last_triggered_at=None,
            trigger_count=0,
        ),
    ]


def alerts_list():
    from app.mock.corpus import _stable_uuid
    from app.schemas.alerts import AlertOut

    rules = alert_rules()
    high = [n for n in corpus().narratives if n.scorecard["priority"] == "high"][:3]
    return [
        AlertOut(
            id=_stable_uuid("alert", narrative.id),
            rule_id=rules[0].id,
            rule_name=rules[0].name,
            project_id=corpus().project_id,
            narrative_id=narrative.id,
            narrative_title=narrative.title,
            subject_type="narrative",
            subject_id=narrative.id,
            triggered_at=DEMO_NOW - timedelta(hours=4 + index),
            payload={
                "metric": "fusion_score",
                "op": ">",
                "threshold": 70,
                "observed": narrative.scorecard["fusion_score"],
                "scoring_version": narrative.scorecard["scoring_version"],
            },
            acknowledged_at=DEMO_NOW - timedelta(hours=1) if index == 0 else None,
            acknowledged_by="demo-analyst" if index == 0 else None,
        )
        for index, narrative in enumerate(high)
    ]


def reports_list():
    from app.mock.corpus import _stable_uuid
    from app.schemas.reports import ReportOut

    identifier = _stable_uuid("report", "exec")
    return [
        ReportOut(
            id=identifier,
            project_id=corpus().project_id,
            template="exec_summary",
            format="pdf",
            status="succeeded",
            params={"include_compass": True},
            size_bytes=284_113,
            download_url=f"/api/v1/reports/{identifier}/download",
            created_at=DEMO_NOW - timedelta(days=1),
            finished_at=DEMO_NOW - timedelta(days=1) + timedelta(seconds=42),
        )
    ]


def ingest_runs():
    from app.mock.corpus import _stable_uuid
    from app.schemas.ingest import IngestRunOut

    return [
        IngestRunOut(
            id=_stable_uuid("ingest_run", "reddit"),
            project_id=corpus().project_id,
            source="reddit",
            status="succeeded",
            mode="both",
            records_in=1503,
            records_loaded=1498,
            records_rejected=5,
            # Every rejected row is accounted for by reason code. Silent data
            # loss between Parquet and Postgres poisons every downstream metric
            # and surfaces three weeks later as an unexplainable number.
            rejection_reasons={"empty_text": 3, "missing_timestamp": 2},
            rejects_path="data/rejects/demo-reddit.parquet",
            manifest_sha="785d4044e38d4f0d5eba7aca1628ff4b43efe481144f9b647e6555fbc3366a3a",
            started_at=DEMO_NOW - timedelta(hours=3),
            finished_at=DEMO_NOW - timedelta(hours=3) + timedelta(seconds=88),
        ),
        IngestRunOut(
            id=_stable_uuid("ingest_run", "youtube"),
            project_id=corpus().project_id,
            source="youtube",
            status="partial",
            mode="fetch",
            records_in=210,
            records_loaded=210,
            records_rejected=0,
            detail="Stopped at the daily quota ceiling; resumes from the stored checkpoint.",
            started_at=DEMO_NOW - timedelta(hours=14),
            finished_at=DEMO_NOW - timedelta(hours=14) + timedelta(seconds=300),
        ),
    ]
