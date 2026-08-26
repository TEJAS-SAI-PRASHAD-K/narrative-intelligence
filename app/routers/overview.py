"""Overview / dashboard.

Every response here carries its own definition text. The UI spec requires a
`(?)` tooltip on each KPI explaining exactly how it was computed, and only the
backend knows what the query summed, which nulls it skipped and which rows the
filter excluded.

The five KPI drilldowns are separate routes rather than one fat endpoint because
they are genuinely separate expensive aggregations. Bundling them would make the
overview page pay for the emotions histogram every time somebody only wanted the
post count.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import Filters, Pagination, RequireRead, rate_limit
from app.errors import NotImplementedYet
from app.mock import responses as mock
from app.schemas.common import PageResponse
from app.schemas.overview import (
    AuthorsBreakdown,
    ConceptsResponse,
    EmotionsBreakdown,
    EngagementsBreakdown,
    KpiBundle,
    PostsBreakdown,
    SentimentBreakdown,
    TimeseriesResponse,
)
from app.schemas.posts import HighRiskPost

router = APIRouter(prefix="/overview", tags=["overview"], dependencies=[Depends(rate_limit)])


@router.get(
    "/kpis",
    response_model=KpiBundle,
    summary="The five headline KPIs",
    description=(
        "Posts, Engagements, Authors, Sentiment, Emotions -- in that fixed order, each "
        "with the definition text the UI renders in its `(?)` tooltip."
    ),
)
async def kpis(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> KpiBundle:
    if mock.demo_mode():
        return mock.overview_kpis(filters)
    from app.repositories.overview import kpi_bundle

    return await kpi_bundle(session, filters)


@router.get(
    "/timeseries",
    response_model=TimeseriesResponse,
    summary="Volume over time",
    description=(
        "Buckets with no matching posts are emitted as zero rather than omitted, so a "
        "gap in the data reads as a gap rather than as a shorter axis. When "
        "`normalize=true` each series is scaled to its own peak: shape becomes "
        "comparable and magnitude does not, and the response says so."
    ),
)
async def timeseries(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
    metric: Annotated[str, Query(pattern="^(posts|engagements|authors)$")] = "posts",
    group_by: Annotated[
        str, Query(pattern="^(platform|narrative|cohort|source_detail|none)$")
    ] = "none",
    interval: Annotated[str, Query(pattern="^(1h|6h|1d|7d)$")] = "1d",
    normalize: bool = False,
    agg: Annotated[str, Query(pattern="^(total|average|peak)$")] = "total",
) -> TimeseriesResponse:
    if mock.demo_mode():
        return mock.overview_timeseries(filters, metric, group_by, interval, normalize, agg)
    from app.repositories.overview import timeseries as query_timeseries

    return await query_timeseries(session, filters, metric, group_by, interval, normalize, agg)


@router.get("/concepts", response_model=ConceptsResponse, summary="Extracted topics and entities")
async def concepts(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConceptsResponse:
    if mock.demo_mode():
        return mock.overview_concepts(filters)
    from app.repositories.overview import concepts as query_concepts

    return await query_concepts(session, filters)


@router.get(
    "/high-risk-posts",
    response_model=PageResponse[HighRiskPost],
    summary="Highest-risk posts under the current filters",
)
async def high_risk_posts(
    principal: RequireRead,
    filters: Filters,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[HighRiskPost]:
    if mock.demo_mode():
        return mock.overview_high_risk_posts(filters, page)
    from app.repositories.overview import high_risk_posts as query_high_risk

    return await query_high_risk(session, filters, page)


# --- KPI drilldowns --------------------------------------------------------
@router.get("/kpis/posts", response_model=PostsBreakdown, summary="Posts KPI drilldown")
async def kpis_posts(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PostsBreakdown:
    if mock.demo_mode():
        return mock.overview_posts_breakdown(filters)
    raise NotImplementedYet(detail={"route": "/overview/kpis/posts", "lands_at": "build step 5"})


@router.get(
    "/kpis/engagements", response_model=EngagementsBreakdown, summary="Engagements KPI drilldown"
)
async def kpis_engagements(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EngagementsBreakdown:
    if mock.demo_mode():
        return mock.overview_engagements_breakdown(filters)
    raise NotImplementedYet(
        detail={"route": "/overview/kpis/engagements", "lands_at": "build step 5"}
    )


@router.get("/kpis/authors", response_model=AuthorsBreakdown, summary="Authors KPI drilldown")
async def kpis_authors(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AuthorsBreakdown:
    if mock.demo_mode():
        return mock.overview_authors_breakdown(filters)
    raise NotImplementedYet(detail={"route": "/overview/kpis/authors", "lands_at": "build step 9"})


@router.get("/kpis/emotions", response_model=EmotionsBreakdown, summary="Emotions KPI drilldown")
async def kpis_emotions(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EmotionsBreakdown:
    if mock.demo_mode():
        return mock.overview_emotions_breakdown(filters)
    from app.repositories.overview import emotions_breakdown

    return await emotions_breakdown(session, filters)


@router.get("/kpis/sentiment", response_model=SentimentBreakdown, summary="Sentiment KPI drilldown")
async def kpis_sentiment(
    principal: RequireRead,
    filters: Filters,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SentimentBreakdown:
    if mock.demo_mode():
        return mock.overview_sentiment_breakdown(filters)
    from app.repositories.overview import sentiment_breakdown

    return await sentiment_breakdown(session, filters)
