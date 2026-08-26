"""The interaction graph and saved comparisons.

The rule that matters most in this module: **truncation is never silent.** When
the node ceiling bites, the response says so, says by how much, and says by what
rule. A graph that quietly drops its periphery makes a coordinated cluster look
more isolated than it is, and for a coordination-detection product that is
exactly the wrong direction to be wrong in.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_api_settings
from app.deps import FilterSpec, Page
from app.errors import NotFound
from app.repositories.filters import decode_cursor, encode_cursor, resolve_project_id
from app.schemas.common import PageResponse
from app.schemas.network import (
    ComparisonCreate,
    ComparisonMetricRow,
    ComparisonOut,
    GraphBucket,
    GraphEdge,
    GraphLegendEntry,
    GraphNode,
    GraphResponse,
    GraphStats,
)

log = logging.getLogger(__name__)

LEGEND = [
    GraphLegendEntry(
        key="bot_like",
        label="Bot-like account",
        colour_role="danger",
        definition=(
            "Bot probability above 0.6 from the XGBoost account classifier. "
            "Generalisation to unseen campaigns is weak; treat as a lead."
        ),
    ),
    GraphLegendEntry(
        key="co_post_similarity",
        label="Co-posting similarity",
        colour_role="accent",
        definition=(
            "Two accounts posted near-duplicate text inside the same window. "
            "Suggestive of coordination, not proof of it: syndicated content and "
            "quote-tweet chains produce the same edge."
        ),
    ),
    GraphLegendEntry(
        key="reply",
        label="Reply",
        colour_role="neutral",
        definition="A direct reply relationship reconstructed from parent_id.",
    ),
    GraphLegendEntry(
        key="repost",
        label="Repost",
        colour_role="neutral",
        definition="One account reposted another's content.",
    ),
    GraphLegendEntry(
        key="mention",
        label="Mention",
        colour_role="muted",
        definition="One account named another in its text.",
    ),
]


async def build_graph(
    session: AsyncSession,
    spec: FilterSpec,
    *,
    narrative_id: str | None = None,
    comparison_id: str | None = None,
    max_nodes: int,
    bucket: str = "12h",
    hide_standalone: bool = False,
) -> GraphResponse:
    from app.models.actors import Author
    from app.models.network import NetworkEdge

    settings = get_api_settings()
    project_id = await resolve_project_id(session, spec.project_id)

    narrative_ids: list[uuid.UUID] = []
    if narrative_id:
        narrative_ids = [uuid.UUID(narrative_id)]
    elif comparison_id:
        comparison = await _load_comparison(session, comparison_id)
        narrative_ids = list(comparison.narrative_ids)

    # Scope the edge set to the narratives' member authors. Done as a semi-join
    # on both endpoints so an edge only survives when *both* ends are inside the
    # narrative -- an edge to an author outside it is not evidence about it.
    scope_clause = ""
    params: dict[str, Any] = {"project_id": project_id}
    if narrative_ids:
        scope_clause = """
        AND e.src_author_id IN (
            SELECT DISTINCT p.author_id FROM narrative_posts np
            JOIN posts p ON p.id = np.post_id
            WHERE np.narrative_id = ANY(:narrative_ids)
        )
        AND e.dst_author_id IN (
            SELECT DISTINCT p.author_id FROM narrative_posts np
            JOIN posts p ON p.id = np.post_id
            WHERE np.narrative_id = ANY(:narrative_ids)
        )
        """
        params["narrative_ids"] = narrative_ids

    date_clause = ""
    if spec.date_from:
        date_clause += " AND e.bucket_start >= :date_from"
        params["date_from"] = spec.date_from
    if spec.date_to:
        date_clause += " AND e.bucket_start < :date_to"
        params["date_to"] = spec.date_to

    # Degree first, over the whole scoped edge set, so the truncation decision
    # is made against the real graph rather than against an arbitrary page of it.
    degrees = (
        await session.execute(
            text(
                f"""
                SELECT author_id, sum(deg) AS degree FROM (
                    SELECT e.src_author_id AS author_id, count(*) AS deg
                    FROM network_edges e
                    WHERE e.project_id = :project_id {scope_clause} {date_clause}
                    GROUP BY e.src_author_id
                    UNION ALL
                    SELECT e.dst_author_id AS author_id, count(*) AS deg
                    FROM network_edges e
                    WHERE e.project_id = :project_id {scope_clause} {date_clause}
                    GROUP BY e.dst_author_id
                ) d
                GROUP BY author_id
                ORDER BY sum(deg) DESC
                """  # noqa: S608 - clauses are built from fixed strings, values are bound
            ),
            params,
        )
    ).all()

    if hide_standalone:
        degrees = [row for row in degrees if row.degree > 1]

    total_nodes = len(degrees)
    truncated = total_nodes > max_nodes
    # int() on the degree: sum() over a bigint returns Decimal through psycopg,
    # and a Decimal serialises as a JSON string. The UI would then compare a
    # threshold against "15" rather than 15 and silently never match.
    kept = {row.author_id: int(row.degree) for row in degrees[:max_nodes]}
    dropped_degree_total = sum(int(row.degree) for row in degrees[max_nodes:])

    edge_rows = (
        (
            await session.execute(
                select(NetworkEdge)
                .where(
                    NetworkEdge.project_id == project_id,
                    NetworkEdge.src_author_id.in_(list(kept) or [""]),
                    NetworkEdge.dst_author_id.in_(list(kept) or [""]),
                )
                .limit(200_000)
            )
        )
        .scalars()
        .all()
    )

    authors = {
        row.author_id: row
        for row in (
            await session.execute(select(Author).where(Author.author_id.in_(list(kept) or [""])))
        ).scalars()
    }

    positions = await _layout_positions(session, project_id, narrative_ids, kept)

    nodes = [
        GraphNode(
            id=author_id,
            label=(authors[author_id].handle if author_id in authors else None),
            source=(authors[author_id].source if author_id in authors else "unknown"),
            degree=degree,
            x=positions.get(author_id, (None, None))[0],
            y=positions.get(author_id, (None, None))[1],
            community_id=(authors[author_id].community_id if author_id in authors else None),
            bot_score=(authors[author_id].bot_score if author_id in authors else None),
            is_bot_like=(
                None
                if author_id not in authors or authors[author_id].bot_score is None
                else authors[author_id].bot_score > 0.6
            ),
            post_count=(authors[author_id].post_count if author_id in authors else 0),
        )
        for author_id, degree in kept.items()
    ]

    hours = int(bucket.rstrip("h")) if bucket.endswith("h") else settings.graph_bucket_hours
    bucket_starts = sorted({e.bucket_start for e in edge_rows})
    buckets = [
        GraphBucket(
            bucket_start=start,
            bucket_end=start + timedelta(hours=hours),
            node_count=len(
                {e.src_author_id for e in edge_rows if e.bucket_start == start}
                | {e.dst_author_id for e in edge_rows if e.bucket_start == start}
            ),
            edge_count=sum(1 for e in edge_rows if e.bucket_start == start),
        )
        for start in bucket_starts
    ]

    node_count = len(nodes)
    return GraphResponse(
        project_id=str(project_id),
        narrative_id=narrative_id,
        comparison_id=comparison_id,
        nodes=nodes,
        edges=[
            GraphEdge(
                source=e.src_author_id,
                target=e.dst_author_id,
                edge_type=e.edge_type,
                weight=e.weight,
                first_ts=e.first_ts,
                last_ts=e.last_ts,
                bucket_start=e.bucket_start,
            )
            for e in edge_rows
        ],
        buckets=buckets,
        layout=("precomputed" if positions else "client"),
        legend=LEGEND,
        stats=GraphStats(
            node_count=node_count,
            edge_count=len(edge_rows),
            truncated=truncated,
            truncation=(
                {
                    "applied_max_nodes": max_nodes,
                    "dropped_nodes": total_nodes - max_nodes,
                    # Each dropped node's degree counts its incident edges, and
                    # an edge between two dropped nodes is counted twice, so
                    # this is halved. Approximate for edges that straddle the
                    # cut, and labelled as such rather than presented as exact.
                    "dropped_edges_approx": dropped_degree_total // 2,
                    "min_degree_kept": (int(degrees[max_nodes - 1].degree) if max_nodes else None),
                    "rule": "descending node degree",
                }
                if truncated
                else None
            ),
            density=(
                round(2 * len(edge_rows) / (node_count * (node_count - 1)), 6)
                if node_count > 1
                else None
            ),
            component_count=None,
            largest_component_size=None,
            modularity=None,
        ),
        hide_standalone=hide_standalone,
        bucket_width=bucket,
    )


async def _layout_positions(
    session: AsyncSession, project_id, narrative_ids, kept: dict[str, int]
) -> dict[str, tuple[float, float]]:
    """Precomputed positions, or a deterministic degree-ranked circle.

    The fallback is not a good layout and is not meant to be -- graph.layout
    computes those. It exists so the browser is never handed a position-less
    graph and asked to run a force simulation over it, which is the difference
    between a slow page and a page the analyst believes is broken.
    """
    from app.models.network import NetworkLayout

    stmt = select(NetworkLayout).where(NetworkLayout.project_id == project_id)
    stmt = (
        stmt.where(NetworkLayout.narrative_id == narrative_ids[0])
        if len(narrative_ids) == 1
        else stmt.where(NetworkLayout.narrative_id.is_(None))
    )
    layout = (
        await session.execute(stmt.order_by(NetworkLayout.computed_at.desc()).limit(1))
    ).scalar_one_or_none()

    if layout and layout.positions:
        stored = {
            author_id: (float(xy[0]), float(xy[1]))
            for author_id, xy in layout.positions.items()
            if author_id in kept
        }
        if stored:
            return stored

    if not kept:
        return {}
    peak = max(kept.values()) or 1
    return {
        author_id: (
            round((100 + 400 * (1 - degree / peak)) * math.cos(2 * math.pi * index / len(kept)), 2),
            round((100 + 400 * (1 - degree / peak)) * math.sin(2 * math.pi * index / len(kept)), 2),
        )
        for index, (author_id, degree) in enumerate(kept.items())
    }


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------
async def _load_comparison(session: AsyncSession, comparison_id: str):
    from app.models.narratives import Comparison

    row = await session.get(Comparison, comparison_id)
    if row is None:
        raise NotFound(f"No comparison with id {comparison_id}.", code="comparison_not_found")
    return row


async def list_comparisons(
    session: AsyncSession, project_id: str, page: Page
) -> PageResponse[ComparisonOut]:
    from app.models.narratives import Comparison

    resolved = await resolve_project_id(session, project_id)
    offset = decode_cursor(page.cursor).get("o", 0)
    rows = (
        (
            await session.execute(
                select(Comparison)
                .where(Comparison.project_id == resolved)
                .order_by(Comparison.created_at.desc())
                .offset(offset)
                .limit(page.limit)
            )
        )
        .scalars()
        .all()
    )
    return PageResponse[ComparisonOut](
        items=[await _to_comparison_out(session, row) for row in rows],
        next_cursor=(
            encode_cursor({"o": offset + page.limit}) if len(rows) == page.limit else None
        ),
        total=None,
        filters_applied={"project_id": project_id},
    )


async def create_comparison(session: AsyncSession, body: ComparisonCreate) -> ComparisonOut:
    from app.models.narratives import Comparison

    project_id = await resolve_project_id(session, body.project_id)
    narrative_ids = [uuid.UUID(n) for n in body.narrative_ids]

    known = set(
        (
            await session.execute(
                text("SELECT id FROM narratives WHERE id = ANY(:ids) AND project_id = :p"),
                {"ids": narrative_ids, "p": project_id},
            )
        ).scalars()
    )
    missing = [str(n) for n in narrative_ids if n not in known]
    if missing:
        raise NotFound(
            f"{len(missing)} narrative(s) are not in this project.",
            code="narrative_not_found",
            detail={"missing": missing},
        )

    row = Comparison(
        project_id=project_id,
        name=body.name,
        narrative_ids=narrative_ids,
        platforms=body.platforms,
        entity_type=body.entity_type,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return await _to_comparison_out(session, row)


async def get_comparison(session: AsyncSession, comparison_id: str) -> ComparisonOut:
    return await _to_comparison_out(session, await _load_comparison(session, comparison_id))


async def delete_comparison(session: AsyncSession, comparison_id: str) -> None:
    row = await _load_comparison(session, comparison_id)
    await session.delete(row)
    await session.commit()


async def _to_comparison_out(session: AsyncSession, row) -> ComparisonOut:
    """Build the side-by-side table and the shared-entity overlap.

    The overlap is the reason to run a comparison at all -- the same accounts
    carrying two narratives is the finding -- so it ships with the metrics
    rather than needing a second call.
    """
    metrics = (
        await session.execute(
            text(
                """
                SELECT n.id, n.title, n.post_count, n.author_count, n.engagement_total,
                       n.coherence, sc.fusion_score, sc.bot_like, sc.toxicity, sc.priority
                FROM narratives n
                LEFT JOIN narrative_scorecards sc ON sc.narrative_id = n.id
                WHERE n.id = ANY(:ids)
                """
            ),
            {"ids": list(row.narrative_ids)},
        )
    ).all()
    by_id = {str(m.id): m for m in metrics}

    def metric_row(name: str, definition: str, attribute: str) -> ComparisonMetricRow:
        return ComparisonMetricRow(
            metric=name,
            definition=definition,
            values={key: getattr(value, attribute) for key, value in by_id.items()},
        )

    shared = (
        await session.execute(
            text(
                """
                SELECT p.author_id, array_agg(DISTINCT np.narrative_id) AS narratives
                FROM narrative_posts np JOIN posts p ON p.id = np.post_id
                WHERE np.narrative_id = ANY(:ids)
                GROUP BY p.author_id
                HAVING count(DISTINCT np.narrative_id) > 1
                ORDER BY count(DISTINCT np.narrative_id) DESC
                LIMIT 50
                """
            ),
            {"ids": list(row.narrative_ids)},
        )
    ).all()

    return ComparisonOut(
        id=str(row.id),
        project_id=str(row.project_id),
        name=row.name,
        narrative_ids=[str(n) for n in row.narrative_ids],
        narrative_titles={key: value.title for key, value in by_id.items()},
        platforms=list(row.platforms or ()),
        entity_type=row.entity_type,
        rows=[
            metric_row("posts", "Member post count.", "post_count"),
            metric_row("authors", "Distinct member authors.", "author_count"),
            metric_row(
                "engagement",
                "Likes + shares + replies on member posts. Views excluded.",
                "engagement_total",
            ),
            metric_row(
                "fusion_score",
                "0-100 composite. Null where the narrative has not been scored.",
                "fusion_score",
            ),
            metric_row(
                "bot_like",
                "Share of member authors above the bot threshold, over scorable authors.",
                "bot_like",
            ),
            metric_row("toxicity", "Mean toxicity over scored member posts.", "toxicity"),
            metric_row("coherence", "Cluster tightness.", "coherence"),
        ],
        shared_entities=[
            {
                "entity": item.author_id,
                "entity_type": "author",
                "narrative_ids": [str(n) for n in item.narratives],
            }
            for item in shared
        ],
        created_at=row.created_at,
    )
