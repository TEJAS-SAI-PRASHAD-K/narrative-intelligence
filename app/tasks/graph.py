"""Graph tasks: edge building, community detection and layout.

Edges are built with SQL, not by pulling the corpus into Python. Reply, repost
and mention edges are joins the database already has indexes for; co-posting
similarity is a self-join on `simhash`. Materialising 4,000 posts into pandas to
compute what one GROUP BY answers is the slow, memory-hungry way to get the same
number.

Every write upserts on the edge identity index, so a retried build refreshes
weights rather than doubling them. A coordination graph that looks more
coordinated after each retry is a subtle bug with expensive consequences.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.tasks.base import TrackedTask, report_progress
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)

#: Reply, repost and mention edges, derived from the corpus itself. Bucketed at
#: write time so the UI's scrubber is an index range rather than a range scan.
#:
#: Note `cast(:bucket AS interval)` rather than `:bucket::interval`. SQLAlchemy's
#: text() parser does not recognise a bind followed by a cast operator and
#: leaves the literal in the statement, which fails at execute time with a
#: syntax error that points at the wrong line.
_INTERACTION_EDGES = """
INSERT INTO network_edges (
    project_id, narrative_id, src_author_id, dst_author_id, edge_type,
    weight, bucket_start, first_ts, last_ts, observations
)
SELECT
    :project_id, NULL, child.author_id, parent.author_id, 'reply',
    count(*)::real,
    date_bin(cast(:bucket AS interval), child."timestamp", timestamp '2000-01-01 00:00+00'),
    min(child."timestamp"), max(child."timestamp"), count(*)
FROM posts child
JOIN posts parent ON parent.id = child.parent_id
WHERE child.project_id = :project_id
  AND parent.project_id = :project_id
  -- A self-reply is a thread, not an interaction between two actors.
  AND child.author_id <> parent.author_id
GROUP BY child.author_id, parent.author_id,
         date_bin(cast(:bucket AS interval), child."timestamp", timestamp '2000-01-01 00:00+00')
ON CONFLICT (project_id, narrative_id, src_author_id, dst_author_id, edge_type, bucket_start)
DO UPDATE SET
    weight = EXCLUDED.weight,
    last_ts = GREATEST(network_edges.last_ts, EXCLUDED.last_ts),
    observations = EXCLUDED.observations
"""

_MENTION_EDGES = """
INSERT INTO network_edges (
    project_id, narrative_id, src_author_id, dst_author_id, edge_type,
    weight, bucket_start, first_ts, last_ts, observations
)
SELECT
    :project_id, NULL, p.author_id, target.author_id, 'mention',
    count(*)::real,
    date_bin(cast(:bucket AS interval), p."timestamp", timestamp '2000-01-01 00:00+00'),
    min(p."timestamp"), max(p."timestamp"), count(*)
FROM posts p
CROSS JOIN LATERAL unnest(p.mentions) AS m(handle)
-- Mentions are handles, and a handle is only unique within a source. Joining
-- across sources would merge two different people who happen to share a name.
JOIN authors target ON lower(target.handle) = lower(m.handle)
                   AND target.project_id = :project_id
                   AND target.source = p.source
WHERE p.project_id = :project_id AND p.author_id <> target.author_id
GROUP BY p.author_id, target.author_id,
         date_bin(cast(:bucket AS interval), p."timestamp", timestamp '2000-01-01 00:00+00')
ON CONFLICT (project_id, narrative_id, src_author_id, dst_author_id, edge_type, bucket_start)
DO UPDATE SET
    weight = EXCLUDED.weight,
    last_ts = GREATEST(network_edges.last_ts, EXCLUDED.last_ts),
    observations = EXCLUDED.observations
"""

#: Co-posting similarity: two different authors publishing near-identical text
#: inside the same bucket. This is the edge that actually detects amplification
#: networks, and also the one most prone to false positives -- syndicated wire
#: copy produces it too, which is why the legend calls it suggestive, not proof.
_COPOST_EDGES = """
INSERT INTO network_edges (
    project_id, narrative_id, src_author_id, dst_author_id, edge_type,
    weight, bucket_start, first_ts, last_ts, observations
)
SELECT
    :project_id, NULL, least(a.author_id, b.author_id), greatest(a.author_id, b.author_id),
    'co_post_similarity', count(*)::real,
    date_bin(cast(:bucket AS interval), a."timestamp", timestamp '2000-01-01 00:00+00'),
    min(a."timestamp"), max(a."timestamp"), count(*)
FROM posts a
JOIN posts b
  ON b.project_id = :project_id
 AND a.simhash IS NOT NULL AND b.simhash IS NOT NULL
 -- Exact simhash equality rather than a Hamming radius. A radius join is a
 -- cross product with a bit-count predicate that no index can serve; at this
 -- corpus size the exact match already finds the repost swarms, and the
 -- approximate version is a separate offline job if it is ever needed.
 AND a.simhash = b.simhash
 AND a.author_id < b.author_id
 AND abs(extract(epoch FROM (a."timestamp" - b."timestamp"))) <= :window_seconds
WHERE a.project_id = :project_id
GROUP BY least(a.author_id, b.author_id), greatest(a.author_id, b.author_id),
         date_bin(cast(:bucket AS interval), a."timestamp", timestamp '2000-01-01 00:00+00')
ON CONFLICT (project_id, narrative_id, src_author_id, dst_author_id, edge_type, bucket_start)
DO UPDATE SET
    weight = EXCLUDED.weight,
    last_ts = GREATEST(network_edges.last_ts, EXCLUDED.last_ts),
    observations = EXCLUDED.observations
"""


@celery.task(name="graph.build_edges", base=TrackedTask, bind=True)
def build_edges(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Materialise the interaction graph for a project."""
    from sqlalchemy import text

    from app.config import get_api_settings
    from app.db import advisory_lock, sync_session
    from app.etl.service import resolve_project

    if params.get("all_projects"):
        from app.tasks.scoring import _fan_out_all_projects

        return _fan_out_all_projects("graph.build_edges")

    settings = get_api_settings()
    bucket = f"{settings.graph_bucket_hours} hours"
    window_seconds = int(params.get("copost_window_seconds", 24 * 3600))
    counts: dict[str, int] = {}

    with sync_session() as session:
        with advisory_lock(session, f"graph.build_edges:{project_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another graph build holds the lock"}

            resolved = resolve_project(session, project_id)
            base = {"project_id": resolved, "bucket": bucket}

            for index, (name, sql, extra) in enumerate(
                (
                    ("reply", _INTERACTION_EDGES, {}),
                    ("mention", _MENTION_EDGES, {}),
                    ("co_post_similarity", _COPOST_EDGES, {"window_seconds": window_seconds}),
                )
            ):
                result = session.execute(text(sql), {**base, **extra})
                counts[name] = result.rowcount
                report_progress(job_id, 0.2 + 0.25 * index)

            total = session.execute(
                text("SELECT count(*) FROM network_edges WHERE project_id = :p"),
                {"p": resolved},
            ).scalar()

    result = {"written": counts, "total_edges": total}

    # Communities depend on the edges that just landed.
    celery.send_task("graph.communities", kwargs={"project_id": project_id}, queue="cpu")

    report_progress(job_id, 1.0, result=result)
    return result


@celery.task(name="graph.communities", base=TrackedTask, bind=True)
def communities(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Louvain communities over the co-posting graph."""
    from sqlalchemy import text

    from app.db import advisory_lock, sync_session
    from app.etl.service import resolve_project

    try:
        import networkx as nx
    except ImportError:
        return {"skipped": True, "reason": "networkx is not installed in this worker"}

    with sync_session() as session:
        with advisory_lock(session, f"graph.communities:{project_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another community run holds the lock"}

            resolved = resolve_project(session, project_id)
            edges = session.execute(
                text(
                    """
                    SELECT src_author_id, dst_author_id, sum(weight) AS weight
                    FROM network_edges WHERE project_id = :p
                    GROUP BY src_author_id, dst_author_id
                    """
                ),
                {"p": resolved},
            ).all()

            if len(edges) < 10:
                return {"skipped": True, "reason": f"only {len(edges)} edges; nothing to partition"}

            report_progress(job_id, 0.3)
            graph = nx.Graph()
            for row in edges:
                graph.add_edge(row.src_author_id, row.dst_author_id, weight=float(row.weight))

            partition = nx.community.louvain_communities(graph, weight="weight", seed=42)
            modularity = nx.community.modularity(graph, partition, weight="weight")
            report_progress(job_id, 0.7)

            payload = []
            for index, members in enumerate(partition):
                # Deterministic per run, but never a foreign key: communities are
                # recomputed wholesale and carry no identity across runs, so an
                # FK would be a lie about stability.
                community_id = f"c{index:04d}"
                payload.extend(
                    {
                        "author_id": author_id,
                        "community_id": community_id,
                        "community_size": len(members),
                        "project_id": resolved,
                    }
                    for author_id in members
                )

            for start in range(0, len(payload), 2000):
                session.execute(
                    text(
                        """
                        UPDATE authors SET
                            community_id = :community_id, community_size = :community_size
                        WHERE author_id = :author_id AND project_id = :project_id
                        """
                    ),
                    payload[start : start + 2000],
                )

    result = {
        "communities": len(partition),
        "authors_assigned": len(payload),
        "modularity": round(float(modularity), 4),
        "largest": max((len(c) for c in partition), default=0),
    }
    report_progress(job_id, 1.0, result=result)
    return result


@celery.task(name="graph.layout", base=TrackedTask, bind=True)
def layout(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Precompute node positions so the browser never runs a force simulation.

    Above a few thousand nodes a client-side layout drops frames until the
    analyst concludes the tool is broken. Computing it once here and storing the
    coordinates turns that into a fetch.
    """
    from sqlalchemy import text

    from app.config import get_api_settings
    from app.db import advisory_lock, sync_session
    from app.etl.service import resolve_project

    try:
        import networkx as nx
    except ImportError:
        return {"skipped": True, "reason": "networkx is not installed in this worker"}

    settings = get_api_settings()
    narrative_id = params.get("narrative_id")
    algorithm = params.get("algorithm", "spring")
    max_nodes = min(
        int(params.get("max_nodes") or settings.max_graph_nodes), settings.max_graph_nodes
    )

    with sync_session() as session:
        with advisory_lock(session, f"graph.layout:{project_id}:{narrative_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another layout run holds the lock"}

            resolved = resolve_project(session, project_id)
            scope = "AND ne.narrative_id = :n" if narrative_id else ""
            edges = session.execute(
                text(
                    f"""
                    SELECT src_author_id, dst_author_id, sum(weight) AS weight
                    FROM network_edges ne
                    WHERE ne.project_id = :p {scope}
                    GROUP BY src_author_id, dst_author_id
                    """  # noqa: S608 - scope is a fixed string, the value is bound
                ),
                {"p": resolved, **({"n": uuid.UUID(narrative_id)} if narrative_id else {})},
            ).all()

            if not edges:
                return {"skipped": True, "reason": "no edges to lay out"}

            graph = nx.Graph()
            for row in edges:
                graph.add_edge(row.src_author_id, row.dst_author_id, weight=float(row.weight))

            if graph.number_of_nodes() > max_nodes:
                # Same rule the read path truncates by, so the layout covers
                # exactly the nodes the graph endpoint will return.
                keep = sorted(graph.degree, key=lambda item: -item[1])[:max_nodes]
                graph = graph.subgraph([node for node, _ in keep]).copy()

            report_progress(job_id, 0.4)
            # seed fixed: an analyst who reloads the page should see the same
            # shape, otherwise the graph looks like it changed when it did not.
            positions = nx.spring_layout(graph, seed=42, weight="weight", scale=500)
            report_progress(job_id, 0.85)

            session.execute(
                text(
                    """
                    INSERT INTO network_layouts (
                        project_id, narrative_id, algorithm, node_count, edge_count,
                        positions, computed_at
                    ) VALUES (
                        :project_id, :narrative_id, :algorithm, :node_count, :edge_count,
                        cast(:positions AS jsonb), now()
                    )
                    """
                ),
                {
                    "project_id": resolved,
                    "narrative_id": uuid.UUID(narrative_id) if narrative_id else None,
                    "algorithm": algorithm,
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                    "positions": json.dumps(
                        {
                            n: [round(float(x), 2), round(float(y), 2)]
                            for n, (x, y) in positions.items()
                        }
                    ),
                },
            )

    result = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "algorithm": algorithm,
    }
    report_progress(job_id, 1.0, result=result)
    return result
