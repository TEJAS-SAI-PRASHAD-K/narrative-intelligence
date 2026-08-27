"""``make seed`` -- the curated demo project.

Runs the whole pipeline over whatever is already on disk, in dependency order,
and stops at the first step that cannot run rather than half-building a project
that looks complete.

The point is that a live demo never depends on a network call or on a model
warming up successfully on stage. Everything here reads from the committed
Parquet corpus and Phase 2/3's committed scored tables; the only optional step
is Compass, which needs an LLM key and refuses cleanly without one.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

DEMO_NAME = "Election Integrity 2026 (seed)"
DEMO_DESCRIPTION = (
    "Curated demo project. Built entirely from the committed Phase 1 corpus and "
    "Phase 2/3 scored tables, so it needs no network and no model warm-up."
)


def build_demo_project(session: Session, *, slug: str = "demo") -> dict[str, Any]:
    """Create the project and run every step that can run.

    Idempotent: every underlying step is, so `make seed` twice is `make seed`
    once. That matters because the first thing anybody does when a demo looks
    wrong is run it again.
    """
    from app.etl.scored_loader import load_all
    from app.etl.service import load_corpus
    from app.models.core import Project

    summary: dict[str, Any] = {}

    project = session.execute(select(Project).where(Project.slug == slug)).scalar_one_or_none()
    if project is None:
        project = Project(slug=slug, name=DEMO_NAME, description=DEMO_DESCRIPTION)
        session.add(project)
        session.flush()
        summary["project"] = f"created {slug} ({project.id})"
    else:
        summary["project"] = f"reusing {slug} ({project.id})"

    # A post belongs to exactly one project: `posts.id` is the primary key and
    # is namespaced by source, not by project. Loading the same Parquet under a
    # second slug is therefore a no-op that produces an empty project which
    # *looks* built -- so detect it and say so instead.
    owner = session.execute(
        text(
            """
            SELECT pr.slug, count(*) AS posts
            FROM posts p JOIN projects pr ON pr.id = p.project_id
            WHERE p.project_id <> :this
            GROUP BY pr.slug
            ORDER BY count(*) DESC
            LIMIT 1
            """
        ),
        {"this": project.id},
    ).first()
    if owner is not None:
        summary["stopped"] = (
            f"The Parquet corpus is already loaded under project {owner.slug!r} "
            f"({owner.posts} posts). A post belongs to exactly one project, so "
            f"seeding {slug!r} would produce an empty project that looks built.\n"
            f"Seed that project instead:  make seed P={owner.slug}\n"
            f"Or start clean:             python -m app.etl.cli truncate "
            f"--project {owner.slug}"
        )
        return summary

    # 1. The corpus. Everything else derives from it.
    report = load_corpus(session, project=slug)
    summary["corpus"] = {
        "read": report.records_in,
        "loaded": report.records_loaded,
        "unchanged": report.records_unchanged,
        "rejected": report.records_rejected,
        "authors": report.authors_touched,
        "domains": report.domains_touched,
    }
    if not report.records_in:
        # Stop rather than proceed. Every later step would "succeed" over zero
        # rows and produce a project that looks built and contains nothing.
        summary["stopped"] = (
            "No Parquet corpus found under data/normalized. Run the Phase 1 "
            "pipeline (`make data`) before seeding."
        )
        return summary

    # 2. Phase 2/3's committed scores. No models, no GPU, no network.
    scored = load_all(session, project=slug)
    summary["scores"] = {
        "post_scores": scored.post_scores,
        "authors_scored": scored.author_scores,
        "narratives": scored.narratives,
        "narrative_posts": scored.narrative_posts,
        "network_edges": scored.network_edges,
        "media_checks": scored.media_checks,
        "embeddings": scored.embeddings,
    }

    # 3. Derived layers, in dependency order. Each is skipped with a reason
    #    rather than failing the seed, so a partial environment still produces
    #    a usable demo.
    summary["cohorts"] = _step(_assign_cohorts, session, project.id)
    summary["graph"] = _step(_build_graph, session, project.id, slug)
    summary["fusion"] = _step(_score_fusion, session, slug)
    summary["domains"] = _step(_score_domains, session, slug)
    summary["alerts"] = _step(_seed_alert_rules, session, project.id)

    return summary


def _step(fn, *args) -> Any:
    """Run one seed step, converting a failure into a recorded reason.

    A seed that dies on step four leaves a project that looks half-built with
    no explanation. A seed that records "graph: networkx not installed" and
    continues leaves one an operator can reason about.
    """
    try:
        return fn(*args)
    except Exception as exc:
        log.warning("seed step %s failed: %s", fn.__name__, exc)
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"[:300]}


def _assign_cohorts(session: Session, project_id) -> dict:
    from app.scoring import cohorts

    return cohorts.assign(session, project_id)


def _build_graph(session: Session, project_id, slug: str) -> dict:
    """Edges and communities, in-process rather than through the broker.

    `make seed` must work without a running worker: somebody setting up for a
    demo should not have to discover that the graph is empty because celery was
    not started.
    """
    from app.tasks.graph import build_edges, communities

    session.commit()  # the tasks open their own sessions
    edges = build_edges.run(job_id=None, project_id=slug)
    community = communities.run(job_id=None, project_id=slug)
    return {"edges": edges, "communities": community}


def _score_fusion(session: Session, slug: str) -> dict:
    from app.tasks.scoring import score_fusion

    session.commit()
    return score_fusion.run(job_id=None, project_id=slug)


def _score_domains(session: Session, slug: str) -> dict:
    from app.tasks.domains import enrich

    session.commit()
    # batch=False and no domain: score from in-corpus signals, attempt no
    # network lookups. A demo must not wait on WHOIS.
    return enrich.run(job_id=None, project_id=slug, skip_enrichment=True)


def _seed_alert_rules(session: Session, project_id) -> dict:
    """Two rules, so the alerts page is not empty on a fresh demo."""
    created = 0
    for name, condition, cooldown in (
        (
            "High fusion score",
            {"metric": "fusion_score", "op": ">", "value": 60, "scope": "narrative"},
            120,
        ),
        (
            "Bot-like amplification",
            {"metric": "bot_like_ratio", "op": ">=", "value": 0.4, "scope": "narrative"},
            60,
        ),
    ):
        existing = session.execute(
            text("SELECT 1 FROM alert_rules WHERE project_id = :p AND name = :n"),
            {"p": project_id, "n": name},
        ).scalar()
        if existing:
            continue
        session.execute(
            text(
                """
                INSERT INTO alert_rules (project_id, name, condition, channels,
                                         enabled, cooldown_minutes)
                VALUES (:p, :n, cast(:c AS jsonb), ARRAY['in_app'], true, :cd)
                """
            ),
            {"p": project_id, "n": name, "c": _dumps(condition), "cd": cooldown},
        )
        created += 1

    from app.tasks.alerts import evaluate

    session.commit()
    return {"rules_created": created, "evaluation": evaluate.run(job_id=None)}


def _dumps(value) -> str:
    import json

    return json.dumps(value)
