"""NLP tasks: embedding, clustering and summarization.

Every one is resumable. The pattern is the same throughout: ask the database
which rows already carry the current model version, skip those, and commit in
batches. A worker killed at 90% resumes at 90%, and because every write is an
upsert keyed on a natural id, a redelivered task cannot double-count.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.tasks.base import TrackedTask, report_progress
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="nlp.embed", base=TrackedTask, bind=True)
def embed(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Embed posts that have no current vector.

    The HNSW index is already built, so these are ordinary inserts into an
    indexed table. That is the right trade here: the bulk load happens through
    `app.etl.cli load-scores`, which runs before the index matters, and this
    task handles the incremental tail where per-row index maintenance is cheap.
    """
    from sqlalchemy import text

    from app.config import get_api_settings
    from app.db import advisory_lock, sync_session
    from app.etl.service import resolve_project
    from nlp.adapters import get_embedder

    settings = get_api_settings()
    batch_size = int(params.get("batch_size", 256))

    embedder = get_embedder()
    if embedder is None:
        return {
            "skipped": True,
            "reason": (
                "No embedding model could be loaded. Phase 2's cached vectors are "
                "already served; load them with `app.etl.cli load-scores`."
            ),
        }

    model_name = embedder.info.name
    embedded = 0

    with sync_session() as session:
        with advisory_lock(session, f"nlp.embed:{project_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another embedding run holds the lock"}

    while True:
        with sync_session() as session:
            resolved = resolve_project(session, project_id)
            pending = session.execute(
                text(
                    f"""
                    SELECT p.id, p."text"
                    FROM posts p
                    LEFT JOIN {settings.embedding_table} e ON e.post_id = p.id
                    WHERE p.project_id = :p
                      AND (e.post_id IS NULL OR e.model IS DISTINCT FROM :m)
                    ORDER BY p.id
                    LIMIT :limit
                    """  # noqa: S608 - table name comes from config, not from input
                ),
                {"p": resolved, "m": model_name, "limit": batch_size},
            ).all()
            if not pending:
                break

            vectors = embedder.encode([row.text for row in pending])
            payload = [
                {
                    "post_id": row.id,
                    "model": model_name,
                    "dim": settings.embedding_dim,
                    "embedding": "[" + ",".join(f"{v:.6f}" for v in vector) + "]",
                }
                for row, vector in zip(pending, vectors, strict=True)
            ]
            session.execute(
                text(
                    f"""
                    INSERT INTO {settings.embedding_table} (post_id, model, dim, embedding)
                    VALUES (:post_id, :model, :dim, cast(:embedding AS vector))
                    ON CONFLICT (post_id) DO UPDATE SET
                        model = EXCLUDED.model, dim = EXCLUDED.dim,
                        embedding = EXCLUDED.embedding
                    """  # noqa: S608 - table name comes from config, not from input
                ),
                payload,
            )
            embedded += len(payload)
        report_progress(job_id, min(0.95, embedded / max(embedded + batch_size, 1)))

    result = {"embedded": embedded, "model": model_name, "dim": settings.embedding_dim}
    report_progress(job_id, 1.0, result=result)
    return result


@celery.task(name="nlp.cluster", base=TrackedTask, bind=True)
def cluster(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """HDBSCAN over the project's embeddings.

    **A user-edited title survives this.** The upsert refreshes membership,
    counts and freshness for every narrative, but the text only where
    `edited_fields` does not claim it, and a manual narrative is skipped
    entirely. The rule lives in the SQL rather than in a code path somebody
    could forget to take, and `tests/test_tasks.py` asserts it.
    """
    import numpy as np
    from sqlalchemy import text

    from app.config import get_api_settings
    from app.db import advisory_lock, sync_session, utcnow
    from app.etl.service import resolve_project

    if params.get("all_projects"):
        from app.tasks.scoring import _fan_out_all_projects

        return _fan_out_all_projects("nlp.cluster")

    settings = get_api_settings()

    try:
        from modeling.text.cluster import NarrativeClusterer
    except ImportError:
        return {
            "skipped": True,
            "reason": (
                "The modeling package is not installed in this worker, so clustering "
                "cannot run. Phase 2/3's committed narratives are already served."
            ),
        }

    with sync_session() as session:
        with advisory_lock(session, f"nlp.cluster:{project_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another clustering run holds the lock"}

            resolved = resolve_project(session, project_id)
            rows = session.execute(
                text(
                    f"""
                    SELECT e.post_id, e.embedding, p."timestamp", p.author_id
                    FROM {settings.embedding_table} e
                    JOIN posts p ON p.id = e.post_id
                    WHERE p.project_id = :p
                    ORDER BY e.post_id
                    """  # noqa: S608 - table name comes from config, not from input
                ),
                {"p": resolved},
            ).all()

            if len(rows) < 50:
                return {
                    "skipped": True,
                    "reason": (
                        f"only {len(rows)} embedded posts; HDBSCAN needs a meaningful "
                        "population before its clusters mean anything. Run nlp.embed first."
                    ),
                }

            report_progress(job_id, 0.2)
            matrix = np.array([_parse_vector(row.embedding) for row in rows], dtype="float32")

            run_id = uuid.uuid4()
            started = utcnow()
            clusterer = NarrativeClusterer()
            assignments = clusterer.fit_predict(matrix)
            labels = assignments["labels"] if isinstance(assignments, dict) else assignments
            probabilities = (
                assignments.get("probabilities") if isinstance(assignments, dict) else None
            )
            report_progress(job_id, 0.6)

            session.execute(
                text(
                    """
                    INSERT INTO clustering_runs (
                        id, project_id, algorithm, params, embedding_model,
                        post_count, narrative_count, status, started_at, finished_at
                    ) VALUES (
                        :id, :project_id, 'hdbscan', cast(:params AS jsonb), :model,
                        :post_count, :narrative_count, 'succeeded', :started, now()
                    )
                    """
                ),
                {
                    "id": run_id,
                    "project_id": resolved,
                    "params": json.dumps(getattr(clusterer, "config", {}), default=str),
                    "model": settings.embedding_model,
                    "post_count": len(rows),
                    # -1 is HDBSCAN's noise label: posts that belong to no
                    # cluster. Counting it as a narrative would create one
                    # enormous "narrative" of everything unassigned.
                    "narrative_count": len({int(v) for v in labels if int(v) >= 0}),
                    "started": started,
                },
            )

            members: dict[int, list[tuple[str, float]]] = {}
            for index, row in enumerate(rows):
                label = int(labels[index])
                if label < 0:
                    continue
                confidence = float(probabilities[index]) if probabilities is not None else 1.0
                members.setdefault(label, []).append((row.post_id, confidence))

            written = _persist_clusters(session, resolved, run_id, members)
            report_progress(job_id, 0.95)

    result = {
        "clustering_run_id": str(run_id),
        "posts": len(rows),
        "narratives": len(members),
        "noise_posts": len(rows) - sum(len(v) for v in members.values()),
        **written,
    }

    # Labelling is a separate queue: it is LLM-bound where this is CPU-bound,
    # and a summarization failure must not invalidate a clustering that worked.
    celery.send_task("nlp.summarize", kwargs={"project_id": project_id}, queue="llm")

    report_progress(job_id, 1.0, result=result)
    return result


def _persist_clusters(session, project_id, run_id, members: dict) -> dict:
    """Write narratives and membership, preserving analyst edits."""
    from sqlalchemy import text

    from app.etl.scored_loader import narrative_uuid

    narrative_rows, membership_rows = [], []
    for cluster_id, posts in members.items():
        narrative_id = narrative_uuid(project_id, f"cluster-{run_id}-{cluster_id}")
        narrative_rows.append(
            {
                "id": narrative_id,
                "project_id": project_id,
                "cluster_id": cluster_id,
                "run_id": run_id,
                "post_count": len(posts),
            }
        )
        representatives = sorted(posts, key=lambda item: -item[1])[:5]
        representative_ids = {post_id for post_id, _ in representatives}
        membership_rows.extend(
            {
                "narrative_id": narrative_id,
                "post_id": post_id,
                "membership_score": confidence,
                "is_representative": post_id in representative_ids,
            }
            for post_id, confidence in posts
        )

    session.execute(
        text(
            """
            INSERT INTO narratives (
                id, project_id, cluster_id, clustering_run_id, post_count,
                title_generated_by, summary_generated_by
            ) VALUES (
                :id, :project_id, :cluster_id, :run_id, :post_count, 'heuristic', 'heuristic'
            )
            ON CONFLICT (id) DO UPDATE SET
                cluster_id = EXCLUDED.cluster_id,
                clustering_run_id = EXCLUDED.clustering_run_id,
                post_count = EXCLUDED.post_count,
                updated_at = now()
            -- A hand-built narrative is never touched by a clustering run, and
            -- an analyst's title is preserved because this statement does not
            -- write the title column at all on conflict.
            WHERE NOT narratives.is_manual
            """
        ),
        narrative_rows,
    )
    for start in range(0, len(membership_rows), 5000):
        session.execute(
            text(
                """
                INSERT INTO narrative_posts (
                    narrative_id, post_id, membership_score, is_representative
                ) VALUES (:narrative_id, :post_id, :membership_score, :is_representative)
                ON CONFLICT (narrative_id, post_id) DO UPDATE SET
                    membership_score = EXCLUDED.membership_score,
                    is_representative = EXCLUDED.is_representative
                """
            ),
            membership_rows[start : start + 5000],
        )

    # Counts and window recomputed from the corpus, not carried from the model.
    session.execute(
        text(
            """
            UPDATE narratives n SET
                author_count = sub.authors,
                date_start = sub.first_ts,
                date_end = sub.last_ts,
                engagement_total = COALESCE(sub.engagement, 0),
                platforms = sub.platforms
            FROM (
                SELECT np.narrative_id,
                       count(DISTINCT p.author_id) AS authors,
                       min(p."timestamp") AS first_ts,
                       max(p."timestamp") AS last_ts,
                       sum(COALESCE(p.likes,0) + COALESCE(p.shares,0)
                           + COALESCE(p.replies,0)) AS engagement,
                       array_agg(DISTINCT p.source) AS platforms
                FROM narrative_posts np JOIN posts p ON p.id = np.post_id
                WHERE p.project_id = :p
                GROUP BY np.narrative_id
            ) sub
            WHERE n.id = sub.narrative_id
            """
        ),
        {"p": project_id},
    )
    return {"narratives_written": len(narrative_rows), "memberships": len(membership_rows)}


def _parse_vector(value) -> list[float]:
    """pgvector round-trips as a string unless the pgvector type is registered."""
    if isinstance(value, str):
        return [float(v) for v in value.strip("[]").split(",") if v]
    return list(value)


@celery.task(name="nlp.summarize", base=TrackedTask, bind=True)
def summarize(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Label narratives. One LLM call per cluster, never per post.

    Falls back to centroid keywords when no LLM is configured, and records
    `generated_by = 'heuristic'` when it does. That distinction is not cosmetic:
    the UI renders its `✨ Generated by AI` pill from it, and putting that pill
    on a keyword join would claim a model wrote something it did not.
    """
    from sqlalchemy import text

    from app.db import advisory_lock, sync_session
    from app.etl.service import resolve_project
    from nlp.adapters import get_summarizer

    summarizer = get_summarizer()
    labelled = 0

    with sync_session() as session:
        with advisory_lock(session, f"nlp.summarize:{project_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another summarize run holds the lock"}

            resolved = resolve_project(session, project_id)
            pending = session.execute(
                text(
                    """
                    SELECT n.id,
                           array_agg(p."text" ORDER BY np.membership_score DESC NULLS LAST)
                               FILTER (WHERE np.is_representative) AS samples
                    FROM narratives n
                    JOIN narrative_posts np ON np.narrative_id = n.id
                    JOIN posts p ON p.id = np.post_id
                    WHERE n.project_id = :p
                      AND n.title IS NULL
                      -- Never relabel what an analyst wrote.
                      AND NOT (n.edited_by_user AND 'title' = ANY(n.edited_fields))
                      AND NOT n.is_manual
                    GROUP BY n.id
                    """
                ),
                {"p": resolved},
            ).all()

            if not pending:
                return {"labelled": 0, "note": "every narrative already has a title"}

            for index, row in enumerate(pending):
                samples = [s for s in (row.samples or []) if s][:5]
                if not samples:
                    continue
                title, summary, source = _label(summarizer, samples)
                session.execute(
                    text(
                        """
                        UPDATE narratives SET
                            title = :title, summary = :summary,
                            title_generated_by = :source, summary_generated_by = :source,
                            summary_model = :model, summary_generated_at = now(),
                            updated_at = now()
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": row.id,
                        "title": title,
                        "summary": summary,
                        "source": source,
                        "model": summarizer.info.name if summarizer else None,
                    },
                )
                labelled += 1
                if index % 5 == 0:
                    report_progress(job_id, 0.1 + 0.85 * index / max(len(pending), 1))

    result = {
        "labelled": labelled,
        "generated_by": "ai" if summarizer else "heuristic",
        "note": (
            None
            if summarizer
            else "No ANTHROPIC_API_KEY, so labels are centroid keywords and are marked "
            "`heuristic`. The UI will not show an AI pill on them."
        ),
    }
    report_progress(job_id, 1.0, result=result)
    return result


def _label(summarizer, samples: list[str]) -> tuple[str, str | None, str]:
    """One cluster's title and summary, and how it was produced."""
    if summarizer is not None:
        try:
            result = summarizer.summarize(samples)
            title = (result.get("label") or result.get("title") or "").strip()
            if title:
                return title[:512], (result.get("summary") or "").strip() or None, "ai"
        except Exception as exc:
            # A provider outage degrades the label, it does not fail the run.
            log.warning("summarizer failed, falling back to keywords: %s", exc)

    from modeling.text.summarize import centroid_label

    return centroid_label(samples[0]), None, "heuristic"
