"""Scoring tasks.

Post scores, author scores and the narrative fusion scorecards.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.tasks.base import TrackedTask, report_progress
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="score.posts", base=TrackedTask, bind=True)
def score_posts(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Toxicity, anomaly, misinfo, stance, sentiment, emotion."""
    raise NotImplementedError(
        "score.posts lands at build step 9. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )


@celery.task(name="score.authors", base=TrackedTask, bind=True)
def score_authors(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Bot classifier plus feature attribution."""
    raise NotImplementedError(
        "score.authors lands at build step 9. The job row records this failure, so the "
        "frontend sees a failed job rather than a request that hangs."
    )


@celery.task(name="score.fusion", base=TrackedTask, bind=True)
def score_fusion(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Compute every narrative's scorecard for a project.

    Whole-project, not per-narrative, and that is forced by the design: the
    normalization is percentile-rank *within the project*, so a single
    narrative's score is undefined without the rest of the distribution. Scoring
    one narrative in isolation would silently fall back to raw values and
    produce a number that does not match what the same narrative gets on the
    next full run.
    """
    from sqlalchemy import text

    from app.db import advisory_lock, sync_session, utcnow
    from app.scoring import signals as signal_source
    from app.scoring.fusion import Normalizers, fusion_score, load_config

    if params.get("all_projects"):
        return _fan_out_all_projects("score.fusion")

    config = load_config()

    with sync_session() as session:
        with advisory_lock(session, f"score.fusion:{project_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another fusion run holds the lock"}

            from app.etl.service import resolve_project

            resolved = resolve_project(session, project_id)
            report_progress(job_id, 0.1)

            collected = signal_source.collect(session, resolved, config)
            if not collected:
                return {"narratives": 0, "note": "no narratives to score"}

            normalizers = Normalizers.build(signal_source.populations(collected), config)
            report_progress(job_id, 0.3)

            payload, distribution = [], {"high": 0, "medium": 0, "low": 0, "unscorable": 0}
            for index, (narrative_id, item) in enumerate(collected.items()):
                result = fusion_score(item, config=config, normalizers=normalizers)
                distribution["unscorable" if result.score is None else result.priority] += 1
                payload.append(
                    {
                        "narrative_id": narrative_id,
                        "priority": result.priority,
                        "bot_like": item.bot_like_ratio,
                        "anomalous": item.temporal_burstiness,
                        "toxicity": item.toxicity,
                        "compass_risk": _band(item.compass_status, config),
                        "negative_sentiment": item.negative_sentiment,
                        "fusion_score": result.score,
                        "components": json.dumps(result.components, default=str),
                        "scoring_version": result.scoring_version,
                        "computed_at": utcnow(),
                    }
                )
                if index % 25 == 0:
                    report_progress(job_id, 0.3 + 0.6 * index / max(len(collected), 1))

            session.execute(
                text(
                    """
                    INSERT INTO narrative_scorecards (
                        narrative_id, priority, bot_like, anomalous, toxicity,
                        compass_risk, negative_sentiment, fusion_score, components,
                        scoring_version, computed_at
                    ) VALUES (
                        cast(:narrative_id AS uuid), :priority, :bot_like, :anomalous,
                        :toxicity, :compass_risk, :negative_sentiment, :fusion_score,
                        cast(:components AS jsonb), :scoring_version, :computed_at
                    )
                    ON CONFLICT (narrative_id) DO UPDATE SET
                        priority = EXCLUDED.priority,
                        bot_like = EXCLUDED.bot_like,
                        anomalous = EXCLUDED.anomalous,
                        toxicity = EXCLUDED.toxicity,
                        compass_risk = EXCLUDED.compass_risk,
                        negative_sentiment = EXCLUDED.negative_sentiment,
                        fusion_score = EXCLUDED.fusion_score,
                        components = EXCLUDED.components,
                        scoring_version = EXCLUDED.scoring_version,
                        computed_at = EXCLUDED.computed_at
                    """
                ),
                payload,
            )

    result = {
        "narratives": len(payload),
        "scoring_version": config["version"],
        "priority_distribution": distribution,
    }
    report_progress(job_id, 1.0, result=result)
    return result


def _band(status: str | None, config: dict) -> str | None:
    """Compass status -> the high/medium/low band the scorecard shows."""
    if not status:
        return None
    value = config.get("compass_risk_values", {}).get(status)
    if value is None:
        return None
    return "high" if value >= 0.6 else ("medium" if value >= 0.35 else "low")


def _fan_out_all_projects(task_name: str) -> dict:
    """Beat schedules pass all_projects; expand to one job per project.

    Beat cannot know which projects exist, and a single task that loops over all
    of them makes one slow project delay every other. One job each keeps them
    independent and individually visible in /jobs.
    """
    from sqlalchemy import select

    from app.db import sync_session
    from app.models.core import Project

    with sync_session() as session:
        slugs = [row for (row,) in session.execute(select(Project.slug)).all()]
    for slug in slugs:
        celery.send_task(task_name, kwargs={"project_id": slug})
    return {"fanned_out": slugs}
