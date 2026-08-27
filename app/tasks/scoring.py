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
    """Score posts that do not already carry the current model versions.

    **Resumable and idempotent, and that is the whole design.** The task asks
    which post ids already carry the current `scoring_version` and skips them, so
    a worker killed halfway through resumes from where it stopped rather than
    rescoring the corpus. Killing it mid-batch and restarting cannot
    double-count, because every write is an upsert keyed on `post_id` and the
    score is a function of the text, not an increment.

    Batches commit as they go. A single transaction over the whole corpus would
    mean a worker killed at 95% loses everything, which is exactly the case
    resumability is for.
    """
    from sqlalchemy import text

    from app.db import advisory_lock, sync_session
    from app.etl.service import resolve_project
    from nlp import availability

    if params.get("all_projects"):
        return _fan_out_all_projects("score.posts")

    batch_size = int(params.get("batch_size", 256))
    rescore_all = bool(params.get("force"))

    capability = availability.get("aux")
    if not capability.available:
        # Not a crash. The job row records that the capability is absent and
        # names it, which is a far more useful failure than a traceback.
        return {
            "skipped": True,
            "reason": f"the 'aux' scorers are unavailable: {capability.detail}",
        }

    from nlp.adapters import get_post_scorer

    scorer = get_post_scorer()
    if scorer is None:
        return {
            "skipped": True,
            "reason": (
                "No live post scorer is mounted. Phase 2/3's committed scores are "
                "already served; load them with `app.etl.cli load-scores`. This task "
                "exists to score NEW posts and needs a checkpoint."
            ),
        }

    version = scorer.info.version
    scored = skipped = 0

    with sync_session() as session:
        with advisory_lock(session, f"score.posts:{project_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another scoring run holds the lock"}
            resolved = resolve_project(session, project_id)

    while True:
        with sync_session() as session:
            resolved = resolve_project(session, project_id)
            pending = session.execute(
                text(
                    """
                    SELECT p.id, p."text", p.lang
                    FROM posts p
                    LEFT JOIN post_scores s ON s.post_id = p.id
                    WHERE p.project_id = :p
                      AND (:force OR s.post_id IS NULL OR s.scoring_version IS DISTINCT FROM :v)
                    ORDER BY p.id
                    LIMIT :limit
                    """
                ),
                {"p": resolved, "v": version, "force": rescore_all, "limit": batch_size},
            ).all()

            if not pending:
                break

            rows = scorer.predict([row.text for row in pending])
            payload = []
            for post, result in zip(pending, rows, strict=True):
                if result.get("skip_reason"):
                    # A row the scorer declined is written with its reason, not
                    # left absent. Absent is indistinguishable from "not reached
                    # yet" and would make the resume query pick it up forever.
                    skipped += 1
                payload.append(_score_row(post.id, result, version))

            session.execute(text(_UPSERT_POST_SCORES), payload)
            scored += len(payload)

        report_progress(job_id, min(0.95, scored / max(scored + batch_size, 1)))

        # `force` rescores everything once; without this the loop would re-select
        # the same rows forever because they now carry the current version.
        if rescore_all:
            rescore_all = False

    result = {"scored": scored, "skipped_by_model": skipped, "scoring_version": version}
    report_progress(job_id, 1.0, result=result)
    return result


_UPSERT_POST_SCORES = """
INSERT INTO post_scores (
    post_id, toxicity, is_toxic, anomaly, is_anomalous, misinfo_likelihood,
    stance, stance_confidence, sentiment, sentiment_score, emotion, emotion_scores,
    skip_reasons, scoring_version, scored_at
) VALUES (
    :post_id, :toxicity, :is_toxic, :anomaly, :is_anomalous, :misinfo_likelihood,
    :stance, :stance_confidence, :sentiment, :sentiment_score, :emotion,
    cast(:emotion_scores AS jsonb), :skip_reasons, :scoring_version, :scored_at
)
ON CONFLICT (post_id) DO UPDATE SET
    toxicity = EXCLUDED.toxicity, is_toxic = EXCLUDED.is_toxic,
    anomaly = EXCLUDED.anomaly, is_anomalous = EXCLUDED.is_anomalous,
    misinfo_likelihood = EXCLUDED.misinfo_likelihood,
    stance = EXCLUDED.stance, stance_confidence = EXCLUDED.stance_confidence,
    sentiment = EXCLUDED.sentiment, sentiment_score = EXCLUDED.sentiment_score,
    emotion = EXCLUDED.emotion, emotion_scores = EXCLUDED.emotion_scores,
    skip_reasons = EXCLUDED.skip_reasons, scoring_version = EXCLUDED.scoring_version,
    scored_at = EXCLUDED.scored_at
"""


def _score_row(post_id: str, result: dict, version: str) -> dict:
    """One scorer output as an upsert row. Nulls stay null."""
    from app.db import utcnow

    emotion_scores = result.get("emotion_scores") or {}
    toxicity = result.get("toxicity")
    anomaly = result.get("anomaly")
    return {
        "post_id": post_id,
        "toxicity": toxicity,
        "is_toxic": None if toxicity is None else toxicity > 0.6,
        "anomaly": anomaly,
        "is_anomalous": None if anomaly is None else anomaly > 0.7,
        "misinfo_likelihood": result.get("misinfo_likelihood"),
        "stance": result.get("stance"),
        "stance_confidence": result.get("stance_confidence"),
        "sentiment": result.get("sentiment"),
        "sentiment_score": result.get("sentiment_score"),
        "emotion": result.get("emotion"),
        "emotion_scores": json.dumps(emotion_scores) if emotion_scores else None,
        "skip_reasons": ([result["skip_reason"]] if result.get("skip_reason") else []),
        "scoring_version": version,
        "scored_at": utcnow(),
    }


@celery.task(name="score.authors", base=TrackedTask, bind=True)
def score_authors(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Bot scores, per-feature attribution, and cohort assignment.

    Cohort assignment runs here rather than as its own task because it reads the
    same author facts and is far cheaper than the round trip of scheduling it
    separately. It is idempotent and never overwrites an analyst's assignment.
    """
    from app.db import advisory_lock, sync_session
    from app.etl.service import resolve_project
    from app.scoring import cohorts as cohort_assigner

    if params.get("all_projects"):
        return _fan_out_all_projects("score.authors")

    result: dict[str, Any] = {}

    with sync_session() as session:
        with advisory_lock(session, f"score.authors:{project_id}") as acquired:
            if not acquired:
                return {"skipped": True, "reason": "another author-scoring run holds the lock"}

            resolved = resolve_project(session, project_id)
            report_progress(job_id, 0.2)

            # Cohorts first: they need no model and are useful even when no bot
            # checkpoint is mounted, so a deployment without one still gets a
            # populated Actors page.
            result["cohorts"] = cohort_assigner.assign(session, resolved)
            report_progress(job_id, 0.5)

            result["bot_scores"] = _score_bots(session, resolved)

    report_progress(job_id, 1.0, result=result)
    return result


def _score_bots(session, project_id) -> dict:
    """Run the bot classifier over authors that lack a current score."""
    from sqlalchemy import text

    from nlp import availability

    capability = availability.get("bot")
    if not capability.available:
        return {"skipped": True, "reason": capability.detail}

    from nlp.adapters import get_bot_scorer

    scorer = get_bot_scorer()
    if scorer is None:
        return {
            "skipped": True,
            "reason": (
                "No live bot classifier is mounted. Phase 2/3's committed author "
                "scores are already served via `app.etl.cli load-scores`."
            ),
        }

    version = scorer.info.version
    rows = session.execute(
        text(
            """
            SELECT author_id, source, followers, following, post_count,
                   created_at_source, first_seen, last_seen
            FROM authors
            WHERE project_id = :p AND scoring_version IS DISTINCT FROM :v
            """
        ),
        {"p": project_id, "v": version},
    ).all()
    if not rows:
        return {"scored": 0, "note": "every author already carries the current version"}

    # News and GDELT "authors" are outlets. Scoring them would produce a
    # confident number about a thing that is not an account; skipping them with
    # a reason code is the honest output, and /authors/{id}/score renders it.
    scorable = [r for r in rows if r.source not in {"news", "gdelt"}]
    outlets = [r for r in rows if r.source in {"news", "gdelt"}]

    if outlets:
        session.execute(
            text(
                """
                UPDATE authors SET
                    bot_score = NULL,
                    skip_reasons = ARRAY['author_is_outlet'],
                    scoring_version = :v
                WHERE author_id = ANY(:ids) AND project_id = :p
                """
            ),
            {"ids": [r.author_id for r in outlets], "v": version, "p": project_id},
        )

    if not scorable:
        return {"scored": 0, "skipped_outlets": len(outlets)}

    features = [
        {
            "author_id": r.author_id,
            "followers": r.followers,
            "following": r.following,
            "post_count": r.post_count,
            # Age *at first post*, not age today: "registered three days before
            # it started posting" is the signal, not "registered in 2019".
            "account_age_days": (
                (r.first_seen - r.created_at_source).days
                if r.created_at_source and r.first_seen
                else None
            ),
            "active_days": (
                (r.last_seen - r.first_seen).days if r.last_seen and r.first_seen else None
            ),
        }
        for r in scorable
    ]
    predictions = scorer.score(features)

    payload = [
        {
            "author_id": feature["author_id"],
            "bot_score": prediction.get("bot_score"),
            "score_components": json.dumps(
                {"top_features": prediction.get("top_features", []), "features": feature},
                default=str,
            ),
            "scoring_version": version,
            "project_id": project_id,
        }
        for feature, prediction in zip(features, predictions, strict=True)
    ]
    for start in range(0, len(payload), 2000):
        session.execute(
            text(
                """
                UPDATE authors SET
                    bot_score = :bot_score,
                    score_components = cast(:score_components AS jsonb),
                    scoring_version = :scoring_version,
                    skip_reasons = '{}'
                WHERE author_id = :author_id AND project_id = :project_id
                """
            ),
            payload[start : start + 2000],
        )
    return {"scored": len(payload), "skipped_outlets": len(outlets), "version": version}


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
