"""Alert rule evaluation.

Runs every 15 minutes from beat. The cooldown is the part that matters: without
it, a narrative hovering on a threshold fires on every cycle, the analyst stops
reading alerts, and the feature is worse than not having it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.tasks.base import TrackedTask, report_progress
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)

#: metric -> (SQL expression, subject id, subject label). Deliberately a fixed
#: mapping rather than a user-supplied expression: a rule DSL evaluated against
#: the corpus on a schedule is a small interpreter with an injection surface,
#: and this covers every rule the product asks for.
_METRICS = {
    "fusion_score": ("sc.fusion_score", "narrative"),
    "bot_like_ratio": ("sc.bot_like", "narrative"),
    "toxicity": ("sc.toxicity", "narrative"),
    "negative_sentiment": ("sc.negative_sentiment", "narrative"),
    "anomalous": ("sc.anomalous", "narrative"),
    "post_count": ("n.post_count::float", "narrative"),
    "author_count": ("n.author_count::float", "narrative"),
}

_OPERATORS = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "==": "=", "!=": "<>"}


@celery.task(name="alerts.evaluate", base=TrackedTask, bind=True)
def evaluate(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Evaluate every enabled rule and fire past the cooldown."""
    from sqlalchemy import text

    from app.db import sync_session, utcnow

    fired = evaluated = suppressed = 0
    errors: list[str] = []

    with sync_session() as session:
        scope = "AND r.project_id = :p" if project_id else ""
        params_sql: dict[str, Any] = {}
        if project_id:
            from app.etl.service import resolve_project

            params_sql["p"] = resolve_project(session, project_id)

        rules = session.execute(
            text(
                f"SELECT r.* FROM alert_rules r WHERE r.enabled {scope}"  # noqa: S608 - fixed
            ),
            params_sql,
        ).all()

        for index, rule in enumerate(rules):
            evaluated += 1
            try:
                outcome = _evaluate_rule(session, rule)
                fired += outcome["fired"]
                suppressed += outcome["suppressed"]
            except Exception as exc:
                # One bad rule must not stop the other rules from evaluating.
                # A rule that always errors is a configuration problem, and the
                # alerts that would have fired are the ones somebody needs.
                log.exception("alert rule %s failed", rule.id)
                errors.append(f"{rule.name}: {type(exc).__name__}: {exc}"[:300])

            session.execute(
                text("UPDATE alert_rules SET last_evaluated_at = :now WHERE id = :id"),
                {"now": utcnow(), "id": rule.id},
            )
            report_progress(job_id, 0.1 + 0.85 * (index + 1) / max(len(rules), 1))

    result = {
        "rules_evaluated": evaluated,
        "alerts_fired": fired,
        "suppressed_by_cooldown": suppressed,
        "errors": errors,
    }
    report_progress(job_id, 1.0, result=result)
    return result


def _evaluate_rule(session, rule) -> dict[str, int]:
    from sqlalchemy import text

    from app.db import utcnow

    condition = rule.condition or {}
    metric = condition.get("metric")
    operator = condition.get("op")
    value = condition.get("value")

    if metric not in _METRICS or operator not in _OPERATORS:
        raise ValueError(
            f"unsupported condition: metric={metric!r} op={operator!r}. "
            f"Supported metrics: {sorted(_METRICS)}"
        )

    expression, subject_type = _METRICS[metric]
    sql_operator = _OPERATORS[operator]

    matches = session.execute(
        text(
            f"""
            SELECT n.id, n.title, {expression} AS observed, sc.scoring_version
            FROM narratives n
            LEFT JOIN narrative_scorecards sc ON sc.narrative_id = n.id
            WHERE n.project_id = :p
              AND {expression} IS NOT NULL
              AND {expression} {sql_operator} :threshold
            """  # noqa: S608 - expression and operator come from fixed dicts
        ),
        {"p": rule.project_id, "threshold": float(value)},
    ).all()

    fired = suppressed = 0
    for match in matches:
        # The cooldown is per rule *and per subject*. A rule that fired for
        # narrative A ten minutes ago should still be able to fire for narrative
        # B, which a rule-level cooldown would wrongly suppress.
        recent = session.execute(
            text(
                """
                SELECT 1 FROM alerts
                WHERE rule_id = :rule_id AND subject_id = :subject_id
                  AND triggered_at > now() - make_interval(mins => :cooldown)
                LIMIT 1
                """
            ),
            {
                "rule_id": rule.id,
                "subject_id": str(match.id),
                "cooldown": rule.cooldown_minutes or 0,
            },
        ).scalar()
        if recent:
            suppressed += 1
            continue

        session.execute(
            text(
                """
                INSERT INTO alerts (
                    rule_id, project_id, narrative_id, subject_type, subject_id,
                    triggered_at, payload
                ) VALUES (
                    :rule_id, :project_id, :narrative_id, :subject_type, :subject_id,
                    :now, cast(:payload AS jsonb)
                )
                """
            ),
            {
                "rule_id": rule.id,
                "project_id": rule.project_id,
                "narrative_id": match.id,
                "subject_type": subject_type,
                "subject_id": str(match.id),
                "now": utcnow(),
                # The alert explains itself: the value, the condition, and the
                # scoring version behind it. An alert that says only "narrative
                # X tripped a rule" sends the reader hunting.
                "payload": json.dumps(
                    {
                        "metric": metric,
                        "op": operator,
                        "threshold": value,
                        "observed": (
                            round(float(match.observed), 4) if match.observed is not None else None
                        ),
                        "narrative_title": match.title,
                        "scoring_version": match.scoring_version,
                    },
                    default=str,
                ),
            },
        )
        fired += 1

    if fired:
        session.execute(
            text(
                """
                UPDATE alert_rules
                SET last_triggered_at = :now, trigger_count = trigger_count + :n
                WHERE id = :id
                """
            ),
            {"now": utcnow(), "n": fired, "id": rule.id},
        )

    return {"fired": fired, "suppressed": suppressed}
