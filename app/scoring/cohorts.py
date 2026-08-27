"""Cohort assignment.

Multi-label by construction: every rule is evaluated against every author and
each match is its own row. An author with no matches gets no rows, which is
different from an author in a "General" bucket -- a catch-all cohort would make
the cohort bar chart meaningless by putting most of the corpus in one bar.

Two things this module will not do:

* **Overwrite an analyst.** Rows with ``assigned_by = 'analyst'`` are left
  alone. The whole reason cohorts and author groups are separate tables is that
  one is a model's guess and the other is a person's assertion, and a model run
  that silently reverses a person is the failure that distinction exists to
  prevent.
* **Assert more than the rule can show.** Keyword rules are capped at low
  confidence in ``configs/cohorts.yaml`` because a journalist writing about
  cryptocurrency matches the same terms as a promoter.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
    from app.config import load_yaml

    config = load_yaml("cohorts")
    if not config:
        raise RuntimeError(
            "configs/cohorts.yaml is missing or empty. The taxonomy is configuration, "
            "so there is deliberately nothing to fall back to."
        )
    return config


def ensure_cohorts(session: Session, project_id: uuid.UUID, config: dict) -> dict[str, uuid.UUID]:
    """Create the taxonomy's rows for a project, idempotently."""
    ids: dict[str, uuid.UUID] = {}
    for cohort in config["cohorts"]:
        row = session.execute(
            text(
                """
                INSERT INTO cohorts (project_id, name, description, category)
                VALUES (:project_id, :name, :description, :category)
                ON CONFLICT (project_id, name) DO UPDATE SET
                    description = EXCLUDED.description,
                    category = EXCLUDED.category
                RETURNING id
                """
            ),
            {
                "project_id": project_id,
                "name": cohort["name"],
                "description": (cohort.get("description") or "").strip(),
                "category": cohort["category"],
            },
        ).scalar_one()
        ids[cohort["name"]] = row
    return ids


def _matches_metric(author: Any, rule: dict) -> bool:
    field = rule["field"]

    if field == "source":
        return author.source in rule.get("equals", ())

    if field == "account_age_days":
        # Age at first post, not age today. "Registered three days before it
        # started posting about the election" is the signal; "registered in
        # 2019" tells you nothing about a 2026 corpus.
        if author.created_at_source is None or author.first_seen is None:
            return False
        value = (author.first_seen - author.created_at_source).days
    else:
        value = getattr(author, field, None)

    # Null is not zero. An author whose follower count the platform never
    # exposed is not a Nano Account; it is unmeasured, and asserting a tier
    # would be inventing the fact the tier is about.
    if value is None:
        return False
    if "min" in rule and value < rule["min"]:
        return False
    return not ("max" in rule and value >= rule["max"])


def assign(session: Session, project_id: uuid.UUID, *, config: dict | None = None) -> dict:
    """Assign every author to every cohort whose rule it satisfies."""
    config = config or load_config()
    cohort_ids = ensure_cohorts(session, project_id, config)
    min_confidence = float(config.get("min_confidence", 0.35))

    authors = session.execute(
        text(
            """
            SELECT author_id, source, followers, post_count, created_at_source, first_seen
            FROM authors WHERE project_id = :p
            """
        ),
        {"p": project_id},
    ).all()
    if not authors:
        return {"authors": 0, "assignments": 0, "cohorts": len(cohort_ids)}

    keyword_rules = [c for c in config["cohorts"] if c["rule"]["type"] == "keyword"]
    hits = _keyword_hits(session, project_id, keyword_rules) if keyword_rules else {}

    payload = []
    for author in authors:
        for cohort in config["cohorts"]:
            rule = cohort["rule"]
            confidence = float(cohort.get("confidence", 0.5))
            if confidence < min_confidence:
                continue

            if rule["type"] == "metric":
                matched = _matches_metric(author, rule)
            else:
                matched = hits.get((author.author_id, cohort["name"]), 0) >= rule.get("min_hits", 3)
            if matched:
                payload.append(
                    {
                        "author_id": author.author_id,
                        "cohort_id": cohort_ids[cohort["name"]],
                        "confidence": confidence,
                    }
                )

    if payload:
        for start in range(0, len(payload), 5000):
            session.execute(
                text(
                    """
                    INSERT INTO author_cohorts (author_id, cohort_id, confidence, assigned_by,
                                                assigned_at)
                    VALUES (:author_id, :cohort_id, :confidence, 'model', now())
                    ON CONFLICT (author_id, cohort_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        assigned_at = now()
                    -- An analyst's assignment is never touched by a model run.
                    WHERE author_cohorts.assigned_by <> 'analyst'
                    """
                ),
                payload[start : start + 5000],
            )

    # Denormalised counts for the cohort list page, recomputed rather than
    # incremented so a partial run cannot leave them drifting.
    session.execute(
        text(
            """
            UPDATE cohorts c SET
                author_count = COALESCE(sub.authors, 0),
                post_pct = sub.post_pct
            FROM (
                SELECT ac.cohort_id,
                       count(DISTINCT ac.author_id) AS authors,
                       100.0 * sum(a.post_count) / NULLIF(
                           (SELECT sum(post_count) FROM authors WHERE project_id = :p), 0
                       ) AS post_pct
                FROM author_cohorts ac
                JOIN authors a ON a.author_id = ac.author_id AND a.project_id = :p
                GROUP BY ac.cohort_id
            ) sub
            WHERE c.id = sub.cohort_id AND c.project_id = :p
            """
        ),
        {"p": project_id},
    )

    log.info("cohorts: %d assignments over %d authors", len(payload), len(authors))
    return {"authors": len(authors), "assignments": len(payload), "cohorts": len(cohort_ids)}


def _keyword_hits(
    session: Session, project_id: uuid.UUID, rules: list[dict]
) -> dict[tuple[str, str], int]:
    """Count term hits per author per keyword cohort, in one pass.

    One query per rule rather than per author: 2,000 authors times four rules is
    8,000 round trips, and the same answer is four indexed aggregates.
    """
    out: dict[tuple[str, str], int] = {}
    for rule in rules:
        terms = rule["rule"]["terms"]
        rows = session.execute(
            text(
                """
                SELECT p.author_id, count(*) AS hits
                FROM posts p
                WHERE p.project_id = :p
                  AND p."text" ILIKE ANY(:patterns)
                GROUP BY p.author_id
                """
            ),
            {"p": project_id, "patterns": [f"%{term}%" for term in terms]},
        ).all()
        for row in rows:
            out[(row.author_id, rule["name"])] = row.hits
    return out
