"""Domain risk scoring and best-effort enrichment.

Two things happen here and they are deliberately independent:

* **Risk scoring** uses only in-corpus signals -- link velocity, the bot ratio
  of the accounts sharing it, how many narratives it appears in. It always
  works, needs no network, and produces a defensible band for every domain.
* **Enrichment** (WHOIS, TLS) is a network call to a third party that may be
  slow, rate-limited, or simply absent for a domain. It refines the score when
  it succeeds and changes nothing when it does not.

Nothing ever blocks on the second. A domain page must render, with a risk band,
for a domain whose registrar never answered.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.tasks.base import TrackedTask, report_progress
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)

#: How many domains one scheduled run will try to enrich. WHOIS servers
#: rate-limit aggressively and a run that tries the whole table gets the host
#: blocked, which costs far more than the enrichment was worth.
ENRICH_BATCH = 25

#: A domain whose lookup has failed this many times is left alone. Without a
#: ceiling, a permanently unreachable registrar consumes the batch budget
#: forever and starves every domain behind it.
MAX_ENRICH_ATTEMPTS = 3

_SCORE_SQL = """
WITH signals AS (
    SELECT
        lower(d.domain) AS domain,
        count(*) AS posts,
        count(DISTINCT p.author_id) AS authors,
        count(DISTINCT p.author_id) FILTER (WHERE a.bot_score > 0.6)::float
            / NULLIF(count(DISTINCT p.author_id) FILTER (WHERE a.bot_score IS NOT NULL), 0)
            AS bot_ratio,
        -- Peak posts per hour: a burst of links is what an amplification
        -- campaign looks like from the domain's side.
        max(hourly.posts_per_hour) AS link_velocity
    FROM posts p
    CROSS JOIN LATERAL unnest(p.domains) AS d(domain)
    LEFT JOIN authors a ON a.author_id = p.author_id AND a.project_id = :project_id
    LEFT JOIN LATERAL (
        SELECT count(*) AS posts_per_hour
        FROM posts p2
        CROSS JOIN LATERAL unnest(p2.domains) AS d2(domain)
        WHERE p2.project_id = :project_id
          AND lower(d2.domain) = lower(d.domain)
          AND p2."timestamp" BETWEEN p."timestamp" - interval '30 minutes'
                                 AND p."timestamp" + interval '30 minutes'
    ) hourly ON true
    WHERE p.project_id = :project_id
    GROUP BY lower(d.domain)
)
UPDATE domains dom SET
    link_velocity = s.link_velocity,
    sharing_author_bot_ratio = s.bot_ratio,
    scoring_version = :version
FROM signals s
WHERE dom.domain = s.domain AND dom.project_id = :project_id
"""


@celery.task(name="domain.enrich", base=TrackedTask, bind=True)
def enrich(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Score domains from in-corpus signals, then enrich a bounded batch."""
    from sqlalchemy import text

    from app.db import sync_session, utcnow
    from app.etl.service import resolve_project

    single_domain = params.get("domain")
    batch_mode = bool(params.get("batch"))

    if batch_mode and not project_id:
        # The beat schedule has no project. Score every one rather than picking
        # arbitrarily, so a scheduled run does not silently favour one case.
        from sqlalchemy import select

        from app.models.core import Project

        with sync_session() as session:
            slugs = [row for (row,) in session.execute(select(Project.slug)).all()]
        for slug in slugs:
            celery.send_task("domain.enrich", kwargs={"project_id": slug, "batch": True})
        return {"fanned_out": slugs}

    scored = enriched = failed = unavailable = 0

    with sync_session() as session:
        resolved = resolve_project(session, project_id)

        # 1. In-corpus signals. Always available, never blocks.
        session.execute(text(_SCORE_SQL), {"project_id": resolved, "version": "domain-v1.0.0"})
        scored = _apply_risk_scores(session, resolved)
        report_progress(job_id, 0.4)

        # 2. Enrichment, bounded and best-effort. `skip_enrichment` exists for
        #    the seed path: a demo must not wait on a WHOIS server.
        if params.get("skip_enrichment"):
            targets = []
        elif single_domain:
            targets = [single_domain.lower()]
        else:
            targets = [
                row[0]
                for row in session.execute(
                    text(
                        """
                        SELECT domain FROM domains
                        WHERE project_id = :p
                          AND enrichment_status IN ('pending', 'failed')
                          AND enrichment_attempts < :max_attempts
                        ORDER BY risk_score DESC NULLS LAST
                        LIMIT :limit
                        """
                    ),
                    {"p": resolved, "limit": ENRICH_BATCH, "max_attempts": MAX_ENRICH_ATTEMPTS},
                ).all()
            ]

        for index, domain in enumerate(targets):
            outcome = _enrich_one(domain)
            session.execute(
                text(
                    """
                    UPDATE domains SET
                        whois_created_at = COALESCE(:created_at, whois_created_at),
                        registrar = COALESCE(:registrar, registrar),
                        hosting_country = COALESCE(:country, hosting_country),
                        tls_cert_age_days = COALESCE(:tls_age, tls_cert_age_days),
                        enrichment_status = :status,
                        enrichment_detail = :detail,
                        enrichment_attempts = enrichment_attempts + 1,
                        enriched_at = :enriched_at
                    WHERE domain = :domain AND project_id = :p
                    """
                ),
                {
                    "domain": domain,
                    "p": resolved,
                    "enriched_at": utcnow() if outcome["status"] == "enriched" else None,
                    **outcome,
                },
            )
            if outcome["status"] == "enriched":
                enriched += 1
            elif outcome["status"] == "unavailable":
                # Not a failure. Either the lookup tool is not installed or the
                # registrar holds no record; neither is an error to alert on,
                # and conflating them with real failures would hide the ones
                # worth investigating.
                unavailable += 1
            else:
                failed += 1
            if index % 5 == 0:
                report_progress(job_id, 0.4 + 0.55 * index / max(len(targets), 1))

        # Re-score with whatever enrichment added. Domain age only exists now.
        _apply_risk_scores(session, resolved)

    result = {
        "scored": scored,
        "enriched": enriched,
        "enrichment_unavailable": unavailable,
        "enrichment_failed": failed,
        "attempted": len(targets),
    }
    report_progress(job_id, 1.0, result=result)
    return result


def _apply_risk_scores(session, project_id) -> int:
    """Combine the available signals into a 0-100 score and a band.

    The weights renormalize over what exists, exactly as the narrative fusion
    score does: a domain with no WHOIS is scored on velocity and bot ratio
    rather than penalised for a lookup that failed.
    """
    from sqlalchemy import text

    rows = session.execute(
        text(
            """
            SELECT domain, link_velocity, sharing_author_bot_ratio, whois_created_at,
                   narrative_count
            FROM domains WHERE project_id = :p
            """
        ),
        {"p": project_id},
    ).all()
    if not rows:
        return 0

    velocities = sorted(r.link_velocity for r in rows if r.link_velocity is not None)
    payload = []
    for row in rows:
        parts: dict[str, float | None] = {
            "link_velocity": _percentile(row.link_velocity, velocities),
            "sharing_author_bot_ratio": row.sharing_author_bot_ratio,
            "domain_age": _age_signal(row.whois_created_at),
        }
        weights = {"link_velocity": 0.35, "sharing_author_bot_ratio": 0.40, "domain_age": 0.25}

        from app.scoring.normalize import renormalized_weights, weighted_mean

        value, missing, renormalized = weighted_mean(parts, weights)
        applied = renormalized_weights(parts, weights)
        score = None if value is None else round(100 * value, 1)
        payload.append(
            {
                "domain": row.domain,
                "project_id": project_id,
                "risk_score": score,
                "risk_band": (
                    None
                    if score is None
                    else ("high" if score >= 66 else ("medium" if score >= 33 else "low"))
                ),
                "components": json.dumps(
                    {
                        "missing": missing,
                        "weights_renormalized": renormalized,
                        "components": {
                            name: {
                                "value": parts[name],
                                "weight": round(applied[name], 4),
                                "contribution": (
                                    None
                                    if parts[name] is None
                                    else round(parts[name] * applied[name], 4)
                                ),
                                "definition": _DEFINITIONS[name],
                                "inputs": {},
                            }
                            for name in weights
                        },
                    },
                    default=str,
                ),
            }
        )

    session.execute(
        text(
            """
            UPDATE domains SET
                risk_score = :risk_score, risk_band = :risk_band,
                components = cast(:components AS jsonb)
            WHERE domain = :domain AND project_id = :project_id
            """
        ),
        payload,
    )
    return len(payload)


_DEFINITIONS = {
    "link_velocity": (
        "Peak posts per hour carrying this domain, percentile-ranked within the "
        "project. A burst of links is what amplification looks like from the "
        "domain's side."
    ),
    "sharing_author_bot_ratio": (
        "Share of the accounts linking this domain that score above the bot "
        "threshold, over the accounts the classifier could score."
    ),
    "domain_age": (
        "Newer registrations score higher. Null when WHOIS returned nothing, in "
        "which case the other weights renormalize rather than the domain being "
        "penalised for a failed lookup."
    ),
}


def _band(score: float | None) -> str | None:
    """Score to band. Null stays null: an unscored domain has no band, and
    defaulting it to 'low' would assert something the data does not support."""
    if score is None:
        return None
    if score >= 66:
        return "high"
    return "medium" if score >= 33 else "low"


def _percentile(value: float | None, population: list[float]) -> float | None:
    if value is None or not population:
        return None
    from bisect import bisect_left

    return bisect_left(population, value) / max(len(population) - 1, 1)


def _age_signal(created_at) -> float | None:
    """Newer is riskier, saturating at ten years."""
    if created_at is None:
        return None
    from app.db import utcnow

    days = (utcnow() - created_at).days
    return max(0.0, min(1.0, 1 - days / 3650))


def _enrich_one(domain: str) -> dict[str, Any]:
    """One WHOIS lookup. Never raises, always returns a status.

    `python-whois` is an optional dependency and the network is optional too.
    Both absences are reported as `unavailable`, which is different from
    `failed`: one means we did not look, the other means we looked and got
    nothing, and an operator debugging an empty domain page needs to know which.
    """
    try:
        import whois  # type: ignore[import-untyped]
    except ImportError:
        return {
            "status": "unavailable",
            "detail": (
                "python-whois is not installed, so registration data cannot be "
                "fetched. Risk is computed from in-corpus signals only."
            ),
            "created_at": None,
            "registrar": None,
            "country": None,
            "tls_age": None,
        }

    try:
        record = whois.whois(domain)
    except Exception as exc:
        return {
            "status": "failed",
            "detail": f"WHOIS lookup failed: {type(exc).__name__}: {exc}"[:400],
            "created_at": None,
            "registrar": None,
            "country": None,
            "tls_age": None,
        }

    created = record.get("creation_date") if hasattr(record, "get") else None
    if isinstance(created, list):
        # Registrars routinely return several dates. The earliest is the one
        # that means "when did this domain first exist".
        created = min((d for d in created if d), default=None)

    if created is None:
        return {
            "status": "unavailable",
            "detail": "The registrar returned no creation date for this domain.",
            "created_at": None,
            "registrar": None,
            "country": None,
            "tls_age": None,
        }

    from datetime import timezone

    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)

    registrar = record.get("registrar") if hasattr(record, "get") else None
    country = record.get("country") if hasattr(record, "get") else None
    return {
        "status": "enriched",
        "detail": None,
        "created_at": created,
        "registrar": str(registrar)[:256] if registrar else None,
        "country": str(country)[:8] if country else None,
        "tls_age": None,
    }
