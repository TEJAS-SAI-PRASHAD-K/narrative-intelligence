"""Gather the raw inputs the fusion score needs, from the database.

Kept apart from ``fusion.py`` on purpose: that module is pure and testable
against hand-built vectors, and this one is all SQL. The seam is
:class:`NarrativeSignals`, which is the only thing that crosses.

Everything here returns ``None`` where a signal could not be measured, and
never a zero. That discipline is the reason the fusion function can renormalize
honestly -- if this layer coalesced, nothing downstream could recover the
difference.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.scoring.fusion import NarrativeSignals

log = logging.getLogger(__name__)

#: One query per project, not per narrative. At a few hundred narratives the
#: per-narrative version is a few hundred round trips for data one GROUP BY
#: already has.
_CONTENT_SIGNALS = """
SELECT
    np.narrative_id,
    -- Engagement-weighted, because a false claim nobody saw is not the same
    -- problem as one that reached a hundred thousand people. The +1 keeps a
    -- zero-engagement post contributing its own weight rather than vanishing.
    sum(s.misinfo_likelihood * (1 + COALESCE(p.likes,0) + COALESCE(p.shares,0)))
        FILTER (WHERE s.misinfo_likelihood IS NOT NULL)
      / NULLIF(sum(1 + COALESCE(p.likes,0) + COALESCE(p.shares,0))
        FILTER (WHERE s.misinfo_likelihood IS NOT NULL), 0)  AS misinfo_agg,
    avg(s.toxicity) FILTER (WHERE s.toxicity IS NOT NULL)     AS toxicity,
    -- Share of *scored* posts that are negative. The FILTER on the denominator
    -- is what keeps unscored posts out of both halves of the fraction.
    count(*) FILTER (WHERE s.sentiment = 'negative')::float
      / NULLIF(count(*) FILTER (WHERE s.sentiment IS NOT NULL), 0)  AS negative_sentiment,
    count(*)                                                   AS member_posts,
    count(*) FILTER (WHERE s.misinfo_likelihood IS NOT NULL)    AS scored_posts
FROM narrative_posts np
JOIN posts p ON p.id = np.post_id
LEFT JOIN post_scores s ON s.post_id = p.id
JOIN narratives n ON n.id = np.narrative_id
WHERE n.project_id = :project_id
GROUP BY np.narrative_id
"""

_AUTHOR_SIGNALS = """
SELECT
    np.narrative_id,
    -- Over the authors the classifier could score, not over all authors. News
    -- outlets are skipped with a reason code and counting them as human would
    -- dilute the ratio toward zero on exactly the narratives news carries.
    count(DISTINCT a.author_id) FILTER (WHERE a.bot_score > :bot_threshold)::float
      / NULLIF(count(DISTINCT a.author_id) FILTER (WHERE a.bot_score IS NOT NULL), 0)
        AS bot_like_ratio,
    count(DISTINCT a.author_id) FILTER (
        WHERE a.handle IS NULL OR a.author_id LIKE '%%__deleted__'
    )::float / NULLIF(count(DISTINCT a.author_id), 0)          AS anonymous_ratio,
    count(DISTINCT a.author_id)                                 AS member_authors,
    count(DISTINCT a.author_id) FILTER (WHERE a.bot_score IS NOT NULL) AS scorable_authors
FROM narrative_posts np
JOIN posts p ON p.id = np.post_id
JOIN authors a ON a.author_id = p.author_id AND a.project_id = :project_id
JOIN narratives n ON n.id = np.narrative_id
WHERE n.project_id = :project_id
GROUP BY np.narrative_id
"""

#: Density of co-posting edges among a narrative's own authors, against a
#: complete graph on the same set. Normalising by n*(n-1) is what makes a
#: 20-author cluster with 50 edges comparable to a 200-author one with 500.
_COORDINATION_DENSITY = """
WITH members AS (
    SELECT DISTINCT np.narrative_id, p.author_id
    FROM narrative_posts np
    JOIN posts p ON p.id = np.post_id
    JOIN narratives n ON n.id = np.narrative_id
    WHERE n.project_id = :project_id
),
sizes AS (
    SELECT narrative_id, count(*) AS n FROM members GROUP BY narrative_id
),
internal AS (
    SELECT m1.narrative_id, count(*) AS edges
    FROM network_edges e
    JOIN members m1 ON m1.author_id = e.src_author_id
    JOIN members m2 ON m2.author_id = e.dst_author_id
                   AND m2.narrative_id = m1.narrative_id
    WHERE e.project_id = :project_id
    GROUP BY m1.narrative_id
)
SELECT s.narrative_id,
       CASE WHEN s.n > 1
            THEN LEAST(1.0, COALESCE(i.edges, 0)::float / (s.n * (s.n - 1)))
            ELSE NULL END AS density
FROM sizes s LEFT JOIN internal i ON i.narrative_id = s.narrative_id
"""

#: Share of a narrative's volume in buckets above `burst_multiple` times its own
#: mean. Relative to itself, not to the project: a small narrative that fires
#: entirely in six hours is coordinated, and comparing it to a large one's
#: absolute volume would hide that.
_BURSTINESS = """
WITH buckets AS (
    SELECT np.narrative_id,
           date_bin(interval '6 hours', p."timestamp", timestamp '2000-01-01 00:00+00') AS b,
           count(*) AS volume
    FROM narrative_posts np
    JOIN posts p ON p.id = np.post_id
    JOIN narratives n ON n.id = np.narrative_id
    WHERE n.project_id = :project_id
    GROUP BY np.narrative_id, b
),
stats AS (
    SELECT narrative_id, avg(volume) AS mean_volume, sum(volume) AS total, count(*) AS n
    FROM buckets GROUP BY narrative_id
)
SELECT s.narrative_id,
       CASE WHEN s.n >= 3 AND s.mean_volume > 0 THEN
            COALESCE(sum(b.volume) FILTER (
                WHERE b.volume > :burst_multiple * s.mean_volume
            ), 0)::float / s.total
       ELSE NULL END AS burstiness
FROM stats s JOIN buckets b ON b.narrative_id = s.narrative_id
GROUP BY s.narrative_id, s.mean_volume, s.total, s.n
"""

_COHORT_CONCENTRATION = """
WITH member_cohorts AS (
    SELECT np.narrative_id, ac.cohort_id, count(*) AS posts
    FROM narrative_posts np
    JOIN posts p ON p.id = np.post_id
    JOIN author_cohorts ac ON ac.author_id = p.author_id
    JOIN narratives n ON n.id = np.narrative_id
    WHERE n.project_id = :project_id
    GROUP BY np.narrative_id, ac.cohort_id
),
totals AS (
    SELECT narrative_id, sum(posts) AS total FROM member_cohorts GROUP BY narrative_id
)
SELECT mc.narrative_id, max(mc.posts)::float / NULLIF(t.total, 0) AS concentration
FROM member_cohorts mc JOIN totals t ON t.narrative_id = mc.narrative_id
GROUP BY mc.narrative_id, t.total
"""

_AUTHENTICITY = """
SELECT
    np.narrative_id,
    count(*) FILTER (WHERE mc.confidence > :deepfake_threshold)::float
      / NULLIF(count(mc.id), 0)                        AS deepfake_hits,
    avg(d.risk_score) / 100.0                           AS domain_risk_agg
FROM narrative_posts np
JOIN posts p ON p.id = np.post_id
JOIN narratives n ON n.id = np.narrative_id
LEFT JOIN media_checks mc ON mc.post_id = p.id
LEFT JOIN LATERAL unnest(p.domains) AS dom(domain) ON true
LEFT JOIN domains d ON d.domain = lower(dom.domain) AND d.project_id = :project_id
WHERE n.project_id = :project_id
GROUP BY np.narrative_id
"""

_COMPASS_STATUS = """
SELECT c.narrative_id, c.verification_status
FROM compass_contexts c
JOIN narratives n ON n.id = c.narrative_id
WHERE n.project_id = :project_id AND c.superseded_by IS NULL
"""


def collect(
    session: Session, project_id: uuid.UUID, config: dict[str, Any]
) -> dict[str, NarrativeSignals]:
    """Every narrative's signals for one project, in six queries."""
    thresholds = config.get("thresholds", {})
    params = {
        "project_id": project_id,
        "bot_threshold": float(thresholds.get("bot_like", 0.6)),
        "deepfake_threshold": float(thresholds.get("deepfake", 0.5)),
        "burst_multiple": float(thresholds.get("burst_multiple", 3.0)),
    }

    signals: dict[str, NarrativeSignals] = {}

    def entry(narrative_id) -> NarrativeSignals:
        key = str(narrative_id)
        if key not in signals:
            signals[key] = NarrativeSignals(narrative_id=key)
        return signals[key]

    for row in session.execute(text(_CONTENT_SIGNALS), params):
        item = entry(row.narrative_id)
        item.misinfo_likelihood_agg = _f(row.misinfo_agg)
        item.toxicity = _f(row.toxicity)
        item.negative_sentiment = _f(row.negative_sentiment)
        item.inputs["member_posts"] = row.member_posts
        item.inputs["scored_posts"] = row.scored_posts

    for row in session.execute(text(_AUTHOR_SIGNALS), params):
        item = entry(row.narrative_id)
        item.bot_like_ratio = _f(row.bot_like_ratio)
        item.anonymous_author_ratio = _f(row.anonymous_ratio)
        item.inputs["member_authors"] = row.member_authors
        item.inputs["bot_scorable_authors"] = row.scorable_authors

    for row in session.execute(text(_COORDINATION_DENSITY), params):
        entry(row.narrative_id).co_post_similarity_density = _f(row.density)

    for row in session.execute(text(_BURSTINESS), params):
        entry(row.narrative_id).temporal_burstiness = _f(row.burstiness)

    for row in session.execute(text(_COHORT_CONCENTRATION), params):
        entry(row.narrative_id).cohort_concentration = _f(row.concentration)

    for row in session.execute(text(_AUTHENTICITY), params):
        item = entry(row.narrative_id)
        item.deepfake_hits = _f(row.deepfake_hits)
        item.domain_risk_agg = _f(row.domain_risk_agg)

    for row in session.execute(text(_COMPASS_STATUS), params):
        entry(row.narrative_id).compass_status = row.verification_status

    return signals


def populations(signals: dict[str, NarrativeSignals]) -> dict[str, list[float | None]]:
    """The per-signal reference distributions the percentile normalizer ranks against."""
    names = (
        "misinfo_likelihood_agg",
        "toxicity",
        "negative_sentiment",
        "bot_like_ratio",
        "co_post_similarity_density",
        "temporal_burstiness",
        "cohort_concentration",
        "deepfake_hits",
        "domain_risk_agg",
        "anonymous_author_ratio",
    )
    return {
        name: [getattr(item, name) for item in signals.values() if getattr(item, name) is not None]
        for name in names
    }


def _f(value) -> float | None:
    """NULL stays NULL. The one conversion this module is allowed to make."""
    return None if value is None else float(value)
