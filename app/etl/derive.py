"""Derived tables: the author roll-up and the domain roll-up.

Both are computed **in the database**, as one INSERT ... SELECT each, rather
than by pulling rows into Python and writing them back. Two reasons: it is an
order of magnitude faster, and an aggregate expressed as SQL is auditable
against the same query an analyst would write by hand to check it.

Order matters and is enforced by the caller: posts, then authors (aggregated
from posts), then domains (unnested from posts). Running domains before posts
produces an empty table and no error.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


#: Phase 1 also writes a per-source ``data/authors/`` Parquet with follower
#: counts and account-creation dates that the posts alone cannot supply. That is
#: merged on top of this aggregate by :func:`merge_author_profiles`.
_AUTHOR_ROLLUP = """
INSERT INTO authors (
    author_id, project_id, source, handle, post_count, first_seen, last_seen
)
SELECT
    p.author_id,
    p.project_id,
    -- The source is encoded in the namespaced id, but taking it from the column
    -- avoids assuming the prefix never contains a colon.
    min(p.source)                       AS source,
    -- An author's handle can change mid-corpus. The most recent one is the one
    -- an analyst will recognise, so order by time rather than taking min().
    (array_agg(p.author_handle ORDER BY p."timestamp" DESC)
        FILTER (WHERE p.author_handle IS NOT NULL))[1] AS handle,
    count(*)                            AS post_count,
    min(p."timestamp")                  AS first_seen,
    max(p."timestamp")                  AS last_seen
FROM posts p
WHERE p.project_id = :project_id
GROUP BY p.author_id, p.project_id
ON CONFLICT (author_id) DO UPDATE SET
    -- Author roll-ups accumulate, so this is DO UPDATE where posts are DO
    -- NOTHING. Only the aggregate columns are touched: the score columns belong
    -- to the scoring tasks and a reload must not blank them.
    post_count = EXCLUDED.post_count,
    first_seen = least(authors.first_seen, EXCLUDED.first_seen),
    last_seen  = greatest(authors.last_seen, EXCLUDED.last_seen),
    handle     = coalesce(EXCLUDED.handle, authors.handle)
"""

#: ``domains`` is keyed on (domain, project_id) so two projects studying the
#: same site score it independently -- risk is relative to a corpus.
_DOMAIN_ROLLUP = """
INSERT INTO domains (
    domain, project_id, first_seen, last_seen, post_count, author_count, enrichment_status
)
SELECT
    lower(d.domain)                     AS domain,
    p.project_id,
    min(p."timestamp")                  AS first_seen,
    max(p."timestamp")                  AS last_seen,
    count(*)                            AS post_count,
    count(DISTINCT p.author_id)         AS author_count,
    'pending'                           AS enrichment_status
FROM posts p
CROSS JOIN LATERAL unnest(p.domains) AS d(domain)
WHERE p.project_id = :project_id
  AND d.domain IS NOT NULL
  AND d.domain <> ''
GROUP BY lower(d.domain), p.project_id
ON CONFLICT (domain, project_id) DO UPDATE SET
    first_seen  = least(domains.first_seen, EXCLUDED.first_seen),
    last_seen   = greatest(domains.last_seen, EXCLUDED.last_seen),
    post_count  = EXCLUDED.post_count,
    author_count = EXCLUDED.author_count
    -- enrichment_status is deliberately NOT reset. A domain that WHOIS already
    -- resolved must not go back to 'pending' every time the corpus reloads;
    -- that would re-queue the whole table against a rate-limited external
    -- service on every ingest.
"""

#: Narrative counts need narratives, which do not exist until clustering has
#: run, so this is a separate statement the scoring task calls later rather than
#: part of the load.
_DOMAIN_NARRATIVE_COUNTS = """
UPDATE domains d SET narrative_count = sub.n
FROM (
    SELECT lower(dom.domain) AS domain, count(DISTINCT np.narrative_id) AS n
    FROM posts p
    CROSS JOIN LATERAL unnest(p.domains) AS dom(domain)
    JOIN narrative_posts np ON np.post_id = p.id
    WHERE p.project_id = :project_id
    GROUP BY lower(dom.domain)
) sub
WHERE d.domain = sub.domain AND d.project_id = :project_id
"""


def derive_authors(session: Session, project_id: uuid.UUID) -> int:
    result = session.execute(text(_AUTHOR_ROLLUP), {"project_id": project_id})
    log.info("authors roll-up touched %d rows", result.rowcount)
    return result.rowcount


def derive_domains(session: Session, project_id: uuid.UUID) -> int:
    result = session.execute(text(_DOMAIN_ROLLUP), {"project_id": project_id})
    log.info("domains roll-up touched %d rows", result.rowcount)
    return result.rowcount


def refresh_domain_narrative_counts(session: Session, project_id: uuid.UUID) -> int:
    result = session.execute(text(_DOMAIN_NARRATIVE_COUNTS), {"project_id": project_id})
    return result.rowcount


def merge_author_profiles(session: Session, project_id: uuid.UUID, authors_dir) -> int:
    """Fold Phase 1's ``data/authors/`` Parquet into the roll-up.

    Follower counts, following counts and account-creation dates come from the
    platform APIs, not from the posts, so they cannot be aggregated -- and they
    are exactly the cheap priors that separate "loud human" from "three-day-old
    account posting 400 times". Missing here means the source never exposed
    them, and stays NULL.
    """
    from pathlib import Path

    import pyarrow.dataset as ds

    authors_dir = Path(authors_dir)
    if not authors_dir.exists():
        log.info("no author profile parquet at %s; follower counts stay null", authors_dir)
        return 0

    table = ds.dataset(str(authors_dir), format="parquet", partitioning="hive").to_table()
    rows = table.to_pylist()
    if not rows:
        return 0

    payload = [
        {
            "author_id": row["author_id"],
            "created_at_source": row.get("created_at"),
            "followers": row.get("followers"),
            "following": row.get("following"),
            # Mastodon exposes a self-declared bot flag. It is the platform's
            # claim, not our estimate, and lives in a different column from
            # bot_score for exactly that reason.
            "is_bot_flagged": _extract_bot_flag(row.get("raw")),
        }
        for row in rows
    ]

    result = session.execute(
        text(
            """
            UPDATE authors a SET
                created_at_source = coalesce(v.created_at_source, a.created_at_source),
                followers         = coalesce(v.followers, a.followers),
                following         = coalesce(v.following, a.following),
                is_bot_flagged    = coalesce(v.is_bot_flagged, a.is_bot_flagged)
            FROM (
                SELECT
                    (elem->>'author_id')::text                        AS author_id,
                    (elem->>'created_at_source')::timestamptz          AS created_at_source,
                    (elem->>'followers')::int                          AS followers,
                    (elem->>'following')::int                          AS following,
                    (elem->>'is_bot_flagged')::boolean                 AS is_bot_flagged
                FROM jsonb_array_elements(cast(:payload AS jsonb)) AS elem
            ) v
            WHERE a.author_id = v.author_id AND a.project_id = :project_id
            """
        ),
        {"payload": _dumps(payload), "project_id": project_id},
    )
    log.info("merged %d author profiles", result.rowcount)
    return result.rowcount


def _extract_bot_flag(raw) -> bool | None:
    import json

    if not raw:
        return None
    try:
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    value = payload.get("bot")
    return bool(value) if isinstance(value, bool) else None


def _dumps(payload) -> str:
    import json

    return json.dumps(payload, default=str)
