"""Retrieval for Compass Context.

**Retrieve first, generate strictly over what was retrieved.** The generator
never sees the open web and never answers from its own parameters; it sees a
numbered list of passages and is told to write only what those support.

Two corpora, in priority order:

1. **Reputable outlets already in the corpus.** Phase 1 ingested news and GDELT
   articles alongside the social posts, and an article from Reuters that is
   already in the database is a better source than anything fetched live: it is
   pinned, it is dated, and it does not depend on a network call at generation
   time.
2. **Nothing else, for now.** A live web-search retriever is the obvious
   extension and is deliberately not here: it needs a search API key, a fetch
   budget, and a policy on what counts as a source, and shipping it half-built
   would mean generating over whatever a search engine returned.

When retrieval finds nothing, the pipeline does not generate. That path ends in
`insufficient_evidence`, which is the honest outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

#: Domains whose articles may be used as evidence. Deliberately a short
#: allowlist rather than a reputation score: the whole point of Compass is that
#: an analyst can see where a statement came from, and "our model rated this
#: domain 0.82" is not an answer to that.
REPUTABLE_DOMAINS = frozenset(
    {
        "reuters.com",
        "apnews.com",
        "bbc.co.uk",
        "bbc.com",
        "npr.org",
        "theguardian.com",
        "nytimes.com",
        "washingtonpost.com",
        "ft.com",
        "bloomberg.com",
        "economist.com",
        "aljazeera.com",
        "cnn.com",
        "politifact.com",
        "snopes.com",
        "factcheck.org",
        "fullfact.org",
    }
)

#: Minimum retrieved passages before generation is attempted at all. Below this
#: there is not enough to write a cited note over, and trying produces exactly
#: the ungrounded paragraph the validator would reject.
MIN_DOCUMENTS = 2


@dataclass
class RetrievedDocument:
    post_id: str
    url: str | None
    domain: str | None
    title: str | None
    publisher: str | None
    snippet: str
    published_at: Any
    similarity: float | None = None

    def as_passage(self, index: int) -> str:
        header = f"[{index}] {self.publisher or self.domain or 'unknown source'}"
        if self.published_at:
            header += f", {self.published_at:%Y-%m-%d}"
        return f"{header}\n{self.snippet.strip()}"


def retrieve(
    session: Session, project_id, narrative_id, *, limit: int = 8
) -> list[RetrievedDocument]:
    """Reputable articles near this narrative, by embedding similarity.

    kNN against the narrative's representative posts rather than keyword search:
    the claim and the reporting about it rarely share vocabulary -- a post says
    "they scanned the ballots twice", the article says "duplicate tabulation
    allegations" -- and a keyword retriever misses exactly the cases that matter.
    """
    from app.config import get_api_settings

    settings = get_api_settings()

    rows = session.execute(
        text(
            f"""
            WITH centroid AS (
                SELECT avg(e.embedding) AS vector
                FROM narrative_posts np
                JOIN {settings.embedding_table} e ON e.post_id = np.post_id
                WHERE np.narrative_id = :narrative_id
                  AND np.is_representative
            )
            SELECT p.id, p."text", p.urls, p.domains, p.source_detail, p."timestamp",
                   e.embedding <=> (SELECT vector FROM centroid) AS distance
            FROM posts p
            JOIN {settings.embedding_table} e ON e.post_id = p.id
            WHERE p.project_id = :project_id
              AND p.content_type = 'article'
              AND EXISTS (
                  SELECT 1 FROM unnest(p.domains) AS d(domain)
                  WHERE lower(d.domain) = ANY(:reputable)
              )
              AND (SELECT vector FROM centroid) IS NOT NULL
            ORDER BY distance
            LIMIT :limit
            """  # noqa: S608 - table name comes from config, not from input
        ),
        {
            "narrative_id": narrative_id,
            "project_id": project_id,
            "reputable": sorted(REPUTABLE_DOMAINS),
            "limit": limit,
        },
    ).all()

    documents = []
    for row in rows:
        domain = next((d for d in (row.domains or []) if d.lower() in REPUTABLE_DOMAINS), None)
        documents.append(
            RetrievedDocument(
                post_id=row.id,
                url=(row.urls or [None])[0],
                domain=domain,
                # Article text in this corpus is title + lede; that is what the
                # generator gets, and the snippet stored on the citation is the
                # same string, so an analyst auditing the note sees exactly what
                # the model saw.
                title=(row.text or "").split("\n")[0][:300] or None,
                publisher=row.source_detail,
                snippet=(row.text or "")[:1200],
                published_at=row.timestamp,
                similarity=(1 - float(row.distance)) if row.distance is not None else None,
            )
        )

    log.info("compass retrieval for narrative %s: %d document(s)", narrative_id, len(documents))
    return documents
