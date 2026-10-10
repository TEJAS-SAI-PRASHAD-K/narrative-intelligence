"""Fact-check verdicts: turning published ratings into training labels.

This is an *enrichment pass*, not a source adapter. The articles are already in
the corpus -- `news_rss` fetched them and `Record`s exist for them -- and what
is missing is the (claim, verdict) pair each one carries. Adding a `factcheck`
source instead would have meant a new `Source` literal, a new database enum, a
migration across 27 tables and a change to `openapi.json`, to express something
that is an annotation on a news article rather than a different platform.

So verdicts are written as a **side Parquet artifact** keyed by `record_id`,
which is the pattern GDELT's `mentions` already uses, for the mirror-image
reason the Phase 1 author gave there: a mentions row has no text of its own, so
inventing a `Record` for it would corrupt the corpus. A verdict has no text of
its own either.

Why this matters more than any other ingestion work in this project: the
misinformation classifier currently does not beat TF-IDF, and its one
full-scale checkpoint is gone. Indian fact-checkers publish machine-readable
verdicts in English and Hindi under open RSS, daily. They are the only
accessible supervision at the scale this needs, and they are the reason the
case is India at all.

**Every row records the tier that produced it.** A tier-1 verdict is the
publisher's own structured rating; a tier-4 verdict would be this project
guessing from a headline. A label column that mixes them without saying so is a
label column nobody can audit, and `artifacts/` is full of evidence that the
author of this repo cared about exactly that distinction.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from ingest.config import Settings, get_settings, load_yaml
from ingest.normalize import clean_text, detect_lang, resolve_domain, simhash

log = logging.getLogger("ingest.factcheck")


class Tier(str, Enum):
    """How a verdict was obtained. Written onto every row; see module docstring."""

    #: schema.org ClaimReview JSON-LD. The publisher states claim and rating.
    CLAIMREVIEW = "claimreview"
    #: The RSS category list carries the verdict as one of its terms.
    RSS_TAG = "rss_tag"


class DropReason(str, Enum):
    """Why a candidate produced no row. Every one is counted, never silent."""

    NO_PUBLISHER_CONFIG = "no_publisher_config"
    NO_URL = "no_url"
    FETCH_FAILED = "fetch_failed"
    #: robots.txt was read and it says no. A publisher's policy; permanent.
    ROBOTS_DISALLOWED = "robots_disallowed"
    #: robots.txt could not be read. We still decline to fetch -- unknown is
    #: not permission -- but this is a transient local problem, not a policy,
    #: and conflating the two lets a flaky network look like a publisher ban.
    ROBOTS_UNREACHABLE = "robots_unreachable"
    NO_CLAIMREVIEW = "no_claimreview"
    NO_RATING = "no_rating"
    NO_CLAIM_TEXT = "no_claim_text"
    #: The rating is real and maps to `drop` -- the middle of the scale.
    RATING_IS_MIDDLE = "rating_is_middle"
    #: The rating is not in the map at all. A vocabulary change, not a middle.
    RATING_UNKNOWN = "rating_unknown"


@dataclass
class FactCheck:
    """One published verdict, joined to the corpus record it came from."""

    #: Namespaced corpus id, so this joins to `data/normalized/**` directly.
    record_id: str
    url: str
    domain: str
    publisher: str
    tier: str
    #: The claim as the PUBLISHER states it -- not the article headline. These
    #: differ in a way that matters: the headline is written to be read, the
    #: claim is written to be checked.
    claim: str
    claim_lang: str | None
    #: The publisher's own rating string, kept verbatim. The normalized label is
    #: derived from it; this column is what lets the mapping be re-derived later
    #: without re-fetching 200 pages.
    rating_raw: str
    #: "false" | "true". Never the middle: those rows are dropped and counted.
    label: str
    claim_simhash: int
    reviewed_at: str | None = None
    extracted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


# --- JSON-LD ---------------------------------------------------------------

_JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)


def jsonld_blocks(html: str) -> Iterator[Any]:
    """Every parseable JSON-LD block on a page.

    Malformed blocks are skipped rather than raising: publishers ship trailing
    commas and raw newlines inside strings, and one bad block on a page must not
    cost the good one next to it.
    """
    for match in _JSONLD_RE.finditer(html or ""):
        raw = match.group(1).strip()
        for candidate in (raw, raw.replace("\n", " ")):
            try:
                yield json.loads(candidate)
                break
            except ValueError:
                continue


def walk_jsonld(node: Any) -> Iterator[dict]:
    """Every dict in a nested JSON-LD structure.

    ClaimReview is often not at the top level -- it turns up inside `@graph`,
    inside a list, or nested under the article node, and which one varies by
    publisher and by CMS version. Walking everything is cheaper than four
    per-publisher paths that each break separately.
    """
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk_jsonld(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_jsonld(value)


def _types(node: dict) -> list[str]:
    value = node.get("@type")
    return [str(v) for v in (value if isinstance(value, list) else [value]) if v]


def find_claim_reviews(html: str) -> list[dict]:
    """ClaimReview nodes on a page, in document order."""
    return [
        node
        for block in jsonld_blocks(html)
        for node in walk_jsonld(block)
        if "ClaimReview" in _types(node)
    ]


def _first_string(value: Any) -> str:
    """Coerce a JSON-LD text field to one string.

    The Quint returns `itemReviewed.name` as a LIST. schema.org permits it, most
    publishers do not do it, and reading it as a string yields the stringified
    Python list -- which looks like successful extraction and is garbage.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_string(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        return _first_string(value.get("name") or value.get("@value") or value.get("text"))
    return str(value)


def claim_and_rating(review: dict) -> tuple[str, str]:
    """``(claim text, publisher rating string)`` from one ClaimReview node."""
    item = review.get("itemReviewed")
    claim = _first_string(item.get("name") if isinstance(item, dict) else item)
    if not claim:
        # Older markup puts the claim directly on the review.
        claim = _first_string(review.get("claimReviewed"))
    rating_node = review.get("reviewRating")
    rating = ""
    if isinstance(rating_node, dict):
        # alternateName is the human-readable verdict ("False", "Half true").
        # ratingValue is a NUMBER on an undeclared scale, so it is unusable
        # across publishers: 1 means "worst" to one and "best" to another.
        rating = _first_string(rating_node.get("alternateName") or rating_node.get("name"))
    return clean_text(claim), clean_text(rating)


# --- rating normalization --------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_rating(rating: str) -> str:
    """Fold a publisher's rating string for lookup: lowercase, depunctuated."""
    text = _PUNCT.sub(" ", (rating or "").lower())
    return _WS.sub(" ", text).strip()


class FactCheckExtractor:
    """Reads corpus records, writes verdict rows. Network only for tier 1."""

    def __init__(self, settings: Settings | None = None, config: dict | None = None):
        self.settings = settings or get_settings()
        self.config = config if config is not None else load_yaml("factcheck")
        self.publishers: dict[str, dict] = self.config.get("publishers", {}) or {}
        self.ratings: dict[str, str] = {
            normalize_rating(k): str(v).lower()
            for k, v in (self.config.get("ratings") or {}).items()
        }
        self.dropped: dict[str, int] = {}
        #: Per-rating counts, so the cost of the label mapping is reportable
        #: rather than inferred. `ratings_seen` includes dropped and unknown.
        self.ratings_seen: dict[str, int] = {}
        self.unknown_ratings: dict[str, int] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._fetch_failures: dict[str, int] = {}
        self._budget = int((self.config.get("fetch") or {}).get("max_articles", 200))

    # --- bookkeeping -----------------------------------------------------
    def drop(self, reason: DropReason, detail: str = "") -> None:
        self.dropped[reason.value] = self.dropped.get(reason.value, 0) + 1
        log.debug("dropped (%s): %s", reason.value, detail[:160])

    def label_for(self, rating: str) -> str | None:
        """Map a publisher rating onto ``false``/``true``, or ``None`` to drop.

        ``None`` is returned for both the middle of the scale and an unseen
        term, but the two are counted separately -- a middle rating is this
        project's decision, an unknown rating is a publisher changing their
        vocabulary, and only the second one is a bug to go and fix.
        """
        key = normalize_rating(rating)
        self.ratings_seen[key] = self.ratings_seen.get(key, 0) + 1
        mapped = self.ratings.get(key)
        if mapped is None:
            self.unknown_ratings[key] = self.unknown_ratings.get(key, 0) + 1
            self.drop(DropReason.RATING_UNKNOWN, rating)
            return None
        if mapped == "drop":
            self.drop(DropReason.RATING_IS_MIDDLE, rating)
            return None
        return mapped

    # --- tier 2: the verdict is an RSS category --------------------------
    def from_rss_tags(self, record: dict, tags: Iterable[str]) -> FactCheck | None:
        """Fact Crescendo ships ``['False', 'International', 'इजरायल', ...]``.

        The verdict is positional only in the sense that it is simply one tag
        among many, so every tag is tested against the rating map and the first
        hit wins. A topic tag cannot collide with a verdict unless a publisher
        names a topic "False".

        No claim text is available on this path -- the publisher does not supply
        one in the feed -- so the article's own title stands in, and that is
        recorded by the tier rather than papered over.
        """
        url = record.get("url") or ""
        for tag in tags:
            key = normalize_rating(str(tag))
            if key not in self.ratings:
                continue
            label = self.label_for(str(tag))
            if label is None:
                return None
            claim = clean_text(record.get("title") or record.get("text") or "")
            if not claim:
                self.drop(DropReason.NO_CLAIM_TEXT, url)
                return None
            return self._build(record, claim, str(tag), label, Tier.RSS_TAG)
        return None

    # --- tier 1: ClaimReview --------------------------------------------
    def from_html(self, record: dict, html: str) -> FactCheck | None:
        url = record.get("url") or ""
        reviews = find_claim_reviews(html)
        if not reviews:
            self.drop(DropReason.NO_CLAIMREVIEW, url)
            return None
        for review in reviews:
            claim, rating = claim_and_rating(review)
            if not rating:
                continue
            if not claim:
                continue
            label = self.label_for(rating)
            if label is None:
                return None
            reviewed = _first_string(review.get("datePublished") or review.get("dateModified"))
            return self._build(record, claim, rating, label, Tier.CLAIMREVIEW, reviewed or None)
        # Reviews existed but carried no usable (claim, rating). Distinguished
        # from "no markup at all" because they are different bugs.
        self.drop(DropReason.NO_RATING, url)
        return None

    def _build(
        self,
        record: dict,
        claim: str,
        rating_raw: str,
        label: str,
        tier: Tier,
        reviewed_at: str | None = None,
    ) -> FactCheck:
        domain = record.get("source_detail") or resolve_domain(record.get("url")) or "unknown"
        publisher = (self.publishers.get(domain) or {}).get("name") or domain
        return FactCheck(
            record_id=str(record.get("id") or ""),
            url=str(record.get("url") or ""),
            domain=domain,
            publisher=publisher,
            tier=tier.value,
            claim=claim,
            claim_lang=detect_lang(claim),
            rating_raw=rating_raw,
            label=label,
            # Near-duplicate detection over CLAIMS, not articles. The same
            # claim is debunked by several publishers within days, and training
            # on all of them inflates the apparent dataset while teaching one
            # example. Phase 2 collapses on this.
            claim_simhash=simhash(claim),
            reviewed_at=reviewed_at,
        )

    # --- fetching --------------------------------------------------------
    def fetch_html(self, url: str) -> str | None:
        """Article HTML, within budget and robots.txt, cached on disk.

        Cached permanently and never revalidated: an article is immutable once
        published and a verdict does not change. That also keeps a rerun from
        hammering publishers who are doing this project a favour.
        """
        config = self.config.get("fetch") or {}
        if self._budget <= 0:
            return None
        domain = resolve_domain(url) or "unknown"
        limit = int(config.get("max_consecutive_failures_per_domain", 3))
        if self._fetch_failures.get(domain, 0) >= limit:
            return None

        cache_path = self._cache_path(url) if config.get("cache", True) else None
        if cache_path and cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")

        allowed, reason = self._robots_allows(url)
        if not allowed:
            self.drop(reason, url)
            return None

        self._budget -= 1
        time.sleep(float(config.get("request_delay_seconds", 1.0)))
        try:
            import requests

            response = requests.get(
                url,
                headers={"User-Agent": self.settings.user_agent},
                timeout=float(config.get("timeout_seconds", 30)),
            )
            response.raise_for_status()
            html = response.text
        except Exception as exc:
            self._fetch_failures[domain] = self._fetch_failures.get(domain, 0) + 1
            self.drop(DropReason.FETCH_FAILED, f"{url}: {type(exc).__name__}: {exc}")
            return None
        self._fetch_failures[domain] = 0
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html, encoding="utf-8")
        return html

    def _cache_path(self, url: str) -> Path:
        import hashlib

        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        domain = resolve_domain(url) or "unknown"
        return self.settings.raw_dir / "factcheck" / domain / f"{digest}.html"

    def _robots_allows(self, url: str) -> tuple[bool, DropReason | None]:
        """``(allowed, reason if not)``, honouring robots.txt per host.

        robots.txt is fetched with ``requests`` and our own User-Agent rather
        than with ``RobotFileParser.read()``. That matters: ``read()`` goes
        through urllib with **no User-Agent and no timeout**, so it fails on a
        site that rejects urllib's default agent and hangs on a slow one. The
        first run of this module reported 53 articles as robots-disallowed
        across four publishers whose robots.txt in fact permits all of them --
        every one of those was `read()` failing, not a publisher saying no.

        An unreachable robots.txt still means no fetch: unknown is not
        permission. But it reports ROBOTS_UNREACHABLE rather than
        ROBOTS_DISALLOWED, so a bad network cannot be mistaken for a ban.
        """
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            parser: RobotFileParser | None = None
            try:
                import requests

                response = requests.get(
                    f"{host}/robots.txt",
                    headers={"User-Agent": self.settings.user_agent},
                    timeout=float((self.config.get("fetch") or {}).get("timeout_seconds", 30)),
                )
                if response.status_code == 200:
                    parser = RobotFileParser()
                    parser.parse(response.text.splitlines())
                elif 400 <= response.status_code < 500:
                    # No robots.txt at all is "crawl permitted" per the
                    # standard, and is not the same as one we could not read.
                    parser = RobotFileParser()
                    parser.parse([])
            except Exception as exc:
                log.debug("robots.txt unreachable for %s: %s", host, exc)
            self._robots[host] = parser

        parser = self._robots[host]
        if parser is None:
            return False, DropReason.ROBOTS_UNREACHABLE
        try:
            if parser.can_fetch(self.settings.user_agent, url):
                return True, None
        except Exception:  # pragma: no cover - malformed robots.txt
            return False, DropReason.ROBOTS_UNREACHABLE
        return False, DropReason.ROBOTS_DISALLOWED

    # --- driver ----------------------------------------------------------
    def run(self, records: Iterable[dict]) -> Iterator[FactCheck]:
        """Extract a verdict per eligible record. Skips are counted, not silent."""
        for record in records:
            domain = record.get("source_detail") or ""
            publisher = self.publishers.get(domain)
            if not publisher:
                self.drop(DropReason.NO_PUBLISHER_CONFIG, domain)
                continue
            url = record.get("url") or ""
            if not url:
                self.drop(DropReason.NO_URL, str(record.get("id")))
                continue

            tier = publisher.get("tier")
            if tier == Tier.RSS_TAG.value:
                result = self.from_rss_tags(record, record.get("tags") or [])
            elif tier == Tier.CLAIMREVIEW.value:
                html = self.fetch_html(url)
                result = self.from_html(record, html) if html else None
            else:
                self.drop(DropReason.NO_PUBLISHER_CONFIG, f"{domain}: tier={tier!r}")
                continue
            if result is not None:
                yield result

    def summary(self) -> dict[str, Any]:
        """What was produced and what it cost. Printed by the CLI."""
        return {
            "dropped": dict(sorted(self.dropped.items())),
            "ratings_seen": dict(sorted(self.ratings_seen.items(), key=lambda kv: -kv[1])),
            "unknown_ratings": dict(sorted(self.unknown_ratings.items(), key=lambda kv: -kv[1])),
            "fetch_budget_left": self._budget,
        }


# --- corpus <-> disk -------------------------------------------------------


def FACTCHECK_SCHEMA():  # noqa: N802 - matches ingest/store.py's naming
    """The side artifact's Parquet schema, declared rather than inferred.

    Built lazily so importing this module does not require pyarrow, matching
    how the rest of the ingestion layer treats its heavier dependencies.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("record_id", pa.string()),
            pa.field("url", pa.string()),
            pa.field("domain", pa.string()),
            pa.field("publisher", pa.string()),
            pa.field("tier", pa.string()),
            pa.field("claim", pa.string()),
            pa.field("claim_lang", pa.string()),
            pa.field("rating_raw", pa.string()),
            pa.field("label", pa.string()),
            pa.field("claim_simhash", pa.uint64()),
            pa.field("reviewed_at", pa.string()),
            pa.field("extracted_at", pa.string()),
        ]
    )


def records_from_corpus(frame: Any, publishers: Iterable[str]) -> Iterator[dict]:
    """Project corpus rows into the flat dicts the extractor consumes.

    ``url``, ``title`` and ``tags`` live inside the ``raw`` blob rather than in
    schema columns, so they are lifted here. RSS ``tags`` arrive either as a
    list of terms or as one comma-joined string depending on the feed, and both
    are flattened -- Fact Crescendo's verdict is a term inside that list, so
    getting this wrong silently loses the whole tier-2 path.
    """
    allowed = set(publishers)
    for _, row in frame.iterrows():
        if row.get("source_detail") not in allowed:
            continue
        raw = row.get("raw")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = {}
        raw = raw or {}
        tags: list[str] = []
        for tag in raw.get("tags") or []:
            tags.extend(part.strip() for part in str(tag).split(",") if part.strip())
        yield {
            "id": row.get("id"),
            "url": raw.get("url") or "",
            "source_detail": row.get("source_detail"),
            "title": raw.get("title") or "",
            "text": row.get("text") or "",
            "tags": tags,
        }


def write_factchecks(
    verdicts: list[FactCheck], settings: Settings | None = None
) -> Path | None:
    """Write the side artifact. One Parquet file per run, never partitioned.

    Not partitioned by date on purpose: these rows are keyed on the article they
    annotate, not on their own timestamp, and a reader always wants all of them
    to join against the corpus. Partitioning would buy nothing and cost a
    predicate on every read.
    """
    if not verdicts:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    settings = settings or get_settings()
    out_dir = settings.data_dir / "factchecks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"factchecks-{stamp}.parquet"
    # Explicit schema, matching ingest/store.py's corpus schema. Inference is
    # not an option here: `claim_simhash` is an unsigned 64-bit value, pyarrow
    # infers int64, and anything above 2**63 raises
    # "OverflowError: Python int too large to convert to C long" -- on roughly
    # half of all simhashes, i.e. intermittently and for no visible reason.
    table = pa.Table.from_pylist([v.as_row() for v in verdicts], schema=FACTCHECK_SCHEMA())
    pq.write_table(table, path, compression="zstd")
    log.info("wrote %d verdicts to %s", len(verdicts), path)
    return path


def label_distribution(verdicts: Iterable[FactCheck]) -> dict[str, int]:
    """Counts per label. Called by the CLI because of what it keeps showing.

    **Published fact-checks are a single-class dataset.** Fact-checkers debunk
    viral falsehoods; confirming a true claim is not newsworthy, so a `true`
    verdict is vanishingly rare. On the first full run over this corpus the
    split was 96 `false` and 0 `true`.

    A binary classifier cannot be trained on one class, so this artifact is
    **positive-class supervision, not a training set**. Negatives have to come
    from somewhere else, and where is a real research decision rather than an
    implementation detail -- see the README section this function is cited in.
    """
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.label] = counts.get(verdict.label, 0) + 1
    return counts
