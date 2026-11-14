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

import fcntl
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from ingest.config import Settings, get_settings, load_yaml
from ingest.normalize import WORD_CHARS, clean_text, detect_lang, resolve_domain, simhash

log = logging.getLogger("ingest.factcheck")


class Tier(str, Enum):
    """How a verdict was obtained. Written onto every row; see module docstring."""

    #: schema.org ClaimReview JSON-LD. The publisher states claim and rating.
    CLAIMREVIEW = "claimreview"
    #: The RSS category list carries the verdict as one of its terms.
    RSS_TAG = "rss_tag"


class OffDomainRedirect(Exception):
    """A response ended on a registrable domain other than the expected one."""


class ResponseTooLarge(Exception):
    """A streamed response held more decoded bytes than the caller allows."""


#: Bytes per read of a streamed, size-bounded response.
_STREAM_CHUNK = 65536

#: robots.txt statuses that mean "no rules published".
_ROBOTS_ABSENT_STATUSES = frozenset({404, 410})


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
    #: An RSS-tier record whose tags hold no term from the rating map.
    NO_VERDICT_TAG = "no_verdict_tag"
    #: The per-run article budget was spent before this record was fetched.
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: This publisher's domain hit its consecutive-failure limit, so it is no
    #: longer fetched this run. The failures themselves were counted as
    #: FETCH_FAILED; this counts the records skipped because of them.
    DOMAIN_GIVEN_UP = "domain_given_up"
    #: The corpus held this record id more than once; only the newest is kept.
    DUPLICATE_RECORD = "duplicate_record"
    #: A corpus record with no id. A verdict is keyed on it, so it cannot be stored.
    NO_RECORD_ID = "no_record_id"
    #: A JSON-LD block nested deeper than the JSON parser can read. The page is
    #: kept in the cache, so without a count it would fail the same way each run.
    JSONLD_TOO_DEEP = "jsonld_too_deep"
    #: A backfill publisher has no usable `backfill` block, or its tier is not
    #: `claimreview`. Counted once per publisher, so a missing block is visible.
    NO_BACKFILL_CONFIG = "no_backfill_config"
    #: A sitemap could not be fetched: network error, bad status, or the
    #: domain's failure limit.
    SITEMAP_UNREACHABLE = "sitemap_unreachable"
    #: A sitemap body is not well-formed XML, has an unknown root element, or
    #: declares a DOCTYPE or entity (refused, never expanded).
    SITEMAP_MALFORMED = "sitemap_malformed"
    #: A sitemap body is larger than `backfill.max_sitemap_bytes`.
    SITEMAP_TOO_LARGE = "sitemap_too_large"
    #: A sitemap index nests deeper than `backfill.max_sitemap_depth`.
    SITEMAP_TOO_DEEP = "sitemap_too_deep"
    #: A sitemap or page URL listed by a sitemap is on another registrable domain.
    SITEMAP_OFF_DOMAIN = "sitemap_off_domain"
    #: The sitemap budget was spent before this child sitemap was read.
    SITEMAP_BUDGET_EXHAUSTED = "sitemap_budget_exhausted"
    #: A discovered URL was left out because the per-publisher URL bound was full.
    BACKFILL_BOUND_REACHED = "backfill_bound_reached"
    #: A backfill request ended on another registrable domain after redirects.
    #: The body is discarded and never cached.
    BACKFILL_OFF_DOMAIN_REDIRECT = "backfill_off_domain_redirect"
    #: A backfilled verdict whose detected claim language is not in the
    #: publisher's `claim_languages`. The row is not written.
    BACKFILL_LANGUAGE_EXCLUDED = "backfill_language_excluded"
    #: A backfill URL that an earlier run fetched and dropped for a stable reason,
    #: so it is neither fetched nor counted against the URL bound. Not a failure:
    #: `--recheck-dropped` fetches these pages again.
    PREVIOUSLY_DROPPED = "previously_dropped"
    #: The ledger of dropped backfill URLs could not be read. The run goes on as
    #: if it were empty, so the cost is fetching those pages again.
    LEDGER_UNREADABLE = "ledger_unreadable"


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


def jsonld_blocks(html: str, on_too_deep: Callable[[], None] | None = None) -> Iterator[Any]:
    """Every parseable JSON-LD block on a page.

    Malformed blocks are skipped rather than raising: publishers ship trailing
    commas and raw newlines inside strings, and one bad block on a page must not
    cost the good one next to it.

    A block nested past the interpreter's recursion limit raises
    ``RecursionError`` from the JSON parser, which is not a ``ValueError``. It
    is skipped like any other bad block and reported through ``on_too_deep``;
    retrying with newlines replaced cannot help, so the block is abandoned.
    """
    for match in _JSONLD_RE.finditer(html or ""):
        raw = match.group(1).strip()
        for candidate in (raw, raw.replace("\n", " ")):
            try:
                yield json.loads(candidate)
                break
            except RecursionError:
                if on_too_deep is not None:
                    on_too_deep()
                break
            except ValueError:
                continue


def walk_jsonld(node: Any) -> Iterator[dict]:
    """Every dict in a nested JSON-LD structure.

    ClaimReview is often not at the top level -- it turns up inside `@graph`,
    inside a list, or nested under the article node, and which one varies by
    publisher and by CMS version. Walking everything is cheaper than four
    per-publisher paths that each break separately.

    The walk uses an explicit stack, in document order, so nesting depth is
    bounded by memory rather than by the interpreter's recursion limit.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(reversed(list(current.values())))
        elif isinstance(current, list):
            stack.extend(reversed(current))


def _types(node: dict) -> list[str]:
    value = node.get("@type")
    return [str(v) for v in (value if isinstance(value, list) else [value]) if v]


def find_claim_reviews(
    html: str, on_too_deep: Callable[[], None] | None = None
) -> list[dict]:
    """ClaimReview nodes on a page, in document order."""
    return [
        node
        for block in jsonld_blocks(html, on_too_deep)
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

_PUNCT = re.compile(rf"[^{WORD_CHARS}\s]+", re.UNICODE)
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
        self.drop(DropReason.NO_VERDICT_TAG, url)
        return None

    # --- tier 1: ClaimReview --------------------------------------------
    def from_html(self, record: dict, html: str) -> FactCheck | None:
        url = record.get("url") or ""
        too_deep: list[int] = []
        reviews = find_claim_reviews(html, on_too_deep=lambda: too_deep.append(1))
        if not reviews:
            self.drop(DropReason.JSONLD_TOO_DEEP if too_deep else DropReason.NO_CLAIMREVIEW, url)
            return None
        if too_deep:
            log.warning("skipped %d JSON-LD blocks nested too deep on %s", len(too_deep), url)
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
    def fetch_html(self, url: str, expected_domain: str | None = None) -> str | None:
        """Article HTML, within budget and robots.txt, cached on disk.

        Cached permanently and never revalidated: an article is immutable once
        published and a verdict does not change. That also keeps a rerun from
        hammering publishers who are doing this project a favour.

        A cached page is returned before the budget and domain-limit checks, so
        it costs no budget. It is used without a new robots check: the robots
        check ran when the page was first fetched.

        ``expected_domain`` is set only by the backfill. A response that ends on
        another registrable domain after redirects is then discarded, not cached,
        and counted as BACKFILL_OFF_DOMAIN_REDIRECT. The daily pass leaves it unset.
        """
        config = self.config.get("fetch") or {}
        cache_path = self._cache_path(url) if config.get("cache", True) else None
        if cache_path and cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")

        if self._budget <= 0:
            self.drop(DropReason.BUDGET_EXHAUSTED, url)
            return None
        domain = resolve_domain(url) or "unknown"
        limit = int(config.get("max_consecutive_failures_per_domain", 3))
        if self._fetch_failures.get(domain, 0) >= limit:
            self.drop(DropReason.DOMAIN_GIVEN_UP, url)
            return None

        allowed, reason = self._robots_allows(url)
        if not allowed:
            self.drop(reason, url)
            return None

        self._budget -= 1
        try:
            html = self._http_get(url, expected_domain=expected_domain)
        except OffDomainRedirect as exc:
            self._fetch_failures[domain] = self._fetch_failures.get(domain, 0) + 1
            self.drop(DropReason.BACKFILL_OFF_DOMAIN_REDIRECT, f"{url}: {exc}")
            log.warning("discarded %s: %s", url, exc)
            return None
        except Exception as exc:
            self._fetch_failures[domain] = self._fetch_failures.get(domain, 0) + 1
            self.drop(DropReason.FETCH_FAILED, f"{url}: {type(exc).__name__}: {exc}")
            return None
        self._fetch_failures[domain] = 0
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html, encoding="utf-8")
        return html

    def _http_get(
        self,
        url: str,
        as_bytes: bool = False,
        max_bytes: int | None = None,
        expected_domain: str | None = None,
    ) -> Any:
        """One polite GET: the configured delay, the project User-Agent, the timeout.

        Raises on a network error or a bad status. The caller counts the failure.
        Returns the raw body when ``as_bytes`` is set, so XML keeps its own encoding.

        With neither ``max_bytes`` nor ``expected_domain`` this is the plain GET
        of the daily pass. Either one switches to a streamed response:

        * ``expected_domain``: the registrable domain of the final URL, after
          redirects, must match it, or ``OffDomainRedirect`` is raised before any
          body byte is read.
        * ``max_bytes``: the body is read in chunks of decoded content, which
          includes gzip decoding, and ``ResponseTooLarge`` is raised as soon as
          the total passes the bound. Memory stays under the bound plus one chunk.
        """
        config = self.config.get("fetch") or {}
        time.sleep(float(config.get("request_delay_seconds", 1.0)))
        import requests

        headers = {"User-Agent": self.settings.user_agent}
        timeout = float(config.get("timeout_seconds", 30))
        if max_bytes is None and expected_domain is None:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.content if as_bytes else response.text

        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        try:
            if expected_domain is not None:
                final = resolve_domain(getattr(response, "url", None) or url)
                if final != expected_domain:
                    raise OffDomainRedirect(f"ended on {final!r}, expected {expected_domain!r}")
            response.raise_for_status()
            if max_bytes is None:
                return response.content if as_bytes else response.text
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=_STREAM_CHUNK):
                total += len(chunk)
                if total > max_bytes:
                    raise ResponseTooLarge(f"more than {max_bytes} bytes")
                chunks.append(chunk)
            body = b"".join(chunks)
            return body if as_bytes else body.decode(response.encoding or "utf-8", "replace")
        finally:
            response.close()

    def set_budget(self, articles: int) -> None:
        """Set the number of live article fetches left in this run."""
        self._budget = int(articles)

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

        Status rule: 200 is parsed; 404 and 410 mean no rules are published
        and allow; every other status, and any network error, is unreachable.
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
                    # utf-8-sig drops a leading byte-order mark, which would
                    # otherwise glue itself to the first directive and hide it.
                    body = response.content.decode("utf-8-sig", errors="replace")
                    parser.parse(body.splitlines())
                elif response.status_code in _ROBOTS_ABSENT_STATUSES:
                    # Only 404 and 410 say "no robots.txt here", which the
                    # standard reads as crawl permitted. Any other status
                    # (401, 403, 429, other 4xx, 5xx) tells us
                    # nothing about the publisher's rules, so the parser
                    # stays None and the host is unreachable. Redirects never
                    # reach this point: requests follows them.
                    parser = RobotFileParser()
                    parser.parse([])
                else:
                    log.debug("robots.txt for %s returned %s", host, response.status_code)
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
            if not record.get("id"):
                self.drop(DropReason.NO_RECORD_ID, url)
                continue

            tier = publisher.get("tier")
            if tier == Tier.RSS_TAG.value:
                result = self.from_rss_tags(record, record.get("tags") or [])
            elif tier == Tier.CLAIMREVIEW.value:
                html = self.fetch_html(url)
                # fetch_html counts every reason it returns None for, so only
                # a page that came back (even an empty one) is parsed here, and
                # an empty page is counted by from_html as having no markup.
                result = self.from_html(record, html) if html is not None else None
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


def _newest_row_positions(frame: Any, allowed: set[str]) -> dict[str, int]:
    """Position of the newest row for each repeated id among ``allowed`` publishers.

    Recency is ``ingested_at``, the time the store took the row in. ``timestamp``
    is the article's publication time and is the same for every copy of it. A
    row with no ``ingested_at`` counts as older than any row that has one; when
    the column is absent or null for every copy, the first copy wins.
    """
    has_ingested = "ingested_at" in getattr(frame, "columns", ())
    best: dict[str, tuple[int, Any]] = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        if row.get("source_detail") not in allowed:
            continue
        record_id = row.get("id")
        if not record_id:
            continue
        stamp = row.get("ingested_at") if has_ingested else None
        if stamp is not None and stamp != stamp:  # NaT or NaN
            stamp = None
        held = best.get(record_id)
        if held is None or (stamp is not None and (held[1] is None or stamp > held[1])):
            best[record_id] = (position, stamp)
    return {record_id: position for record_id, (position, _) in best.items()}


def records_from_corpus(
    frame: Any,
    publishers: Iterable[str],
    on_duplicate: Callable[[str], None],
) -> Iterator[dict]:
    """Project corpus rows into the flat dicts the extractor consumes.

    ``url``, ``title`` and ``tags`` live inside the ``raw`` blob rather than in
    schema columns, so they are lifted here. RSS ``tags`` arrive either as a
    list of terms or as one comma-joined string depending on the feed, and both
    are flattened -- Fact Crescendo's verdict is a term inside that list, so
    getting this wrong silently loses the whole tier-2 path.

    Each id is yielded once, and the newest copy wins (see
    ``_newest_row_positions``): the store can return the same id twice, and a
    feed that revises an article's tags later can add the verdict term to the
    newer copy only. ``on_duplicate`` is required and is called with the id of
    every older copy that is skipped, so no caller can drop duplicates uncounted.
    """
    allowed = set(publishers)
    newest = _newest_row_positions(frame, allowed)
    for position, (_, row) in enumerate(frame.iterrows()):
        if row.get("source_detail") not in allowed:
            continue
        record_id = row.get("id")
        if record_id and newest[record_id] != position:
            on_duplicate(str(record_id))
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


#: The one output file. Fixed so that a rerun updates it rather than adding a
#: sibling that every reader would have to know to combine.
FACTCHECKS_FILENAME = "factchecks.parquet"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of one side-artifact file. A corrupt file raises rather than being
    skipped: overwriting it with a merge that omits it would lose its rows."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


#: Lock file taken by every writer. A dot-file, so dataset readers skip it.
_LOCK_FILENAME = ".factchecks.lock"

#: Subdirectory that receives merged legacy files. The underscore prefix makes
#: pyarrow's dataset discovery skip it, so a directory read sees each row once.
_LEGACY_DIRNAME = "_legacy"


def write_factchecks(
    verdicts: list[FactCheck], settings: Settings | None = None
) -> Path | None:
    """Merge ``verdicts`` into ``data/factchecks/factchecks.parquet``.

    The file is keyed on ``record_id``: these rows annotate an article, not a
    run, so a rerun must replace a row rather than repeat it. When a
    ``record_id`` is present more than once the row with the latest
    ``extracted_at`` wins (ISO-8601 UTC strings sort chronologically), and a
    tie goes to the later source, which is the new run.

    The whole read, merge and replace runs under an exclusive ``flock`` on a
    lock file in the output directory, so overlapping runs take turns and the
    second one sees the first one's rows instead of overwriting them.

    Files named ``factchecks-<timestamp>.parquet`` are from an earlier layout
    that wrote one file per run. They are read as the oldest source so their
    rows are not lost. After the merged file is in place they are moved, not
    deleted, into ``_legacy/``: left beside the main file, a directory read
    would return every one of their rows twice. Pyarrow skips paths that start
    with an underscore, so ``_legacy/`` is invisible to readers.

    Not partitioned by date on purpose: a reader always wants all of them to
    join against the corpus, and partitioning would cost a predicate on every
    read.
    """
    if not verdicts:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    settings = settings or get_settings()
    out_dir = settings.data_dir / "factchecks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FACTCHECKS_FILENAME

    with open(out_dir / _LOCK_FILENAME, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        legacy_files = sorted(out_dir.glob("factchecks-*.parquet"))
        sources: list[Iterable[dict[str, Any]]] = [_read_rows(legacy) for legacy in legacy_files]
        if path.exists():
            sources.append(_read_rows(path))
        sources.append(v.as_row() for v in verdicts)

        merged: dict[str, dict[str, Any]] = {}
        for rows in sources:
            for row in rows:
                current = merged.get(row["record_id"])
                if current is None or (row["extracted_at"] or "") >= (
                    current["extracted_at"] or ""
                ):
                    merged[row["record_id"]] = row

        # Explicit schema, matching ingest/store.py's corpus schema. Inference
        # is not an option here: `claim_simhash` is an unsigned 64-bit value,
        # pyarrow infers int64, and anything above 2**63 raises
        # "OverflowError: Python int too large to convert to C long" -- on
        # roughly half of all simhashes, i.e. intermittently and for no visible
        # reason.
        table = pa.Table.from_pylist(list(merged.values()), schema=FACTCHECK_SCHEMA())

        # Write beside the target and rename: os.replace is atomic on one
        # filesystem, so a crash leaves the old file or the new one, never
        # half. The fsync makes the bytes durable before the rename can be.
        fd, tmp_name = tempfile.mkstemp(dir=out_dir, prefix=".factchecks-", suffix=".tmp")
        os.close(fd)
        try:
            pq.write_table(table, tmp_name, compression="zstd")
            with open(tmp_name, "rb") as written:
                os.fsync(written.fileno())
            # mkstemp creates the file private to this user; the artifact is shared data.
            os.chmod(tmp_name, 0o644)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

        if legacy_files:
            archive = out_dir / _LEGACY_DIRNAME
            archive.mkdir(exist_ok=True)
            for legacy in legacy_files:
                destination = archive / legacy.name
                if destination.exists():
                    destination = archive / f"{legacy.stem}-{time.time_ns()}{legacy.suffix}"
                os.replace(legacy, destination)
    log.info("wrote %d verdicts (%d new this run) to %s", len(merged), len(verdicts), path)
    return path


class CollapseReason(str, Enum):
    """Why a stored verdict row is left out of a deduplicated read. Counted, never silent."""

    #: Its claim simhash is within the Hamming radius of a kept row's claim,
    #: directly or through a chain of such rows.
    DUPLICATE_CLAIM = "duplicate_claim"


#: Claims whose simhashes differ in this many bits or fewer are the same claim.
#: Matches `simhash_hamming` in configs/models.yaml, used for corpus records.
CLAIM_HAMMING_RADIUS = 3


@dataclass
class CollapseResult:
    """Outcome of `collapse_duplicate_claims`."""

    #: One row per group, sorted by ``record_id``.
    kept: list[dict[str, Any]]
    #: ``(record_id, kept_record_id, hamming distance to that row)`` per collapsed row.
    #: In a chain the distance can be larger than the radius.
    collapsed: list[tuple[str, str, int]]
    counts: dict[str, int]


def _row_dict(row: Any) -> dict[str, Any]:
    return row.as_row() if isinstance(row, FactCheck) else row


def collapse_duplicate_claims(
    rows: Iterable[Any], max_distance: int = CLAIM_HAMMING_RADIUS
) -> CollapseResult:
    """Keep one row per claim, for a reader that must not count a claim twice.

    The same claim is debunked by several publishers, so an evaluation set built
    from the raw artifact counts it more than once. This is a read-side view:
    ``factchecks.parquet`` stays lossless and nothing is deleted from it.

    Two rows are linked when their ``claim_simhash`` values differ in
    ``max_distance`` bits or fewer. Groups are the connected components of that
    relation (union-find), so a chain A-B-C collapses to one row even when A and
    C are farther apart than the radius. Chaining can merge claims that are not
    the same claim. The radius is small to keep that rare, and the distance of
    each collapsed row to its representative is returned for audit.

    The representative of a group is the row with the smallest ``record_id``.
    ``extracted_at`` plays no part: it is re-stamped when an extraction reruns, so
    ordering by it would let the representative change between runs. The result
    does not depend on input order.

    A hash of ``0`` (or ``None``) means "no hash", so that row never links to
    anything and is always kept. Those rows are counted as ``unhashed_kept``.
    Each collapsed row is counted under ``CollapseReason.DUPLICATE_CLAIM``.
    A group whose rows carry more than one distinct ``label`` is still collapsed,
    and is counted once under ``label_conflict`` so the disagreement is visible.
    """
    items = [_row_dict(row) for row in rows]
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes: list[int] = []
    for item in items:
        value = item.get("claim_simhash")
        value = 0 if value is None or value != value else int(value)
        if not 0 <= value < 1 << 64:
            raise ValueError(f"claim_simhash {value} does not fit in an unsigned 64-bit integer")
        hashes.append(value)

    # Pigeonhole: two 64-bit values within 3 bits differ in at most 3 of 4
    # 16-bit blocks, so they share at least one whole block. Bucketing by
    # (block, value) finds every pair without comparing all of them. A larger
    # radius needs more blocks, so the block count follows it. Bits left over
    # when 64 is not divisible by the block count only lower the budget.
    blocks = max_distance + 1
    width = 64 // blocks
    mask = (1 << width) - 1
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, value in enumerate(hashes):
        if value == 0:
            continue
        for block in range(blocks):
            key = (block, (value >> (block * width)) & mask)
            buckets.setdefault(key, []).append(index)
    for members in buckets.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if find(i) != find(j) and bin(hashes[i] ^ hashes[j]).count("1") <= max_distance:
                    parent[find(j)] = find(i)

    def order(i: int) -> str:
        return str(items[i].get("record_id") or "")

    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)

    kept: list[dict[str, Any]] = []
    collapsed: list[tuple[str, str, int]] = []
    label_conflicts = 0
    for members in groups.values():
        representative = min(members, key=order)
        labels = {items[i].get("label") for i in members if items[i].get("label") is not None}
        if len(labels) > 1:
            label_conflicts += 1
        kept.append(items[representative])
        rep_id = str(items[representative].get("record_id") or "")
        for index in members:
            if index != representative:
                distance = bin(hashes[index] ^ hashes[representative]).count("1")
                collapsed.append((str(items[index].get("record_id") or ""), rep_id, distance))
    kept.sort(key=lambda row: str(row.get("record_id") or ""))
    collapsed.sort()
    counts = {
        CollapseReason.DUPLICATE_CLAIM.value: len(collapsed),
        "unhashed_kept": sum(1 for value in hashes if value == 0),
        "label_conflict": label_conflicts,
    }
    return CollapseResult(kept=kept, collapsed=collapsed, counts=counts)


def read_deduplicated_factchecks(settings: Settings | None = None) -> CollapseResult:
    """``factchecks.parquet`` with duplicate claims collapsed, for an evaluation harness.

    An absent file gives an empty result. The file is not modified.
    """
    settings = settings or get_settings()
    path = settings.data_dir / "factchecks" / FACTCHECKS_FILENAME
    rows = _read_rows(path) if path.exists() else []
    result = collapse_duplicate_claims(rows)
    if result.collapsed:
        log.info(
            "collapsed %d verdict rows as %s",
            len(result.collapsed),
            CollapseReason.DUPLICATE_CLAIM.value,
        )
    return result


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
