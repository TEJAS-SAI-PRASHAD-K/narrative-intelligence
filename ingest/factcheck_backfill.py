"""Archive backfill for fact-check verdicts, driven by publisher sitemaps.

RSS feeds hold only the newest items, so the daily pass cannot reach older
articles. A publisher's sitemap lists its archive. This module discovers article
URLs from the sitemaps named in `configs/factcheck.yaml` and sends each one
through the same tier 1 ClaimReview path the RSS-sourced records use
(`FactCheckExtractor.fetch_html` and `from_html`). It adds no extraction code, so
rows have the same columns, tier, id scheme and language detection.

Sitemap XML is untrusted input. Three checks run before the parser sees a byte:

* The body must start, after an optional UTF-8 byte-order mark, with an ASCII
  ``<``. It must hold no NUL byte near the start. That refuses UTF-16 and UTF-32,
  which a byte search for ``<!DOCTYPE`` cannot see.
* An XML declaration may name only UTF-8, ASCII or ISO-8859-1.
* A body with a byte sequence ``<!DOCTYPE`` or ``<!ENTITY`` is refused, so no
  entity expands and no external entity loads.

The byte search for a DOCTYPE is meaningful only because the first two checks
admit ASCII-compatible encodings alone. The parser is the standard library's,
which needs no new dependency.
A sitemap response is read as a bounded stream, so its decoded size cannot pass
`backfill.max_sitemap_bytes`. Sitemap and article responses must end on the
publisher's registrable domain.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ingest.config import Settings
from ingest.factcheck import (
    FACTCHECKS_FILENAME,
    DropReason,
    FactCheck,
    FactCheckExtractor,
    OffDomainRedirect,
    ResponseTooLarge,
    Tier,
)
from ingest.normalize import resolve_domain
from ingest.sources.news_rss import _article_id

log = logging.getLogger("ingest.factcheck.backfill")

_DTD_RE = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_XML_DECL_ENCODING_RE = re.compile(
    rb"^<\?xml[^>]*?encoding\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)
_SAFE_ENCODINGS = frozenset({"utf-8", "us-ascii", "ascii", "iso-8859-1"})
_UTF8_BOM = b"\xef\xbb\xbf"
_NUL_SCAN_BYTES = 1024

DEFAULTS: dict[str, Any] = {
    "max_urls_per_publisher": 200,
    "max_sitemaps_per_publisher": 10,
    "max_sitemap_depth": 3,
    "max_sitemap_bytes": 50_000_000,
    "min_lastmod": None,
}


@dataclass
class SitemapEntry:
    loc: str
    lastmod: datetime | None = None
    #: The `lastmod` element held text that is not an ISO 8601 date.
    lastmod_bad: bool = False


@dataclass
class PublisherStats:
    """What one publisher's backfill did. Printed by the CLI."""

    publisher: str
    discovered: int = 0
    already_present: int = 0
    fetched: int = 0
    verdicts: int = 0
    sitemaps_read: int = 0
    sitemaps_skipped_old: int = 0
    sitemaps_filtered_pattern: int = 0
    filtered_pattern: int = 0
    filtered_lastmod: int = 0
    lastmod_absent_kept: int = 0
    lastmod_unparseable: int = 0
    #: A `<loc>` element with no text.
    empty_loc: int = 0
    #: A page URL listed more than once.
    duplicate_loc: int = 0
    #: Child sitemaps still queued when discovery stopped at the wanted count.
    sitemaps_unread: int = 0
    #: Verdicts left out by `claim_languages`, per detected language.
    language_excluded: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)


_LASTMOD_RE = re.compile(
    r"^(\d{4})(?:-(\d{2})(?:-(\d{2})"
    r"(?:T(\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?(Z|[+-]\d{2}:?\d{2})?)?)?)?$"
)


def parse_lastmod(value: str | None) -> datetime | None:
    """A sitemap `lastmod` as an aware UTC datetime, or ``None``.

    Accepts the W3C datetime forms ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD`` and
    ``YYYY-MM-DDThh:mm[:ss[.s]][Z|+hh:mm]``. Anything else, such as ``"5"`` or
    ``"yesterday"``, gives ``None``. There is no lenient fallback. A value with no
    zone is read as UTC.
    """
    if not value:
        return None
    match = _LASTMOD_RE.match(value.strip())
    if not match:
        return None
    year, month, day, hour, minute, second, fraction, zone = match.groups()
    try:
        parsed = datetime(
            int(year),
            int(month or 1),
            int(day or 1),
            int(hour or 0),
            int(minute or 0),
            int(second or 0),
            int((fraction or "0")[:6].ljust(6, "0")),
            tzinfo=timezone.utc,
        )
        if zone and zone != "Z":
            from datetime import timedelta

            digits = zone[1:].replace(":", "")
            offset = timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
            parsed = parsed - offset if zone[0] == "+" else parsed + offset
    except (ValueError, OverflowError):
        return None
    return parsed


def _as_cutoff(value: Any) -> datetime | None:
    """A config date (a YAML date or a string) as an aware UTC datetime."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    parsed = parse_lastmod(str(value))
    if parsed is None:
        raise ValueError(f"min_lastmod {value!r} is not a date")
    return parsed


def _refuse_unsafe_encoding(body: bytes) -> None:
    """Raise ``ValueError`` unless ``body`` is UTF-8 or ASCII XML, by its first bytes."""
    if b"\x00" in body[:_NUL_SCAN_BYTES]:
        raise ValueError("NUL byte near the start, so not UTF-8 or ASCII")
    start = body[len(_UTF8_BOM) :] if body.startswith(_UTF8_BOM) else body
    stripped = start.lstrip(b" \t\r\n")
    if not stripped.startswith(b"<"):
        raise ValueError("first byte is not an ASCII '<', so not UTF-8 or ASCII")
    declared = _XML_DECL_ENCODING_RE.match(stripped)
    if declared and declared.group(1).decode("ascii", "replace").lower() not in _SAFE_ENCODINGS:
        raise ValueError(f"declared encoding {declared.group(1)!r} is not allowed")


def parse_sitemap(
    body: bytes, counters: dict[str, int] | None = None
) -> tuple[str, list[SitemapEntry]]:
    """``("index" | "urlset", entries)`` from sitemap XML bytes.

    Raises ``ValueError`` for a body that is not UTF-8 or ASCII, a DOCTYPE or
    entity declaration, bad XML, or an unknown root element. Namespaces are
    ignored: some sitemaps add image or news namespaces beside the standard one.
    ``counters["empty_loc"]`` counts entries with a `<loc>` that holds no text.
    """
    _refuse_unsafe_encoding(body)
    if _DTD_RE.search(body):
        raise ValueError("document declares a DOCTYPE or entity")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError(f"not well-formed XML: {exc}") from exc
    kind = root.tag.rsplit("}", 1)[-1]
    if kind not in ("sitemapindex", "urlset"):
        raise ValueError(f"unexpected root element {kind!r}")
    child_tag = "sitemap" if kind == "sitemapindex" else "url"
    entries: list[SitemapEntry] = []
    for child in root:
        if child.tag.rsplit("}", 1)[-1] != child_tag:
            continue
        loc = lastmod = None
        for field_node in child:
            name = field_node.tag.rsplit("}", 1)[-1]
            if name == "loc" and loc is None:
                loc = (field_node.text or "").strip()
            elif name == "lastmod" and lastmod is None:
                lastmod = field_node.text
        if not loc:
            if counters is not None:
                counters["empty_loc"] = counters.get("empty_loc", 0) + 1
            continue
        parsed = parse_lastmod(lastmod)
        bad = parsed is None and bool(lastmod and lastmod.strip())
        entries.append(SitemapEntry(loc, parsed, bad))
    return ("index" if kind == "sitemapindex" else "urlset"), entries


def existing_record_ids(settings: Settings) -> set[str]:
    """Record ids already in the side artifact, legacy per-run files included."""
    import pyarrow.parquet as pq

    out_dir = settings.data_dir / "factchecks"
    paths: list[Path] = sorted(out_dir.glob("factchecks-*.parquet"))
    if (out_dir / FACTCHECKS_FILENAME).exists():
        paths.append(out_dir / FACTCHECKS_FILENAME)
    ids: set[str] = set()
    for path in paths:
        ids.update(pq.read_table(path, columns=["record_id"]).column("record_id").to_pylist())
    return ids


def record_for_url(url: str, domain: str) -> dict:
    """The record dict `FactCheckExtractor.from_html` takes, with the news adapter's id."""
    return {
        "id": f"news:{_article_id(url)}",
        "url": url,
        "source_detail": domain,
        "title": "",
        "text": "",
        "tags": [],
    }


class SitemapBackfill:
    """Discovers archive URLs per publisher and extracts their verdicts."""

    def __init__(self, extractor: FactCheckExtractor, known_ids: set[str] | None = None):
        self.extractor = extractor
        self.known_ids = known_ids if known_ids is not None else set()
        self.bounds = {**DEFAULTS, **(extractor.config.get("backfill") or {})}
        self.stats: dict[str, PublisherStats] = {}

    # --- configuration ---------------------------------------------------
    def eligible(self) -> list[str]:
        """Publisher keys with a `backfill` block, in config order."""
        return [d for d, p in self.extractor.publishers.items() if p.get("backfill")]

    def _count(
        self, stats: PublisherStats, reason: DropReason, detail: str, warn: bool = True
    ) -> None:
        self.extractor.drop(reason, detail)
        stats.dropped[reason.value] = stats.dropped.get(reason.value, 0) + 1
        if warn:
            log.warning("backfill %s (%s): %s", stats.publisher, reason.value, detail[:200])

    # --- sitemap reading -------------------------------------------------
    def _read_sitemap(
        self, url: str, domain: str, stats: PublisherStats
    ) -> tuple[str, list[SitemapEntry]] | None:
        """One sitemap, or ``None`` after counting why it was not read."""
        extractor = self.extractor
        if resolve_domain(url) != domain:
            self._count(stats, DropReason.SITEMAP_OFF_DOMAIN, url)
            return None
        fetch_config = extractor.config.get("fetch") or {}
        limit = int(fetch_config.get("max_consecutive_failures_per_domain", 3))
        if extractor._fetch_failures.get(domain, 0) >= limit:
            self._count(stats, DropReason.DOMAIN_GIVEN_UP, url)
            return None
        allowed, reason = extractor._robots_allows(url)
        if not allowed:
            self._count(stats, reason or DropReason.ROBOTS_UNREACHABLE, url)
            return None
        try:
            body = extractor._http_get(
                url,
                as_bytes=True,
                max_bytes=int(self.bounds["max_sitemap_bytes"]),
                expected_domain=domain,
            )
        except ResponseTooLarge as exc:
            self._count(stats, DropReason.SITEMAP_TOO_LARGE, f"{url}: {exc}")
            return None
        except OffDomainRedirect as exc:
            extractor._fetch_failures[domain] = extractor._fetch_failures.get(domain, 0) + 1
            self._count(stats, DropReason.BACKFILL_OFF_DOMAIN_REDIRECT, f"{url}: {exc}")
            return None
        except Exception as exc:
            extractor._fetch_failures[domain] = extractor._fetch_failures.get(domain, 0) + 1
            self._count(
                stats, DropReason.SITEMAP_UNREACHABLE, f"{url}: {type(exc).__name__}: {exc}"
            )
            return None
        extractor._fetch_failures[domain] = 0
        try:
            counters: dict[str, int] = {}
            parsed = parse_sitemap(body, counters)
            stats.empty_loc += counters.get("empty_loc", 0)
        except ValueError as exc:
            self._count(stats, DropReason.SITEMAP_MALFORMED, f"{url}: {exc}")
            return None
        stats.sitemaps_read += 1
        return parsed

    def _discover(
        self,
        sitemap_urls: list[str],
        domain: str,
        stats: PublisherStats,
        cutoff: datetime | None,
        wanted: int,
        accept: Callable[[SitemapEntry], bool],
        sitemap_filter: re.Pattern[str] | None = None,
    ) -> Iterator[SitemapEntry]:
        """Accepted page entries, reading child sitemaps newest first.

        Reading stops once ``wanted`` new entries exist.

        The loop is iterative with an explicit queue and a seen set, so a sitemap
        index that lists itself cannot loop. The sitemap budget bounds the work.
        """
        max_sitemaps = int(self.bounds["max_sitemaps_per_publisher"])
        max_depth = int(self.bounds["max_sitemap_depth"])
        queue: list[tuple[str, int]] = [(u, 0) for u in sitemap_urls]
        seen_sitemaps: set[str] = set()
        found_new = 0
        while queue:
            url, depth = queue.pop(0)
            if url in seen_sitemaps:
                continue
            seen_sitemaps.add(url)
            if found_new >= wanted:
                unread = 1 + sum(1 for u, _ in queue if u not in seen_sitemaps)
                stats.sitemaps_unread += unread
                log.warning(
                    "backfill %s: stopped at the wanted count, %d sitemaps unread",
                    stats.publisher,
                    unread,
                )
                return
            if stats.sitemaps_read >= max_sitemaps:
                self._count(stats, DropReason.SITEMAP_BUDGET_EXHAUSTED, url)
                continue
            result = self._read_sitemap(url, domain, stats)
            if result is None:
                continue
            kind, entries = result
            if kind == "index":
                if depth + 1 > max_depth:
                    self._count(stats, DropReason.SITEMAP_TOO_DEEP, url)
                    continue
                # Newest first, so a bounded run reaches recent archive pages
                # before old ones. A child with no lastmod sorts last.
                entries.sort(
                    key=lambda e: e.lastmod or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )
                children = []
                for entry in entries:
                    if entry.lastmod_bad:
                        stats.lastmod_unparseable += 1
                    if sitemap_filter and not sitemap_filter.search(entry.loc):
                        stats.sitemaps_filtered_pattern += 1
                        continue
                    if cutoff and entry.lastmod and entry.lastmod < cutoff:
                        stats.sitemaps_skipped_old += 1
                        continue
                    children.append((entry.loc, depth + 1))
                queue = children + queue
                continue
            for entry in entries:
                if not accept(entry):
                    continue
                yield entry
                if f"news:{_article_id(entry.loc)}" not in self.known_ids:
                    found_new += 1

    # --- per publisher ---------------------------------------------------
    def candidates(self, domain: str) -> list[SitemapEntry]:
        """URLs to fetch for one publisher: filtered, not yet stored, bounded, newest first."""
        publisher = self.extractor.publishers[domain]
        block = publisher.get("backfill") or {}
        stats = self.stats.setdefault(domain, PublisherStats(publisher.get("name") or domain))
        sitemap_urls = [str(u) for u in (block.get("sitemaps") or []) if u]
        if not sitemap_urls:
            self._count(stats, DropReason.NO_BACKFILL_CONFIG, f"{domain}: no sitemaps listed")
            return []
        if publisher.get("tier") != Tier.CLAIMREVIEW.value:
            self._count(
                stats, DropReason.NO_BACKFILL_CONFIG, f"{domain}: tier is not claimreview"
            )
            return []
        include = re.compile(block["url_pattern"]) if block.get("url_pattern") else None
        exclude = re.compile(block["exclude_pattern"]) if block.get("exclude_pattern") else None
        cutoffs = [
            c
            for c in (_as_cutoff(self.bounds["min_lastmod"]), _as_cutoff(block.get("min_lastmod")))
            if c
        ]
        cutoff = max(cutoffs) if cutoffs else None
        cap = int(self.bounds["max_urls_per_publisher"])
        if block.get("max_urls"):
            cap = min(cap, int(block["max_urls"]))

        sitemap_filter = (
            re.compile(block["sitemap_pattern"]) if block.get("sitemap_pattern") else None
        )
        kept: dict[str, SitemapEntry] = {}

        def accept(entry: SitemapEntry) -> bool:
            if resolve_domain(entry.loc) != domain:
                self._count(stats, DropReason.SITEMAP_OFF_DOMAIN, entry.loc)
                return False
            if (include and not include.search(entry.loc)) or (
                exclude and exclude.search(entry.loc)
            ):
                stats.filtered_pattern += 1
                return False
            if entry.lastmod_bad:
                stats.lastmod_unparseable += 1
            if cutoff:
                if entry.lastmod is None:
                    # Nothing to compare, so the URL stays, whether the date is
                    # absent or unreadable. Counted, because the bound cannot be
                    # enforced for it.
                    if not entry.lastmod_bad:
                        stats.lastmod_absent_kept += 1
                elif entry.lastmod < cutoff:
                    stats.filtered_lastmod += 1
                    return False
            if entry.loc in kept:
                stats.duplicate_loc += 1
                return False
            return True

        for entry in self._discover(
            sitemap_urls, domain, stats, cutoff, cap, accept, sitemap_filter
        ):
            kept[entry.loc] = entry
        stats.discovered = len(kept)

        fresh = []
        for entry in kept.values():
            if f"news:{_article_id(entry.loc)}" in self.known_ids:
                stats.already_present += 1
            else:
                fresh.append(entry)
        fresh.sort(
            key=lambda e: (e.lastmod or datetime.min.replace(tzinfo=timezone.utc), e.loc),
            reverse=True,
        )
        for entry in fresh[cap:]:
            self._count(stats, DropReason.BACKFILL_BOUND_REACHED, entry.loc, warn=False)
        if len(fresh) > cap:
            log.warning(
                "backfill %s (%s): %d URLs left for a later run",
                stats.publisher,
                DropReason.BACKFILL_BOUND_REACHED.value,
                len(fresh) - cap,
            )
        return fresh[:cap]

    def _language_allowed(self, domain: str, verdict: FactCheck, stats: PublisherStats) -> bool:
        """False, after counting, when `claim_languages` excludes the claim's language.

        A claim with no detected language is excluded when a list is set.
        """
        block = self.extractor.publishers[domain].get("backfill") or {}
        allowed = block.get("claim_languages")
        if not allowed or verdict.claim_lang in allowed:
            return True
        language = verdict.claim_lang or "?"
        stats.language_excluded[language] = stats.language_excluded.get(language, 0) + 1
        self.extractor.drop(DropReason.BACKFILL_LANGUAGE_EXCLUDED, f"{verdict.url}: {language}")
        stats.dropped[DropReason.BACKFILL_LANGUAGE_EXCLUDED.value] = (
            stats.dropped.get(DropReason.BACKFILL_LANGUAGE_EXCLUDED.value, 0) + 1
        )
        return False

    def run(self, domains: list[str] | None = None) -> Iterator[FactCheck]:
        """Yield one verdict per extractable discovered URL. Every skip is counted."""
        extractor = self.extractor
        if domains is None:
            for domain in extractor.publishers:
                if domain not in self.eligible():
                    extractor.drop(DropReason.NO_BACKFILL_CONFIG, f"{domain}: no backfill block")
        for domain in domains or self.eligible():
            if domain not in extractor.publishers:
                continue
            candidates = self.candidates(domain)
            stats = self.stats[domain]
            extractor.set_budget(len(candidates))
            for entry in candidates:
                before = dict(extractor.dropped)
                record = record_for_url(entry.loc, domain)
                html = extractor.fetch_html(entry.loc, expected_domain=domain)
                verdict = extractor.from_html(record, html) if html is not None else None
                if html is not None:
                    stats.fetched += 1
                for reason, count in extractor.dropped.items():
                    delta = count - before.get(reason, 0)
                    if delta:
                        stats.dropped[reason] = stats.dropped.get(reason, 0) + delta
                if verdict is not None and not self._language_allowed(domain, verdict, stats):
                    verdict = None
                if verdict is not None:
                    stats.verdicts += 1
                    yield verdict
