"""Archive backfill for fact-check verdicts, driven by publisher sitemaps.

RSS feeds hold only the newest items, so the daily pass cannot reach older
articles. A publisher's sitemap lists its archive. This module discovers article
URLs from the sitemaps named in `configs/factcheck.yaml` and sends each one
through the same tier 1 ClaimReview path the RSS-sourced records use
(`FactCheckExtractor.fetch_html` and `from_html`). It adds no extraction code, so
rows have the same columns, tier, id scheme and language detection.

Sitemap XML is untrusted input. A document that declares a DOCTYPE or an entity
is refused before it reaches the parser, so no entity expands and no external
entity loads. The parser is the standard library's, which needs no new
dependency.
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
    Tier,
)
from ingest.normalize import resolve_domain
from ingest.sources.news_rss import _article_id

log = logging.getLogger("ingest.factcheck.backfill")

_DTD_RE = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)

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
    dropped: dict[str, int] = field(default_factory=dict)


def parse_lastmod(value: str | None) -> datetime | None:
    """A sitemap `lastmod` as an aware UTC datetime, or ``None`` when absent or unreadable."""
    if not value or not value.strip():
        return None
    from dateutil import parser as date_parser

    try:
        parsed = date_parser.isoparse(value.strip())
    except (ValueError, OverflowError):
        try:
            parsed = date_parser.parse(value.strip())
        except (ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def parse_sitemap(body: bytes) -> tuple[str, list[SitemapEntry]]:
    """``("index" | "urlset", entries)`` from sitemap XML bytes.

    Raises ``ValueError`` for a DOCTYPE or entity declaration, bad XML, or an
    unknown root element. Namespaces are ignored: some sitemaps add image or news
    namespaces beside the standard one.
    """
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
        if loc:
            entries.append(SitemapEntry(loc, parse_lastmod(lastmod)))
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
            body = extractor._http_get(url, as_bytes=True)
        except Exception as exc:
            extractor._fetch_failures[domain] = extractor._fetch_failures.get(domain, 0) + 1
            self._count(
                stats, DropReason.SITEMAP_UNREACHABLE, f"{url}: {type(exc).__name__}: {exc}"
            )
            return None
        extractor._fetch_failures[domain] = 0
        if len(body) > int(self.bounds["max_sitemap_bytes"]):
            self._count(stats, DropReason.SITEMAP_TOO_LARGE, f"{url}: {len(body)} bytes")
            return None
        try:
            parsed = parse_sitemap(body)
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
            if cutoff:
                if entry.lastmod is None:
                    # Nothing to compare, so the URL stays. Counted, because the
                    # bound cannot be enforced for it.
                    stats.lastmod_absent_kept += 1
                elif entry.lastmod < cutoff:
                    stats.filtered_lastmod += 1
                    return False
            return entry.loc not in kept

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
                html = extractor.fetch_html(entry.loc)
                verdict = extractor.from_html(record, html) if html is not None else None
                if html is not None:
                    stats.fetched += 1
                for reason, count in extractor.dropped.items():
                    delta = count - before.get(reason, 0)
                    if delta:
                        stats.dropped[reason] = stats.dropped.get(reason, 0) + delta
                if verdict is not None:
                    stats.verdicts += 1
                    yield verdict
