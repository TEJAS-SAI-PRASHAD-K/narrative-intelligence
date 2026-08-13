"""GDELT: topic-scoped article search (DOC 2.0) and the raw 15-minute drops.

Two access paths, implemented because they answer different questions:

* **DOC 2.0** (via ``gdeltdoc``) -- "which outlets covered this claim, in which
  language, on which day". Driven by ``configs/topics.yaml``.
* **Raw GKG files** -- "what themes, tone and named entities did GDELT extract
  from that coverage", plus the ``mentions`` stream for propagation velocity
  (the same event re-covered over time).

GDELT gives article *metadata*, not full text. Records therefore carry the
title (plus a short extract when GKG supplies one) and set
``content_type="article"``; full-text scraping of arbitrary news sites is out of
scope for Phase 1, legally and operationally.

What this source gives you: publisher domain, language, timestamp, themes, tone
-- the baseline for the Domain Risk pillar and for narrative velocity over time.
What it costs: nothing (open data), but the raw files are ~10-100MB per drop.
What it cannot tell you: what the article actually said. Tone and themes are
GDELT's inference, not ours, and the corpus is biased toward outlets GDELT
monitors -- overwhelmingly English-language and web-published.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

from ingest.config import sources_config, topics_config
from ingest.normalize import build_text_fields, resolve_domain
from ingest.schema import DropReason, EngagementMetrics, Record, make_id
from ingest.sources.base import BaseSource, SourceUnavailable

LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

#: GDELT ships headerless CSVs; the column order *is* the spec. GKG 2.1.
GKG_COLUMNS = [
    "GKGRECORDID",
    "DATE",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "Counts",
    "V2Counts",
    "Themes",
    "V2Themes",
    "Locations",
    "V2Locations",
    "Persons",
    "V2Persons",
    "Organizations",
    "V2Organizations",
    "V2Tone",
    "Dates",
    "GCAM",
    "SharingImage",
    "RelatedImages",
    "SocialImageEmbeds",
    "SocialVideoEmbeds",
    "Quotations",
    "AllNames",
    "Amounts",
    "TranslationInfo",
    "Extras",
]

#: Mentions 2.0. Parsed and kept as a side artifact, not as Records -- a mention
#: row has no text of its own, and inventing one would corrupt the corpus.
MENTIONS_COLUMNS = [
    "GLOBALEVENTID",
    "EventTimeDate",
    "MentionTimeDate",
    "MentionType",
    "MentionSourceName",
    "MentionIdentifier",
    "SentenceID",
    "Actor1CharOffset",
    "Actor2CharOffset",
    "ActionCharOffset",
    "InRawText",
    "Confidence",
    "MentionDocLen",
    "MentionDocTone",
    "MentionDocTranslationInfo",
    "Extras",
]

#: GDELT reports language names, not codes.
LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "russian": "ru",
    "arabic": "ar",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "hindi": "hi",
    "turkish": "tr",
    "polish": "pl",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "greek": "el",
    "hebrew": "he",
    "indonesian": "id",
    "vietnamese": "vi",
    "thai": "th",
    "ukrainian": "uk",
}

_PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.IGNORECASE | re.DOTALL)


class GdeltSource(BaseSource):
    name = "gdelt"
    source = "gdelt"
    requires_package = None  # DOC path needs gdeltdoc; the raw path needs only requests

    # --- fetch -----------------------------------------------------------
    def fetch(self) -> Iterator[dict]:
        config = sources_config().get(self.name, {})
        if self.options.get("mode") != "raw":
            yield from self._fetch_doc_api(config.get("doc_api", {}))
        if self.options.get("mode") != "doc":
            yield from self._fetch_raw_files(config.get("raw_files", {}))

    # --- DOC 2.0 ---------------------------------------------------------
    def _fetch_doc_api(self, config: dict) -> Iterator[dict]:
        try:
            from gdeltdoc import Filters, GdeltDoc
        except ImportError:
            self.log.warning(
                "gdeltdoc is not installed; skipping the DOC 2.0 path and using raw files only. "
                'Install with `pip install -e ".[sources]"`.'
            )
            return

        client = GdeltDoc()
        lookback = int(
            self.options.get("lookback_days")
            or topics_config().get("lookback_days")
            or self.settings.gdelt_lookback_days
        )
        max_records = int(
            self.options.get("max_records")
            or config.get("max_records_per_query")
            or self.settings.gdelt_max_records
        )
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=lookback)

        # GDELT throttles the DOC API hard and answers with an explicit rate
        # limit error rather than a 429, so pace ourselves rather than retry.
        from ingest.ratelimit import TokenBucket

        # 1 query / 10s. GDELT documents "one every 5 seconds" but enforces it
        # with a penalty window: once tripped, even 35-second spacing keeps
        # getting refused for a while. Pacing well under the stated limit is
        # cheaper than losing a topic's worth of coverage to a throttle.
        bucket = TokenBucket(rate_per_sec=float(config.get("queries_per_second", 0.1)), burst=1)
        cooloff = float(config.get("rate_limit_cooloff_seconds", 30))

        for topic in topics_config().get("topics", []):
            # `keyword` means *exact phrase* in gdeltdoc; a list is OR-joined
            # into ("a" OR "b"). Passing a hand-built boolean string instead
            # gets the whole thing quoted as one phrase, and the API rejects it
            # with "phrase search that was too short or too long". Verified
            # against the live API, not the docs.
            query = topic.get("gdelt_keywords") or topic.get("keywords", [])
            if not query:
                continue
            cursor_key = f"doc.{topic.get('id')}.{end.isoformat()}"
            if self.checkpoint.is_done(cursor_key) and not self.options.get("force"):
                self.log.info("skipping %s: already fetched today", cursor_key)
                continue
            filters = Filters(
                keyword=list(query),
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                num_records=max_records,
                language=_language_filter(config.get("languages", ["English"])),
            )
            bucket.acquire()
            try:
                articles = self._search_with_cooloff(client, filters, cooloff)
            except Exception as exc:
                # An invalid query or a throttle must not kill the other topics.
                # The exception carries no message for some error classes, so
                # log the type too or the line reads as "failed: " and tells
                # you nothing at 2am.
                self.log.warning(
                    "DOC query for %s failed: %s: %s",
                    topic.get("id"),
                    type(exc).__name__,
                    exc or "(no message from the API)",
                )
                self.note("doc_query_failed", f"{topic.get('id')}: {type(exc).__name__}")
                continue
            if articles is None or getattr(articles, "empty", True):
                self.log.info("DOC query for %s returned nothing", topic.get("id"))
                self.checkpoint.mark_done(cursor_key)
                continue

            rows = articles.to_dict(orient="records")
            self.log.info("DOC: %d articles for topic %s", len(rows), topic.get("id"))
            for row in rows:
                yield {"_kind": "doc", "_topic": topic.get("id"), **row}
            self.checkpoint.mark_done(cursor_key)
            self.record_manifest(
                cursor_key,
                url="https://api.gdeltproject.org/api/v2/doc/doc",
                rows=len(rows),
                extra={
                    "query": list(query),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )

    def _search_with_cooloff(self, client: Any, filters: Any, cooloff: float) -> Any:
        """One retry after a throttle, because GDELT's limiter is stateful.

        A rate-limited query is not a failed query -- it is a query asked too
        soon. Dropping the topic instead of waiting would silently bias the
        corpus toward whichever topic happens to run first.
        """
        import time

        try:
            return client.article_search(filters)
        except Exception as exc:
            if "RateLimit" not in type(exc).__name__:
                raise
            self.log.warning("gdelt throttled; waiting %.0fs before one retry", cooloff)
            self.note("doc_rate_limited")
            time.sleep(cooloff)
            return client.article_search(filters)

    # --- raw 15-minute drops ---------------------------------------------
    def _fetch_raw_files(self, config: dict) -> Iterator[dict]:
        if not config.get("enabled", True):
            return
        wanted = set(config.get("files") or ["gkg"])
        max_rows = int(config.get("max_rows_per_file", 5000))
        http = self.http()

        try:
            listing = http.get(LASTUPDATE_URL).text
        except Exception as exc:
            self.log.warning("could not fetch %s: %s", LASTUPDATE_URL, exc)
            return

        for url in _parse_lastupdate(listing):
            kind = _file_kind(url)
            if kind not in wanted and not (kind == "mentions" and "mentions" in wanted):
                continue
            if self.checkpoint.is_done(url) and not self.options.get("force"):
                self.log.info("skipping %s: already ingested", url)
                continue

            local = self.settings.raw_dir_for(self.name) / url.rsplit("/", 1)[-1]
            try:
                response = http.get(url)
            except Exception as exc:
                self.log.warning("download failed for %s: %s", url, exc)
                continue

            # Observed live (2026-08): lastupdate.txt lists the gkg file, but
            # requesting it returns 404 while export/mentions return 200. A
            # listed file is therefore not a guaranteed-present file. Never
            # leave a truncated artifact on disk -- a 0-byte file would be
            # "reused" on the next run and fail identically forever.
            if response.status_code != 200 or not response.content:
                self.note("raw_file_unavailable", f"{url} -> {response.status_code}")
                self.log.warning(
                    "%s listed in lastupdate.txt but not downloadable (status %s); skipping",
                    url,
                    response.status_code,
                )
                continue
            if not response.content.startswith(b"PK"):
                self.note("raw_file_not_a_zip", url)
                self.log.warning("%s is not a zip archive; skipping", url)
                continue
            local.write_bytes(response.content)

            try:
                rows = list(
                    _read_zipped_csv(local, GKG_COLUMNS if kind == "gkg" else MENTIONS_COLUMNS)
                )
            except SourceUnavailable as exc:
                # A corrupt drop is a bad file, not a broken pipeline.
                self.note("raw_file_unreadable", str(exc))
                self.log.warning("%s", exc)
                local.unlink(missing_ok=True)
                continue
            rows = rows[:max_rows]
            self.record_manifest(
                local.name, path=local, url=url, rows=len(rows), extra={"kind": kind}
            )

            if kind == "mentions":
                # Mentions have no text of their own. They answer "how fast did
                # this spread", so they are kept as a side artifact for Phase 2
                # to join on GLOBALEVENTID rather than forced into Record.
                self._write_side_artifact("mentions", rows)
                self.checkpoint.mark_done(url)
                continue

            for row in rows:
                yield {"_kind": "gkg", **row}
            self.checkpoint.mark_done(url)

    def _write_side_artifact(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_dir = self.settings.raw_dir_for(self.name) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = out_dir / f"{name}-{stamp}.parquet"
        table = pa.Table.from_pylist([{k: str(v) for k, v in row.items()} for row in rows])
        pq.write_table(table, path, compression="zstd")
        self.log.info("wrote %d %s rows to %s (not Records: no text field)", len(rows), name, path)
        self.record_manifest(path.name, path=path, rows=len(rows), extra={"kind": name})

    # --- map -------------------------------------------------------------
    def to_record(self, raw: dict) -> Record | None:
        if raw.get("_kind") == "gkg":
            return self._gkg_to_record(raw)
        return self._doc_to_record(raw)

    def _doc_to_record(self, raw: dict) -> Record | None:
        url = raw.get("url") or raw.get("documentidentifier")
        title = (raw.get("title") or "").strip()
        if not url:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None
        if not title:
            self.drop(DropReason.EMPTY_TEXT, url)
            return None

        timestamp = _parse_gdelt_dt(raw.get("seendate"))
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, url)
            return None

        domain = (raw.get("domain") or resolve_domain(url) or "unknown").lower()
        fields = build_text_fields(title, structured_urls=[url])
        fields["lang"] = _language_code(raw.get("language")) or fields.get("lang")

        media = [raw["socialimage"]] if raw.get("socialimage") else []
        return Record(
            native_id=_url_id(url),
            source="gdelt",
            source_detail=domain,
            content_type="article",
            # An outlet is the actor here, not a person. Attributing an article
            # to a pseudo-author keyed on the domain keeps the author roll-up
            # meaningful without inventing a byline GDELT never gave us.
            author_id=make_id("gdelt", domain),
            author_handle=domain,
            timestamp=timestamp,
            engagement=EngagementMetrics.unavailable(),  # GDELT measures none of it
            media_urls=media,
            raw={
                "api": "doc-2.0",
                "topic": raw.get("_topic"),
                "url": url,
                "url_mobile": raw.get("url_mobile"),
                "sourcecountry": raw.get("sourcecountry"),
                "language": raw.get("language"),
                "socialimage": raw.get("socialimage"),
            },
            **fields,
        )

    def _gkg_to_record(self, raw: dict) -> Record | None:
        url = (raw.get("DocumentIdentifier") or "").strip()
        if not url or not url.startswith("http"):
            self.drop(DropReason.UNSUPPORTED_TYPE, f"non-web gkg row {raw.get('GKGRECORDID')}")
            return None

        timestamp = _parse_gdelt_dt(raw.get("DATE"))
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, url)
            return None

        title = _page_title(raw.get("Extras")) or _title_from_url(url)
        if not title:
            self.drop(DropReason.EMPTY_TEXT, url)
            return None
        if not _page_title(raw.get("Extras")):
            # Be explicit that this "title" was reconstructed from the slug.
            self.note("title_from_url_slug", url)

        domain = (raw.get("SourceCommonName") or resolve_domain(url) or "unknown").lower()
        fields = build_text_fields(title, structured_urls=[url])

        tone = _parse_tone(raw.get("V2Tone"))
        media = [raw["SharingImage"]] if raw.get("SharingImage") else []

        return Record(
            native_id=str(raw.get("GKGRECORDID") or _url_id(url)),
            source="gdelt",
            source_detail=domain,
            content_type="article",
            author_id=make_id("gdelt", domain),
            author_handle=domain,
            timestamp=timestamp,
            engagement=EngagementMetrics.unavailable(),
            media_urls=media,
            raw={
                "api": "gkg-2.1",
                "url": url,
                "themes": _split_gdelt_list(raw.get("Themes")),
                "v2themes": _split_gdelt_list(raw.get("V2Themes"))[:50],
                "persons": _split_gdelt_list(raw.get("Persons")),
                "organizations": _split_gdelt_list(raw.get("Organizations")),
                "locations": _split_gdelt_list(raw.get("Locations"))[:20],
                "tone": tone,
                "source_collection": raw.get("SourceCollectionIdentifier"),
            },
            **fields,
        )


# --- module helpers -------------------------------------------------------


def _language_filter(languages: Any) -> str | list[str] | None:
    """Normalize the language filter into a form gdeltdoc emits validly.

    gdeltdoc renders a *list* as a parenthesized OR group. With one element
    that becomes ``(sourcelang:English)`` -- parentheses around a single term --
    and GDELT rejects it outright: "Parentheses may only be used around OR'd
    statements." A bare string renders as ``sourcelang:English``, which is
    accepted, and two or more languages render as a genuine OR group, which is
    also accepted. Verified against the live API; the docs do not mention it.
    """
    if not languages:
        return None
    if isinstance(languages, str):
        return languages
    values = [str(v) for v in languages if str(v).strip()]
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def _parse_lastupdate(text: str) -> list[str]:
    """``lastupdate.txt`` is three lines of ``size hash url``."""
    urls = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1].startswith("http"):
            urls.append(parts[-1])
    return urls


def _file_kind(url: str) -> str:
    lowered = url.lower()
    for kind in ("gkg", "mentions", "export"):
        if f".{kind}." in lowered:
            return kind
    return "unknown"


def _read_zipped_csv(path, columns: list[str]) -> Iterator[dict]:
    """Stream a zipped, headerless, tab-separated GDELT file into dicts."""
    try:
        with zipfile.ZipFile(path) as archive:
            name = archive.namelist()[0]
            with archive.open(name) as handle:
                stream = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
                for row in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
                    if not row:
                        continue
                    # Pad/truncate: GDELT occasionally ships a short trailing row.
                    padded = (row + [""] * len(columns))[: len(columns)]
                    yield dict(zip(columns, padded, strict=False))
    except (zipfile.BadZipFile, IndexError, OSError) as exc:
        raise SourceUnavailable(f"could not read GDELT archive {path}: {exc}") from exc


def _parse_gdelt_dt(value: Any) -> datetime | None:
    """GDELT stamps are ``YYYYMMDDHHMMSS`` (GKG) or ``YYYYMMDDTHHMMSSZ`` (DOC)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("T", "").replace("Z", "").replace("-", "")
    text = text.replace(":", "").replace(" ", "")
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _language_code(name: Any) -> str | None:
    if not name:
        return None
    text = str(name).strip().lower()
    if len(text) in (2, 3) and text.isalpha():
        return text
    return LANGUAGE_CODES.get(text)


def _url_id(url: str) -> str:
    """Stable id for an article: host + path, without the query string."""
    parts = urlsplit(url)
    return f"{parts.netloc}{parts.path}".strip("/").lower() or url


def _title_from_url(url: str) -> str:
    """Reconstruct a headline from the URL slug. Marked as such by the caller."""
    slug = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"\.(html?|php|aspx?)$", "", slug, flags=re.IGNORECASE)
    slug = re.sub(r"[-_]+", " ", slug)
    slug = re.sub(r"\b\d{5,}\b", "", slug).strip()
    return slug if len(slug) > 12 else ""


def _page_title(extras: Any) -> str | None:
    if not extras:
        return None
    match = _PAGE_TITLE_RE.search(str(extras))
    return match.group(1).strip() if match and match.group(1).strip() else None


def _split_gdelt_list(value: Any, sep: str = ";") -> list[str]:
    if not value:
        return []
    return [item.split(",")[0] for item in str(value).split(sep) if item.strip()]


def _parse_tone(value: Any) -> dict[str, float] | None:
    """``V2Tone`` is a comma-separated vector; the first field is average tone."""
    if not value:
        return None
    parts = str(value).split(",")
    keys = [
        "tone",
        "positive_score",
        "negative_score",
        "polarity",
        "activity_reference_density",
        "self_group_reference_density",
        "word_count",
    ]
    out: dict[str, float] = {}
    for key, part in zip(keys, parts, strict=False):
        try:
            out[key] = float(part)
        except ValueError:
            continue
    return out or None
