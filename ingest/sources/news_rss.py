"""News: RSS/Atom feeds (reliable path) plus NewsAPI (optional bonus path).

RSS needs no credentials and never expires, so it is the path the pipeline
depends on. NewsAPI sits behind a key check and degrades to RSS-only with a
warning when the key is absent -- its free tier is ~100 requests/day, results
are delayed ~24h, and the licence is development/non-commercial only, none of
which a reproducible corpus should depend on.

Syndicated wire copy is the interesting failure mode here: one AP story appears
verbatim across dozens of outlets. Those records are *kept*, not deduplicated
away, because republication breadth is itself a spread signal -- the ``simhash``
column is what lets Phase 2 collapse them when it wants to.

What this source gives you: headline, summary, publication time, outlet domain,
and full text where the outlet publishes it openly.
What it costs: nothing for RSS; NewsAPI's free tier is 100 req/day and 24h
delayed.
What it cannot tell you: anything about reach or engagement -- no news feed
reports how many people read an article, so those metrics stay null.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from ingest.config import sources_config, topics_config
from ingest.normalize import build_text_fields, resolve_domain
from ingest.schema import DropReason, EngagementMetrics, Record, make_id
from ingest.sources.base import BaseSource

NEWSAPI_URL = "https://newsapi.org/v2/everything"


class NewsRssSource(BaseSource):
    name = "news_rss"
    source = "news"
    requires_package = "feedparser"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._robots: dict[str, RobotFileParser | None] = {}
        self._fulltext_budget = 0

    # --- fetch -----------------------------------------------------------
    def fetch(self) -> Iterator[dict]:
        config = sources_config().get(self.name, {})
        yield from self._fetch_rss(config)
        yield from self._fetch_newsapi(config.get("newsapi", {}))

    def _fetch_rss(self, config: dict) -> Iterator[dict]:
        import feedparser

        feeds: list[str] = self.options.get("feeds") or config.get("feeds") or []
        fulltext = config.get("fulltext", {})
        self._fulltext_budget = (
            int(fulltext.get("max_articles", 0)) if fulltext.get("enabled") else 0
        )
        http = self.http()

        for feed_url in feeds:
            try:
                response = http.get(feed_url)
                parsed = feedparser.parse(response.content)
            except Exception as exc:
                self.log.warning("feed %s failed: %s", feed_url, exc)
                continue

            entries = parsed.get("entries", [])
            if not entries:
                self.log.warning("feed %s returned no entries", feed_url)
                continue
            self.log.info("feed %s: %d entries", feed_url, len(entries))

            for entry in entries:
                yield {
                    "_kind": "rss",
                    "_feed": feed_url,
                    "_feed_title": (parsed.get("feed") or {}).get("title"),
                    **{k: _jsonable(v) for k, v in entry.items()},
                }
            self.checkpoint.set(f"last_seen.{feed_url}", datetime.now(timezone.utc).isoformat())
            self.record_manifest(_feed_key(feed_url), url=feed_url, rows=len(entries))

    def _fetch_newsapi(self, config: dict) -> Iterator[dict]:
        if not config.get("enabled", True):
            return
        if not self.settings.newsapi_key:
            # Graceful degradation, loudly. This is an acceptance criterion.
            self.log.warning(
                "NEWSAPI_KEY absent: continuing with RSS only. NewsAPI is optional "
                "(free tier ~100 req/day, ~24h delayed, non-commercial licence)."
            )
            return

        http = self.http()
        page_size = int(config.get("page_size", 100))
        max_pages = int(config.get("max_pages", 1))
        for topic in topics_config().get("topics", []):
            query = " OR ".join(f'"{k}"' for k in topic.get("keywords", [])[:5])
            if not query:
                continue
            for page in range(1, max_pages + 1):
                try:
                    response = http.get(
                        NEWSAPI_URL,
                        params={
                            "q": query,
                            "pageSize": page_size,
                            "page": page,
                            "language": "en",
                            "sortBy": "publishedAt",
                        },
                        headers={"X-Api-Key": self.settings.newsapi_key},
                    )
                    payload = response.json()
                except Exception as exc:
                    self.log.warning("newsapi query failed for %s: %s", topic.get("id"), exc)
                    break
                if payload.get("status") != "ok":
                    self.log.warning("newsapi error: %s", payload.get("message"))
                    break
                articles = payload.get("articles", [])
                for article in articles:
                    yield {"_kind": "newsapi", "_topic": topic.get("id"), **article}
                if len(articles) < page_size:
                    break

    # --- map -------------------------------------------------------------
    def to_record(self, raw: dict) -> Record | None:
        if raw.get("_kind") == "newsapi":
            return self._newsapi_to_record(raw)
        return self._rss_to_record(raw)

    def _rss_to_record(self, raw: dict) -> Record | None:
        url = raw.get("link") or raw.get("id")
        if not url:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None

        timestamp = _entry_time(raw)
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, url)
            return None

        title = (raw.get("title") or "").strip()
        summary = _entry_summary(raw)
        body = self._maybe_fulltext(url)
        if body:
            text = f"{title}\n\n{body}" if title else body
        else:
            text = "\n\n".join(part for part in (title, summary) if part)

        if not text.strip():
            self.drop(DropReason.EMPTY_TEXT, url)
            return None

        domain = resolve_domain(url) or "unknown"
        fields = build_text_fields(text, is_html=True, structured_urls=[url])

        return Record(
            native_id=_article_id(url),
            source="news",
            source_detail=domain,
            content_type="article",
            author_id=make_id("news", domain),
            author_handle=raw.get("author") or raw.get("_feed_title") or domain,
            timestamp=timestamp,
            # No news feed reports readership, so every engagement metric is
            # genuinely unmeasured. Zeroes here would be a lie.
            engagement=EngagementMetrics.unavailable(),
            media_urls=_media_from_entry(raw),
            raw={
                "api": "rss",
                "feed": raw.get("_feed"),
                "feed_title": raw.get("_feed_title"),
                "url": url,
                "title": title,
                "summary": summary,
                "tags": [t.get("term") for t in raw.get("tags", []) if isinstance(t, dict)],
                "fulltext_extracted": bool(body),
            },
            **fields,
        )

    def _newsapi_to_record(self, raw: dict) -> Record | None:
        url = raw.get("url")
        if not url:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None
        timestamp = _parse_iso(raw.get("publishedAt"))
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, url)
            return None

        title = (raw.get("title") or "").strip()
        # NewsAPI truncates `content` at ~200 chars on the free tier; keep the
        # description too rather than pretending the truncation is the article.
        parts = [title, raw.get("description") or "", raw.get("content") or ""]
        text = "\n\n".join(p.strip() for p in parts if p and p.strip())
        if not text:
            self.drop(DropReason.EMPTY_TEXT, url)
            return None

        domain = resolve_domain(url) or "unknown"
        fields = build_text_fields(text, structured_urls=[url])
        return Record(
            native_id=_article_id(url),
            source="news",
            source_detail=domain,
            content_type="article",
            author_id=make_id("news", domain),
            author_handle=raw.get("author") or (raw.get("source") or {}).get("name") or domain,
            timestamp=timestamp,
            engagement=EngagementMetrics.unavailable(),
            media_urls=[raw["urlToImage"]] if raw.get("urlToImage") else [],
            raw={
                "api": "newsapi",
                "topic": raw.get("_topic"),
                "url": url,
                "title": title,
                "description": raw.get("description"),
                "content_truncated": raw.get("content"),
                "source": raw.get("source"),
            },
            **fields,
        )

    # --- full text -------------------------------------------------------
    def _maybe_fulltext(self, url: str) -> str | None:
        """Extract article text with trafilatura, within budget and robots.txt.

        Failure keeps the record with title+summary rather than dropping it: a
        headline is still a claim, and dropping it would bias the corpus toward
        outlets with scraper-friendly markup.
        """
        config = sources_config().get(self.name, {}).get("fulltext", {})
        if not config.get("enabled") or self._fulltext_budget <= 0:
            return None
        try:
            import trafilatura
        except ImportError:
            self.log.warning("trafilatura not installed; keeping title+summary only")
            self._fulltext_budget = 0
            return None

        if not self._robots_allows(url):
            self.note("robots_disallowed", url)
            return None

        self._fulltext_budget -= 1
        time.sleep(float(config.get("request_delay_seconds", 1.0)))
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                self.note("fulltext_fetch_failed", url)
                return None
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        except Exception as exc:
            self.note("fulltext_error", f"{url}: {exc}")
            return None
        if not text:
            self.note("fulltext_empty", url)
            return None
        return text

    def _robots_allows(self, url: str) -> bool:
        """Honour robots.txt per host, fetched once and cached for the run."""
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            parser = RobotFileParser()
            parser.set_url(f"{host}/robots.txt")
            try:
                parser.read()
            except Exception:
                parser = None  # unreachable robots.txt: treat as unknown, not as allow-all
            self._robots[host] = parser
        parser = self._robots[host]
        if parser is None:
            return False
        try:
            return parser.can_fetch(self.settings.user_agent, url)
        except Exception:  # pragma: no cover - malformed robots.txt
            return False


# --- module helpers -------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, time.struct_time):
        return list(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _entry_time(entry: dict) -> datetime | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return datetime(*[int(v) for v in list(parsed)[:6]], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    for key in ("published", "updated", "created"):
        stamp = _parse_iso(entry.get(key))
        if stamp:
            return stamp
    return None


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            parsed = parsedate_to_datetime(str(value))
        except (TypeError, ValueError):
            return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _entry_summary(entry: dict) -> str:
    content = entry.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and first.get("value"):
            return str(first["value"])
    return str(entry.get("summary") or entry.get("description") or "")


def _media_from_entry(entry: dict) -> list[str]:
    urls: list[str] = []
    for item in entry.get("media_content", []) or []:
        if isinstance(item, dict) and item.get("url"):
            urls.append(item["url"])
    for item in entry.get("links", []) or []:
        if isinstance(item, dict) and str(item.get("type", "")).startswith(("image/", "video/")):
            if item.get("href"):
                urls.append(item["href"])
    for item in entry.get("enclosures", []) or []:
        if isinstance(item, dict) and item.get("href"):
            urls.append(item["href"])
    return list(dict.fromkeys(urls))


def _article_id(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.netloc}{parts.path}".strip("/").lower() or url


def _feed_key(url: str) -> str:
    return urlsplit(url).netloc.lower() or url
