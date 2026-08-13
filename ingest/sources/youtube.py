"""YouTube Data API v3: video metadata and comment threads.

Quota is the binding constraint, not rate. The daily budget is 10,000 units per
UTC day and the costs are wildly asymmetric::

    search.list          100 units   <- discovery; budgeted hard
    videos.list            1 unit    <- hydration; effectively free
    commentThreads.list    1 unit    <- comments; effectively free

So the strategy is: spend a small, explicit number of searches to discover video
ids, then hydrate and harvest comments cheaply. Every call is charged to the
ledger in :mod:`ingest.checkpoint` *before* it is made, and the run stops
cleanly when the budget is gone. Nothing wastes a day faster than discovering at
11am that a loop spent 10,000 units on searches.

What this source gives you: view/like/comment counts (the only source here that
reports views at all), channel identity, and threaded comments.
What it costs: 10,000 units/UTC day, hard.
What it cannot tell you: who watched, or anything about a video whose comments
are disabled -- which is common on exactly the political content of interest, so
comment coverage is non-random and the EDA says so.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from ingest.checkpoint import QuotaExhausted, QuotaLedger
from ingest.config import sources_config, topics_config
from ingest.normalize import build_text_fields
from ingest.schema import Author, DropReason, EngagementMetrics, Record, make_id
from ingest.sources.base import BaseSource, SourceUnavailable

WATCH_URL = "https://www.youtube.com/watch?v={video_id}"

COST_SEARCH = 100
COST_VIDEOS = 1
COST_COMMENTS = 1


class YouTubeSource(BaseSource):
    name = "youtube"
    source = "youtube"
    requires_package = "googleapiclient"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.ledger = QuotaLedger(
            self.checkpoint,
            daily_limit=int(self.settings.youtube_daily_quota_units),
        )

    # --- client ----------------------------------------------------------
    def _client(self) -> Any:
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - guarded by preflight
            raise SourceUnavailable(f"google-api-python-client is not installed: {exc}") from exc
        if not self.settings.youtube_api_key:
            raise SourceUnavailable("youtube: YOUTUBE_API_KEY absent; skipping")
        # API key auth only: everything read here is public, so OAuth would add
        # a consent flow for no additional access.
        return build(
            "youtube",
            "v3",
            developerKey=self.settings.youtube_api_key,
            cache_discovery=False,
        )

    # --- fetch -----------------------------------------------------------
    def fetch(self) -> Iterator[dict]:
        config = sources_config().get(self.name, {})
        client = self._client()
        self.log.info("%s", self.ledger.summary())

        video_ids = self._discover(client, config.get("discovery", {}))
        if not video_ids:
            self.log.warning("no video ids discovered or carried over; nothing to hydrate")
            return

        videos = list(self._hydrate(client, video_ids, config.get("hydration", {})))
        if videos:
            path = self.save_raw_payload("videos", videos)
            self.record_manifest(
                "videos",
                path=path,
                url="https://www.googleapis.com/youtube/v3/videos",
                rows=len(videos),
            )
        yield from videos

        comments_config = config.get("comments", {})
        if comments_config.get("enabled", True):
            harvested: list[dict] = []
            for video in videos[: int(comments_config.get("max_videos", 50))]:
                for comment in self._comments(client, video["id"], comments_config):
                    harvested.append(comment)
                    yield comment
            if harvested:
                path = self.save_raw_payload("comments", harvested)
                self.record_manifest(
                    "comments",
                    path=path,
                    url="https://www.googleapis.com/youtube/v3/commentThreads",
                    rows=len(harvested),
                )

    def _discover(self, client: Any, config: dict) -> list[str]:
        """search.list at 100 units a call, capped by config *and* by .env.

        Ids are checkpointed so tomorrow's run can skip discovery entirely and
        spend its whole budget on the cheap calls.
        """
        pending: list[str] = list(self.checkpoint.get("pending_video_ids", []))
        max_searches = min(
            int(self.options.get("max_searches") or config.get("max_searches_per_run", 4)),
            int(self.settings.youtube_max_searches_per_day),
        )
        already = self.ledger.count("search.list")
        allowed = max(0, max_searches - already)
        if allowed == 0:
            self.log.info(
                "search budget for today already spent (%d searches); "
                "using %d carried-over video ids",
                already,
                len(pending),
            )
            return pending

        queries: list[str] = list(self.options.get("queries") or [])
        if not queries:
            for topic in topics_config().get("topics", []):
                queries.extend(topic.get("youtube_queries", []))
        queries = list(dict.fromkeys(queries))[:allowed]

        published_after = (
            datetime.now(timezone.utc)
            - timedelta(days=int(config.get("published_within_days", 30)))
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        found: list[str] = []
        for query in queries:
            try:
                self.ledger.charge(COST_SEARCH, call="search.list")
            except QuotaExhausted as exc:
                self.log.warning("%s", exc)
                break
            self.log.info("search.list %r (100 units)", query)
            response = (
                client.search()
                .list(
                    q=query,
                    part="id",
                    type="video",
                    maxResults=int(config.get("results_per_search", 25)),
                    order=config.get("order", "relevance"),
                    publishedAfter=published_after,
                )
                .execute()
            )
            for item in response.get("items", []):
                video_id = (item.get("id") or {}).get("videoId")
                if video_id:
                    found.append(video_id)

        ids = list(dict.fromkeys(pending + found))
        self.checkpoint.set("pending_video_ids", ids)
        self.log.info("%d video ids queued for hydration; %s", len(ids), self.ledger.summary())
        return ids

    def _hydrate(self, client: Any, video_ids: list[str], config: dict) -> Iterator[dict]:
        """videos.list at 1 unit per call, 50 ids per call."""
        batch_size = int(config.get("videos_per_batch", 50))
        remaining = list(video_ids)
        for start in range(0, len(video_ids), batch_size):
            batch = video_ids[start : start + batch_size]
            self.ledger.charge(COST_VIDEOS, call="videos.list")
            response = (
                client.videos()
                .list(part="snippet,statistics,contentDetails", id=",".join(batch), maxResults=50)
                .execute()
            )
            for item in response.get("items", []):
                yield {"_kind": "video", **item}
            remaining = [v for v in remaining if v not in batch]
            self.checkpoint.set("pending_video_ids", remaining)

    def _comments(self, client: Any, video_id: str, config: dict) -> Iterator[dict]:
        """commentThreads.list at 1 unit per page."""
        page_token = self.checkpoint.get(f"comment_page.{video_id}")
        for page in range(int(config.get("max_comment_pages_per_video", 2))):
            self.ledger.charge(COST_COMMENTS, call="commentThreads.list")
            try:
                response = (
                    client.commentThreads()
                    .list(
                        part="snippet,replies",
                        videoId=video_id,
                        maxResults=int(config.get("page_size", 100)),
                        order=config.get("order", "relevance"),
                        textFormat="plainText",
                        pageToken=page_token,
                    )
                    .execute()
                )
            except Exception as exc:
                # Comments disabled is the single most common failure, and it is
                # not random: it correlates with exactly the political content
                # this project studies. Count it so the EDA can say so.
                self.note("comments_unavailable", f"{video_id}: {type(exc).__name__}")
                self.log.debug("comments unavailable for %s: %s", video_id, exc)
                return
            for item in response.get("items", []):
                top = (item.get("snippet") or {}).get("topLevelComment") or {}
                yield {"_kind": "video_comment", "_video_id": video_id, **top}
                for reply in (item.get("replies") or {}).get("comments", []) or []:
                    yield {"_kind": "video_comment", "_video_id": video_id, **reply}
            page_token = response.get("nextPageToken")
            self.checkpoint.set(f"comment_page.{video_id}", page_token)
            if not page_token:
                break
            self.log.debug("video %s: comment page %d done", video_id, page + 1)

    # --- map -------------------------------------------------------------
    def to_record(self, raw: dict) -> Record | None:
        if raw.get("_kind") == "video_comment":
            return self._comment_to_record(raw)
        return self._video_to_record(raw)

    def _video_to_record(self, raw: dict) -> Record | None:
        video_id = str(raw.get("id") or "").strip()
        if not video_id:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None
        snippet = raw.get("snippet") or {}
        stats = raw.get("statistics") or {}

        timestamp = _parse_dt(snippet.get("publishedAt"))
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, video_id)
            return None

        title = (snippet.get("title") or "").strip()
        description = (snippet.get("description") or "").strip()
        text = "\n\n".join(part for part in (title, description) if part)
        if not text:
            self.drop(DropReason.EMPTY_TEXT, video_id)
            return None

        channel_id = str(snippet.get("channelId") or "unknown")
        fields = build_text_fields(text, structured_tags=snippet.get("tags"))
        fields["lang"] = _lang(
            snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage")
        ) or fields.get("lang")

        return Record(
            native_id=video_id,
            source="youtube",
            source_detail=channel_id,
            content_type="video",
            author_id=make_id("youtube", channel_id),
            author_handle=snippet.get("channelTitle"),
            timestamp=timestamp,
            conversation_id=make_id("youtube", video_id),
            engagement=EngagementMetrics(
                likes=_as_int(stats.get("likeCount")),
                # YouTube removed public dislikes and never exposed shares.
                shares=None,
                replies=_as_int(stats.get("commentCount")),
                views=_as_int(stats.get("viewCount")),
            ),
            # The video URL itself: this is the hook for the Phase 2 deepfake
            # module, which needs something fetchable, not a thumbnail.
            media_urls=[WATCH_URL.format(video_id=video_id)],
            raw={
                "kind": "video",
                "channel_id": channel_id,
                "channel_title": snippet.get("channelTitle"),
                "title": title,
                "description": description,
                "tags": snippet.get("tags"),
                "category_id": snippet.get("categoryId"),
                "duration": (raw.get("contentDetails") or {}).get("duration"),
                "thumbnails": snippet.get("thumbnails"),
                "statistics": stats,
            },
            **fields,
        )

    def _comment_to_record(self, raw: dict) -> Record | None:
        comment_id = str(raw.get("id") or "").strip()
        snippet = raw.get("snippet") or {}
        if not comment_id:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None

        timestamp = _parse_dt(snippet.get("publishedAt"))
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, comment_id)
            return None

        text = (snippet.get("textOriginal") or snippet.get("textDisplay") or "").strip()
        if not text:
            self.drop(DropReason.EMPTY_TEXT, comment_id)
            return None

        author_channel = (snippet.get("authorChannelId") or {}).get("value") or "unknown"
        video_id = snippet.get("videoId") or raw.get("_video_id")
        parent = snippet.get("parentId") or video_id
        is_html = bool(snippet.get("textDisplay")) and not snippet.get("textOriginal")
        fields = build_text_fields(text, is_html=is_html)

        return Record(
            native_id=comment_id,
            source="youtube",
            source_detail=str(snippet.get("channelId") or video_id or "unknown"),
            content_type="video_comment",
            author_id=make_id("youtube", author_channel),
            author_handle=snippet.get("authorDisplayName"),
            timestamp=timestamp,
            parent_id=make_id("youtube", parent) if parent else None,
            conversation_id=make_id("youtube", video_id) if video_id else None,
            engagement=EngagementMetrics(
                likes=_as_int(snippet.get("likeCount")),
                shares=None,
                replies=None,  # only known on the thread, not on the comment
                views=None,
            ),
            raw={
                "kind": "video_comment",
                "video_id": video_id,
                "parent_id": snippet.get("parentId"),
                "author_channel_id": author_channel,
                "author_display_name": snippet.get("authorDisplayName"),
                "updated_at": snippet.get("updatedAt"),
                "text_display": snippet.get("textDisplay"),
            },
            **fields,
        )

    def to_author(self, raw: dict, record: Record) -> Author | None:
        snippet = raw.get("snippet") or {}
        if raw.get("_kind") == "video_comment":
            handle = snippet.get("authorDisplayName")
            extra = {"author_profile_image_url": snippet.get("authorProfileImageUrl")}
        else:
            handle = snippet.get("channelTitle")
            extra = {"channel_id": snippet.get("channelId")}
        return Author(
            author_id=record.author_id,
            source="youtube",
            handle=handle,
            # Channel creation date needs a separate channels.list call; not
            # spent here. Null rather than guessed.
            created_at=None,
            followers=None,
            following=None,
            post_count=1,
            first_seen=record.timestamp,
            last_seen=record.timestamp,
            raw=extra,
        )


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _lang(value: Any) -> str | None:
    if not value:
        return None
    return str(value).split("-")[0].lower() or None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
