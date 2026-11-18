"""YouTube through yt-dlp: video metadata, subtitles and comments, no API key.

yt-dlp reads the public watch pages, so there is no credential and no daily
quota. What replaces the quota is YouTube's own tolerance for scraping, which is
unmetered but not unlimited: a burst of extractions can be answered with a bot
check, and that surfaces here as a counted per-video failure, not a crash.

Discovery is a flat ``ytsearchN:<query>`` (ids only, one request per query).
Each id is then extracted in full without downloading any media. A full
extraction costs one request per video and is what yields the upload timestamp,
the statistics, the caption tracks and, optionally, comments.

Subtitles are the reason this adapter is worth more than the Data API one: a
title and description are a few hundred characters, a transcript is thousands.
The record text is ``title + description + transcript`` and ``raw`` says which
caption track supplied it.

What this source gives you: view/like/comment counts (the only source here that
reports views at all), channel identity, transcripts, and comments.
What it cannot tell you: who watched, a video whose comments are disabled, or
whether an automatic caption is right. Auto captions are speech recognition and
are noisier on Hindi and on code-mixed speech, hence ``subtitle_kind`` in ``raw``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from ingest.config import sources_config, topics_config
from ingest.normalize import build_text_fields
from ingest.schema import Author, DropReason, EngagementMetrics, Record, make_id
from ingest.sources.base import BaseSource, SourceUnavailable

WATCH_URL = "https://www.youtube.com/watch?v={video_id}"

#: caption formats the parser understands, best first. json3 is structured, so
#: it needs no tag stripping; vtt is the fallback and is what most tracks offer.
SUBTITLE_EXTS = ("json3", "vtt")

#: reason codes for the paths that skip or degrade a video without a record drop
FLAG_OUTSIDE_WINDOW = "outside_published_window"
FLAG_EXTRACT_FAILED = "extract_failed"
FLAG_SEARCH_FAILED = "search_failed"
FLAG_SUBTITLE_FETCH_FAILED = "subtitle_fetch_failed"
FLAG_NO_SUBTITLES = "no_subtitles"
FLAG_WITH_SUBTITLES = "with_subtitles"
FLAG_COMMENTS_UNAVAILABLE = "comments_unavailable"


class YouTubeSource(BaseSource):
    name = "youtube"
    source = "youtube"
    requires_package = "yt_dlp"

    # --- client ----------------------------------------------------------
    def _ydl_options(self, config: dict, *, flat: bool, comments: int = 0) -> dict[str, Any]:
        discovery = config.get("discovery", {})
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noprogress": True,
            "socket_timeout": int(config.get("socket_timeout_seconds", 20)),
            "retries": 2,
            "user_agent": self.settings.user_agent,
        }
        if flat:
            opts["extract_flat"] = "in_playlist"
        # geo_bypass_country sends a spoofed X-Forwarded-For so a video that is
        # geo-blocked outside that country can be read. It does not rank search
        # results for that country; see the note in configs/sources.yaml.
        region = discovery.get("region_code")
        if region:
            opts["geo_bypass_country"] = str(region).upper()
        youtube_args: dict[str, list[str]] = {}
        # `lang` is the hl parameter of the API response (interface language),
        # not a relevance filter. Omitted entirely when unset.
        relevance_language = discovery.get("relevance_language")
        if relevance_language:
            youtube_args["lang"] = [str(relevance_language)]
        if comments > 0:
            opts["getcomments"] = True
            # max-comments,max-parents,max-replies,max-replies-per-thread
            # (one list item each: the API form does not split on commas)
            youtube_args["max_comments"] = [str(comments), "all", "all", "all"]
        if youtube_args:
            opts["extractor_args"] = {"youtube": youtube_args}
        return opts

    def _client(self, config: dict, *, flat: bool, comments: int = 0) -> Any:
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover - guarded by preflight
            raise SourceUnavailable(f"yt-dlp is not installed: {exc}") from exc
        return yt_dlp.YoutubeDL(self._ydl_options(config, flat=flat, comments=comments))

    # --- fetch -----------------------------------------------------------
    def fetch(self) -> Iterator[dict]:
        config = sources_config().get(self.name, {})
        discovery = config.get("discovery", {})
        comments_config = config.get("comments", {})
        comments_per_video = (
            int(comments_config.get("max_comments_per_video", 100))
            if comments_config.get("enabled", True)
            else 0
        )
        max_comment_videos = int(comments_config.get("max_videos", 50))

        search = self._client(config, flat=True)
        detail = self._client(config, flat=False)
        detail_with_comments = (
            self._client(config, flat=False, comments=comments_per_video)
            if comments_per_video
            else detail
        )

        cutoff = datetime.now(timezone.utc) - timedelta(
            days=int(discovery.get("published_within_days", 30))
        )
        seen: set[str] = set()
        extracted = 0
        archived: list[dict] = []

        # Search lazily, one query at a time, so --limit stops the run before
        # the remaining queries are spent on videos that would be thrown away.
        for query in self._queries(discovery):
            for video_id in self._search(search, query, discovery):
                if video_id in seen:
                    continue
                seen.add(video_id)
                client = detail_with_comments if extracted < max_comment_videos else detail
                info = self._extract(client, video_id)
                if info is None:
                    continue
                extracted += 1
                published = _parse_timestamp(info)
                if published is not None and published < cutoff:
                    self.note(FLAG_OUTSIDE_WINDOW, video_id)
                    continue
                payload = self._payload(detail, info, config)
                archived.append(payload)
                yield payload
                yield from self._comment_payloads(payload, info, comments_per_video)

        if archived:
            path = self.save_raw_payload("videos", archived)
            self.record_manifest(
                "videos",
                path=path,
                url="https://www.youtube.com/results (yt-dlp)",
                rows=len(archived),
            )

    def _queries(self, config: dict) -> list[str]:
        queries: list[str] = list(self.options.get("queries") or [])
        if not queries:
            for topic in topics_config().get("topics", []):
                queries.extend(topic.get("youtube_queries", []))
        queries = list(dict.fromkeys(queries))
        cap = int(self.options.get("max_searches") or config.get("max_searches_per_run", 12))
        return queries[:cap]

    def _search(self, client: Any, query: str, config: dict) -> list[str]:
        """Video ids for one query. A failed search is counted and skipped."""
        count = int(config.get("results_per_search", 25))
        prefix = "ytsearchdate" if str(config.get("order", "relevance")) == "date" else "ytsearch"
        self.log.info("%s%d: %r", prefix, count, query)
        try:
            result = client.extract_info(f"{prefix}{count}:{query}", download=False)
        except Exception as exc:
            self.note(FLAG_SEARCH_FAILED, f"{query!r}: {type(exc).__name__}")
            self.log.warning("search failed for %r: %s", query, exc)
            return []
        ids: list[str] = []
        for entry in (result or {}).get("entries") or []:
            video_id = (entry or {}).get("id")
            if video_id:
                ids.append(str(video_id))
        return ids

    def _extract(self, client: Any, video_id: str) -> dict | None:
        try:
            return client.extract_info(WATCH_URL.format(video_id=video_id), download=False)
        except Exception as exc:
            # Private, removed, age-gated and bot-checked videos all land here.
            self.note(FLAG_EXTRACT_FAILED, f"{video_id}: {type(exc).__name__}")
            self.log.debug("extraction failed for %s: %s", video_id, exc)
            return None

    def _payload(self, client: Any, info: dict, config: dict) -> dict:
        """Flatten yt-dlp's info dict to the fields the mapper reads, plus the transcript."""
        payload = {"_kind": "video", **{key: info.get(key) for key in INFO_FIELDS}}
        payload["_subtitle"] = self._subtitle(client, info)
        return payload

    def _comment_payloads(self, payload: dict, info: dict, limit: int) -> Iterator[dict]:
        if not limit:
            return
        comments = info.get("comments")
        if not comments:
            # Disabled comments are the usual cause, and they correlate with
            # exactly the political content this project studies.
            self.note(FLAG_COMMENTS_UNAVAILABLE, str(payload.get("id")))
            return
        for comment in comments[:limit]:
            yield {
                "_kind": "video_comment",
                "_video_id": payload.get("id"),
                "_channel_id": payload.get("channel_id"),
                **{key: comment.get(key) for key in COMMENT_FIELDS},
            }

    # --- subtitles -------------------------------------------------------
    def _subtitle(self, client: Any, info: dict) -> dict:
        """Pick a caption track, download it and return ``{lang, kind, text}``.

        Never raises: a video with no usable track is a normal outcome.
        """
        choice = select_subtitle_track(info)
        if choice is None:
            return {"lang": None, "kind": "none", "text": ""}
        lang, kind, track = choice
        try:
            body = client.urlopen(track["url"]).read().decode("utf-8", errors="replace")
            text = parse_subtitles(body, track.get("ext", "vtt"))
        except Exception as exc:
            self.note(FLAG_SUBTITLE_FETCH_FAILED, f"{info.get('id')}: {type(exc).__name__}")
            self.log.debug("subtitle fetch failed for %s: %s", info.get("id"), exc)
            return {"lang": lang, "kind": "none", "text": ""}
        if not text:
            return {"lang": lang, "kind": "none", "text": ""}
        return {"lang": lang, "kind": kind, "text": text}

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

        timestamp = _parse_timestamp(raw)
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, video_id)
            return None

        title = (raw.get("title") or "").strip()
        description = (raw.get("description") or "").strip()
        subtitle = raw.get("_subtitle") or {}
        transcript = (subtitle.get("text") or "").strip()
        subtitle_kind = subtitle.get("kind") if transcript else "none"
        text = "\n\n".join(part for part in (title, description, transcript) if part)
        if not text:
            self.drop(DropReason.EMPTY_TEXT, video_id)
            return None

        if transcript:
            self.note(FLAG_WITH_SUBTITLES, video_id)
            self.note(f"subtitles_{subtitle_kind}", video_id)
        else:
            self.note(FLAG_NO_SUBTITLES, video_id)

        channel_id = str(raw.get("channel_id") or raw.get("uploader_id") or "unknown")
        channel_title = raw.get("channel") or raw.get("uploader")
        fields = build_text_fields(text, structured_tags=raw.get("tags"))
        fields["lang"] = _lang(raw.get("language")) or fields.get("lang")

        return Record(
            native_id=video_id,
            source="youtube",
            source_detail=channel_id,
            content_type="video",
            author_id=make_id("youtube", channel_id),
            author_handle=channel_title,
            timestamp=timestamp,
            conversation_id=make_id("youtube", video_id),
            engagement=EngagementMetrics(
                likes=_as_int(raw.get("like_count")),
                # YouTube removed public dislikes and never exposed shares.
                shares=None,
                replies=_as_int(raw.get("comment_count")),
                views=_as_int(raw.get("view_count")),
            ),
            # The watch page of the video itself, not a thumbnail.
            media_urls=[WATCH_URL.format(video_id=video_id)],
            raw={
                "kind": "video",
                "channel_id": channel_id,
                "channel_title": channel_title,
                "title": title,
                "description": description,
                "tags": raw.get("tags"),
                "categories": raw.get("categories"),
                "duration_seconds": raw.get("duration"),
                "language": raw.get("language"),
                "subtitle_lang": subtitle.get("lang") if transcript else None,
                "subtitle_kind": subtitle_kind,
                "statistics": {
                    "view_count": raw.get("view_count"),
                    "like_count": raw.get("like_count"),
                    "comment_count": raw.get("comment_count"),
                },
            },
            **fields,
        )

    def _comment_to_record(self, raw: dict) -> Record | None:
        comment_id = str(raw.get("id") or "").strip()
        if not comment_id:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None

        timestamp = _parse_timestamp(raw)
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, comment_id)
            return None

        text = (raw.get("text") or "").strip()
        if not text:
            self.drop(DropReason.EMPTY_TEXT, comment_id)
            return None

        video_id = raw.get("_video_id")
        # yt-dlp marks a top-level comment's parent as the literal "root".
        parent_native = raw.get("parent")
        parent = video_id if not parent_native or parent_native == "root" else parent_native
        author_channel = raw.get("author_id") or "unknown"
        fields = build_text_fields(text)

        return Record(
            native_id=comment_id,
            source="youtube",
            source_detail=str(raw.get("_channel_id") or video_id or "unknown"),
            content_type="video_comment",
            author_id=make_id("youtube", author_channel),
            author_handle=raw.get("author"),
            timestamp=timestamp,
            parent_id=make_id("youtube", parent) if parent else None,
            conversation_id=make_id("youtube", video_id) if video_id else None,
            engagement=EngagementMetrics(
                likes=_as_int(raw.get("like_count")),
                shares=None,
                replies=None,  # only known on the thread, not on the comment
                views=None,
            ),
            raw={
                "kind": "video_comment",
                "video_id": video_id,
                "parent_id": None if parent_native == "root" else parent_native,
                "author_channel_id": author_channel,
                "author_display_name": raw.get("author"),
            },
            **fields,
        )

    def to_author(self, raw: dict, record: Record) -> Author | None:
        if raw.get("_kind") == "video_comment":
            handle = raw.get("author")
            extra = {"author_channel_id": raw.get("author_id")}
        else:
            handle = raw.get("channel") or raw.get("uploader")
            extra = {"channel_id": raw.get("channel_id")}
        return Author(
            author_id=record.author_id,
            source="youtube",
            handle=handle,
            # Channel creation date is not in a watch page. Null rather than guessed.
            created_at=None,
            followers=_as_int(raw.get("channel_follower_count")),
            following=None,
            post_count=1,
            first_seen=record.timestamp,
            last_seen=record.timestamp,
            raw=extra,
        )


#: the yt-dlp info fields the mapper reads; everything else is dropped on purpose
INFO_FIELDS = (
    "id",
    "title",
    "description",
    "channel_id",
    "channel",
    "uploader",
    "uploader_id",
    "channel_follower_count",
    "timestamp",
    "upload_date",
    "tags",
    "categories",
    "duration",
    "view_count",
    "like_count",
    "comment_count",
    "language",
)

COMMENT_FIELDS = ("id", "parent", "text", "author", "author_id", "like_count", "timestamp")


# --- subtitles ------------------------------------------------------------


def _lang_matches(key: str, lang: str) -> bool:
    """``en`` matches ``en``, ``en-US`` and ``en-orig``; never ``eng`` or ``hi``."""
    return key == lang or key.startswith(f"{lang}-")


def _best_format(formats: Any) -> dict | None:
    by_ext = {f.get("ext"): f for f in formats or [] if f.get("url")}
    for ext in SUBTITLE_EXTS:
        if ext in by_ext:
            return by_ext[ext]
    return None


def select_subtitle_track(info: dict) -> tuple[str, str, dict] | None:
    """Choose ``(lang, kind, format)``: manual in the video's language, then auto.

    A manual track is a human's work and is always preferred. Automatic captions
    come as one speech-recognition track in the spoken language (``xx-orig``)
    plus a machine translation into every other language. The translations are
    never used: they would put a different language's text under the record.

    When the video's language is unknown, a manual or ``-orig`` track is taken
    only if the video has exactly one. Videos list an ``-orig`` track for many
    languages at once, so picking among several would be a guess.
    """
    lang = (info.get("language") or "").split("-")[0].lower() or None
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    def pick(tracks: dict, keys: list[str]) -> tuple[str, dict] | None:
        for key in keys:
            fmt = _best_format(tracks.get(key))
            if fmt is not None:
                return key, fmt
        return None

    if lang:
        found = pick(manual, [k for k in manual if _lang_matches(k, lang)])
        if found:
            return found[0], "manual", found[1]
        # Prefer the recognised original over a same-language regional variant.
        keys = sorted(
            (k for k in auto if _lang_matches(k, lang)), key=lambda k: not k.endswith("-orig")
        )
        found = pick(auto, keys)
        if found:
            return found[0], "auto", found[1]
        return None

    if len(manual) == 1:
        found = pick(manual, list(manual))
        if found:
            return found[0], "manual", found[1]
    originals = [k for k in auto if k.endswith("-orig")]
    if len(originals) == 1:
        found = pick(auto, originals)
        if found:
            return found[0], "auto", found[1]
    return None


_VTT_TIMING = re.compile(r"^\s*(\d+:)?\d{2}:\d{2}\.\d{3}\s+-->")
_VTT_TAG = re.compile(r"<[^>]+>")
_VTT_HEADER = ("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION")


def _dedupe_consecutive(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        line = " ".join(line.split())
        if line and (not out or out[-1] != line):
            out.append(line)
    return out


def parse_vtt(body: str) -> str:
    """WebVTT to plain text.

    Auto captions are rolling: each cue repeats the previous cue's last line and
    adds a new one, so a transcript read naively says everything twice. Cue
    timings, inline word timestamps and styling tags are removed first, then
    consecutive identical lines are collapsed.
    """
    lines: list[str] = []
    in_cue = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if _VTT_TIMING.match(line):
            in_cue = True
            continue
        if not raw_line:
            # Only a truly empty line ends a cue. YouTube writes a lone space
            # between the timing line and the text, which must not.
            in_cue = False
            continue
        if not line or not in_cue or line.startswith(_VTT_HEADER):
            continue
        lines.append(unescape_entities(_VTT_TAG.sub("", line)))
    return "\n".join(_dedupe_consecutive(lines))


def parse_json3(body: str) -> str:
    """YouTube json3 captions to plain text, one line per caption event."""
    data = json.loads(body)
    lines: list[str] = []
    for event in data.get("events") or []:
        segs = event.get("segs")
        if not segs:
            continue
        # Auto-caption events can contain only a newline seg; those collapse to "".
        lines.extend("".join(seg.get("utf8", "") for seg in segs).split("\n"))
    return "\n".join(_dedupe_consecutive(lines))


def parse_subtitles(body: str, ext: str) -> str:
    if ext == "json3":
        return parse_json3(body)
    return parse_vtt(body)


def unescape_entities(text: str) -> str:
    import html

    return html.unescape(text).replace(" ", " ")


# --- helpers --------------------------------------------------------------


def _parse_timestamp(raw: dict) -> datetime | None:
    """yt-dlp gives an epoch ``timestamp`` when it has one, else a ``YYYYMMDD`` date."""
    value = raw.get("timestamp")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    date = raw.get("upload_date")
    if date:
        try:
            return datetime.strptime(str(date), "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


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
