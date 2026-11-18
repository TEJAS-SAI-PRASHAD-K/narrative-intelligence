"""Telegram: public broadcast channels, read over MTProto with Telethon.

Why this source matters for the thesis: political forwards circulate on Telegram
long before they reach the platforms the rest of the corpus covers, and a
channel post carries two measured counters (views and forwards) that news
records never have.

What this source gives you: channel posts with view and forward counts, the
reaction total and reply count where the channel exposes them, forward
provenance (which channel a post was copied from) and edit timestamps.
What it costs: a free API id and hash from my.telegram.org plus a pre-generated
``StringSession``, so no interactive login happens at fetch time. Telegram
answers aggressive reads with flood waits, so channels are read one at a time
behind a token bucket.
What it cannot tell you: anything about private groups or channels, who viewed
a post, or likes (Telegram has reactions, not likes, so ``likes`` is the
reaction total and ``None`` when the channel hides reactions).

Telethon is asynchronous. ``fetch`` drives it on a private event loop and
flattens every message into a plain dict before yielding, so ``to_record`` is
testable from a recorded fixture without Telethon installed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from ingest.config import sources_config
from ingest.normalize import build_text_fields
from ingest.ratelimit import TokenBucket
from ingest.schema import Author, DropReason, EngagementMetrics, Record, make_id
from ingest.sources.base import BaseSource, SourceUnavailable

PERMALINK = "https://t.me/{username}/{message_id}"

_DEFAULT_MESSAGES_PER_CHANNEL = 200
# Telegram tolerates a few resolve and history calls a second per session before
# it starts issuing flood waits; one call every two seconds stays well clear.
_DEFAULT_RATE_PER_SEC = 0.5


class TelegramSource(BaseSource):
    name = "telegram"
    source = "telegram"
    requires_package = "telethon"

    # --- preflight -------------------------------------------------------
    def preflight(self) -> None:
        missing = [
            env
            for env, value in (
                ("TELEGRAM_API_ID", self.settings.telegram_api_id),
                ("TELEGRAM_API_HASH", self.settings.telegram_api_hash),
                ("TELEGRAM_SESSION", self.settings.telegram_session),
            )
            if not value
        ]
        if missing:
            self.note("missing_credentials", ", ".join(missing))
            raise SourceUnavailable(
                f"telegram: missing_credentials ({', '.join(missing)} absent in .env); "
                "skipping this source."
            )
        super().preflight()

    # --- client ----------------------------------------------------------
    def _client(self) -> Any:
        try:
            from telethon import TelegramClient
            from telethon.sessions import StringSession
        except ImportError as exc:  # pragma: no cover - guarded by preflight
            raise SourceUnavailable(f"telethon is not installed: {exc}") from exc

        try:
            api_id = int(self.settings.telegram_api_id or "")
        except ValueError as exc:
            raise SourceUnavailable("telegram: TELEGRAM_API_ID must be an integer") from exc
        if not (self.settings.telegram_api_hash and self.settings.telegram_session):
            raise SourceUnavailable("telegram: credentials absent; skipping")

        return TelegramClient(
            StringSession(self.settings.telegram_session),
            api_id,
            self.settings.telegram_api_hash,
            # Short waits are slept out by Telethon itself; anything longer is
            # raised so the adapter can count it and move to the next channel.
            flood_sleep_threshold=60,
        )

    # --- fetch -----------------------------------------------------------
    def _channels(self) -> list[str]:
        config = sources_config().get(self.name, {})
        channels = self.options.get("channels") or config.get("channels", [])
        cleaned = [str(c).strip().lstrip("@") for c in channels if str(c).strip()]
        # Deduplicate but keep config order, so a run is reproducible.
        return list(dict.fromkeys(cleaned))

    def fetch(self) -> Iterator[dict]:
        config = sources_config().get(self.name, {})
        backfill = config.get("backfill", {})
        per_channel = int(
            self.options.get("messages_per_channel")
            or backfill.get("messages_per_channel", _DEFAULT_MESSAGES_PER_CHANNEL)
        )
        if self.limit is not None:
            per_channel = min(per_channel, self.limit)
        bucket = TokenBucket(
            float(backfill.get("requests_per_sec", _DEFAULT_RATE_PER_SEC)),
            burst=1,
        )
        force = bool(self.options.get("force"))

        channels = self._channels()
        if not channels:
            self.log.warning("telegram: no channels listed in configs/sources.yaml; nothing to do")
            self.note("no_channels_configured")
            return

        loop = asyncio.new_event_loop()
        client = self._client()
        try:
            loop.run_until_complete(client.connect())
            if not loop.run_until_complete(client.is_user_authorized()):
                self.note("session_not_authorized")
                raise SourceUnavailable(
                    "telegram: session_not_authorized (TELEGRAM_SESSION is expired or revoked)"
                )
            for channel in channels:
                min_id = 0 if force else int(self.checkpoint.get(f"last_id.{channel}", 0) or 0)
                bucket.acquire()
                messages = loop.run_until_complete(
                    self._read_channel(client, channel, per_channel, min_id)
                )
                archive: list[dict] = []
                for message in messages:
                    archive.append(message)
                    yield message
                if archive:
                    # Only reached when every message was consumed: a run cut
                    # short by --limit leaves the cursor where it was, so the
                    # unread older messages are not skipped next time.
                    self.checkpoint.set(f"last_id.{channel}", str(max(m["id"] for m in archive)))
                    path = self.save_raw_payload(channel, archive)
                    self.record_manifest(
                        channel,
                        path=path,
                        url=f"https://t.me/s/{channel}",
                        rows=len(archive),
                    )
        finally:
            try:
                loop.run_until_complete(client.disconnect())
            finally:
                loop.close()

    async def _read_channel(
        self, client: Any, channel: str, limit: int, min_id: int
    ) -> list[dict]:
        """Resolve one channel and flatten its newest ``limit`` messages.

        Failures are per channel: one renamed or private channel must not abandon
        the others, but each is counted under its own reason code.
        """
        try:
            from telethon import errors
        except ImportError:  # pragma: no cover - guarded by preflight
            errors = None

        def classify(exc: Exception) -> str | None:
            if errors is None:
                return None
            if isinstance(exc, errors.FloodWaitError):
                return "flood_wait"
            if isinstance(exc, (errors.ChannelPrivateError, errors.ChannelInvalidError)):
                return "channel_private"
            if isinstance(exc, (errors.UsernameNotOccupiedError, errors.UsernameInvalidError)):
                return "channel_not_found"
            return None

        try:
            entity = await client.get_entity(channel)
            channel_info = _flatten_channel(entity, fallback_username=channel)
            out: list[dict] = []
            async for message in client.iter_messages(entity, limit=limit, min_id=min_id):
                flat = _flatten_message(message, channel_info)
                if flat is None:
                    # A service message (join, pin, title change) has no body to keep.
                    self.note("service_message", f"{channel}:{getattr(message, 'id', '?')}")
                    continue
                out.append(flat)
            return out
        except Exception as exc:
            code = classify(exc)
            if code is None:
                raise
            self.note(code, channel)
            self.log.warning("telegram: %s skipped (%s: %s)", channel, code, exc)
            return []

    # --- map -------------------------------------------------------------
    def to_record(self, raw: dict) -> Record | None:
        message_id = raw.get("id")
        channel = raw.get("channel") or {}
        channel_id = str(channel.get("id") or "").strip()
        if message_id is None or not channel_id:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None

        timestamp = _parse_dt(raw.get("date"))
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, f"{channel_id}:{message_id}")
            return None

        fields = build_text_fields(raw.get("text") or "")
        if not fields["text"]:
            self.drop(DropReason.EMPTY_TEXT, f"{channel_id}:{message_id}")
            return None

        username = channel.get("username")
        # Message ids restart at 1 in every channel, so the channel id is part
        # of the native id or two channels would collide in the corpus.
        native_id = f"{channel_id}:{message_id}"
        reply_to = raw.get("reply_to_id")
        permalink = (
            PERMALINK.format(username=username, message_id=message_id) if username else None
        )

        return Record(
            native_id=native_id,
            source="telegram",
            source_detail=username or channel_id,
            content_type="post",
            author_id=make_id("telegram", channel_id),
            author_handle=username,
            timestamp=timestamp,
            parent_id=make_id("telegram", f"{channel_id}:{reply_to}") if reply_to else None,
            # A channel post is its own thread; comments live in a linked
            # discussion group that this adapter does not read.
            conversation_id=None if reply_to else make_id("telegram", native_id),
            engagement=EngagementMetrics(
                # Telegram has reactions rather than likes: the total across all
                # emoji, None when the channel does not show reactions.
                likes=_as_int(raw.get("reactions_total")),
                shares=_as_int(raw.get("forwards")),
                replies=_as_int(raw.get("replies")),
                views=_as_int(raw.get("views")),
            ),
            media_urls=[],
            raw={
                "message_id": message_id,
                "channel_id": channel_id,
                "channel_username": username,
                "channel_title": channel.get("title"),
                "permalink": permalink,
                "edit_date": raw.get("edit_date"),
                "media_type": raw.get("media_type"),
                "grouped_id": raw.get("grouped_id"),
                "post_author": raw.get("post_author"),
                "forwarded_from": raw.get("forwarded_from"),
                "reactions": raw.get("reactions"),
                "reply_to_id": reply_to,
            },
            **fields,
        )

    def to_author(self, raw: dict, record: Record) -> Author | None:
        channel = raw.get("channel") or {}
        if not channel:
            return None
        return Author(
            author_id=record.author_id,
            source="telegram",
            handle=channel.get("username"),
            created_at=_parse_dt(channel.get("date")),
            followers=_as_int(channel.get("participants_count")),
            following=None,
            post_count=1,
            first_seen=record.timestamp,
            last_seen=record.timestamp,
            raw={k: v for k, v in channel.items() if v is not None},
        )


# --- flattening (Telethon objects -> plain JSON-able dicts) -----------------


def _flatten_channel(entity: Any, fallback_username: str | None = None) -> dict:
    return {
        "id": getattr(entity, "id", None),
        "username": getattr(entity, "username", None) or fallback_username,
        "title": getattr(entity, "title", None),
        "date": _iso(getattr(entity, "date", None)),
        # Only present when the entity came from a full-channel lookup; None
        # otherwise, never 0.
        "participants_count": getattr(entity, "participants_count", None),
    }


def _flatten_message(message: Any, channel: dict) -> dict | None:
    """One Telethon ``Message`` -> a plain dict. ``None`` for a service message.

    Absent counters stay ``None``. Telegram omits ``views`` on some posts and
    ``replies`` on channels without a discussion group, and a measured zero is
    only ever a counter the platform actually sent.
    """
    if getattr(message, "action", None) is not None:
        return None

    replies_obj = getattr(message, "replies", None)
    reactions_obj = getattr(message, "reactions", None)
    reactions: list[dict] | None = None
    reactions_total: int | None = None
    if reactions_obj is not None:
        reactions = []
        for result in getattr(reactions_obj, "results", None) or []:
            emoji = getattr(result, "reaction", None)
            reactions.append(
                {
                    "reaction": getattr(emoji, "emoticon", None) or type(emoji).__name__,
                    "count": getattr(result, "count", None),
                }
            )
        reactions_total = sum(r["count"] or 0 for r in reactions)

    fwd = getattr(message, "fwd_from", None)
    forwarded_from = None
    if fwd is not None:
        from_id = getattr(fwd, "from_id", None)
        forwarded_from = {
            "channel_id": getattr(from_id, "channel_id", None),
            "from_name": getattr(fwd, "from_name", None),
            "channel_post": getattr(fwd, "channel_post", None),
            "date": _iso(getattr(fwd, "date", None)),
        }

    reply_header = getattr(message, "reply_to", None)
    media = getattr(message, "media", None)
    return {
        "id": message.id,
        "date": _iso(getattr(message, "date", None)),
        "edit_date": _iso(getattr(message, "edit_date", None)),
        # ``message`` is the body, or the caption when the post carries media.
        "text": getattr(message, "message", None) or "",
        "views": getattr(message, "views", None),
        "forwards": getattr(message, "forwards", None),
        "replies": getattr(replies_obj, "replies", None) if replies_obj is not None else None,
        "reactions_total": reactions_total,
        "reactions": reactions,
        "forwarded_from": forwarded_from,
        "reply_to_id": getattr(reply_header, "reply_to_msg_id", None) if reply_header else None,
        "media_type": type(media).__name__ if media is not None else None,
        "grouped_id": str(g) if (g := getattr(message, "grouped_id", None)) is not None else None,
        "post_author": getattr(message, "post_author", None),
        "channel": channel,
    }


# --- helpers ---------------------------------------------------------------


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
