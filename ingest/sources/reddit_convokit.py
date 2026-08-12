"""Reddit via ConvoKit (Cornell) -- the primary Reddit source.

The Reddit API is not available to this project and Pushshift is shut down, so
Reddit data comes from static pre-collected corpora only. ConvoKit is the best
of those options because it ships *conversation structure*: Speakers,
Utterances and Conversations map almost 1:1 onto our schema, which means
threading and stable author identity come free. A flat CSV dump gives neither,
and without threading the Phase 2 coordination graph has no edges.

What this source gives you: threaded Reddit conversations, stable pseudonymous
speaker ids, per-utterance score, subreddit, permalink.
What it costs: nothing (no key, no rate limit) beyond disk and a slow first
download.
What it cannot tell you: anything about *current* Reddit. These corpora are
historical snapshots with a fixed end date, and deleted content is already
tombstoned at collection time.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingest.config import sources_config
from ingest.normalize import build_text_fields, is_deleted_text
from ingest.schema import DELETED_AUTHOR, Author, DropReason, EngagementMetrics, Record, make_id
from ingest.sources.base import BaseSource, SourceUnavailable

#: ConvoKit's own download cache. We symlink it under data/raw/ so the manifest
#: can checksum what was actually used.
CONVOKIT_HOME = Path(os.environ.get("CONVOKIT_DATA_DIR", Path.home() / ".convokit" / "downloads"))

DOWNLOAD_URL = "https://zissou.infosci.cornell.edu/convokit/datasets/{name}/"


class ConvoKitSource(BaseSource):
    name = "reddit_convokit"
    source = "reddit"
    requires_package = "convokit"

    # --- fetch -----------------------------------------------------------
    def fetch(self) -> Iterator[dict]:
        config = sources_config().get(self.name, {})
        corpora: list[str] = (
            self.options.get("corpora") or config.get("corpora") or ["reddit-corpus-small"]
        )
        cap = self.options.get("max_utterances") or config.get("max_utterances_per_corpus")

        for corpus_name in corpora:
            if self.checkpoint.is_done(corpus_name) and not self.options.get("force"):
                self.log.info("skipping %s: already completed (use --force to redo)", corpus_name)
                continue
            yield from self._fetch_corpus(corpus_name, cap)
            self.checkpoint.mark_done(corpus_name)

    def _fetch_corpus(self, corpus_name: str, cap: int | None) -> Iterator[dict]:
        corpus, path = self._load_corpus(corpus_name)
        self.log.info("loaded %s from %s", corpus_name, path)

        emitted = 0
        for utterance in corpus.iter_utterances():
            if cap is not None and emitted >= cap:
                self.log.info("hit max_utterances_per_corpus=%d for %s", cap, corpus_name)
                break
            emitted += 1
            yield self._utterance_to_dict(utterance, corpus, corpus_name)
            if emitted % 5000 == 0:
                self.checkpoint.set(f"{corpus_name}.emitted", emitted)

        self.checkpoint.set(f"{corpus_name}.emitted", emitted)
        self.record_manifest(
            corpus_name,
            path=path,
            url=DOWNLOAD_URL.format(name=corpus_name),
            rows=emitted,
            extra={"kind": "convokit-corpus"},
        )

    def _load_corpus(self, corpus_name: str) -> tuple[Any, Path]:
        """Download (or reuse) a corpus and symlink it under ``data/raw/``."""
        try:
            from convokit import Corpus, download
        except ImportError as exc:  # pragma: no cover - guarded by preflight
            raise SourceUnavailable(f"convokit is not installed: {exc}") from exc

        local = self.options.get("path")
        if local:
            path = Path(local).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"--path {path} does not exist")
        else:
            self.log.info(
                "downloading convokit corpus %r (first run can take a while)", corpus_name
            )
            path = Path(download(corpus_name)).resolve()

        self._link_into_raw(corpus_name, path)
        return Corpus(filename=str(path)), path

    def _link_into_raw(self, corpus_name: str, path: Path) -> None:
        """Symlink the ConvoKit cache into ``data/raw/`` for the manifest.

        A symlink rather than a copy: these corpora run to gigabytes and
        duplicating them buys nothing. The manifest checksums the real files.
        """
        link = self.settings.raw_dir_for(self.name) / corpus_name
        try:
            if link.is_symlink():
                if link.resolve() == path:
                    return
                link.unlink()
            elif link.exists():
                return  # a real directory was placed here on purpose; leave it
            link.symlink_to(path, target_is_directory=True)
        except OSError as exc:  # pragma: no cover - platform/permissions
            self.log.warning("could not symlink %s -> %s: %s", link, path, exc)

    @staticmethod
    def _utterance_to_dict(utterance: Any, corpus: Any, corpus_name: str) -> dict:
        """Flatten a ConvoKit Utterance to a plain dict.

        Deliberate: ``to_record`` then operates on JSON-serializable dicts, so
        the mapping is testable against recorded fixtures with convokit absent.
        """
        meta = dict(utterance.meta) if utterance.meta else {}
        speaker = utterance.speaker
        conversation_meta: dict[str, Any] = {}
        try:
            conversation = corpus.get_conversation(utterance.conversation_id)
            if conversation is not None and conversation.meta:
                conversation_meta = dict(conversation.meta)
        except Exception:  # pragma: no cover - malformed corpora
            conversation_meta = {}
        return {
            "id": utterance.id,
            "speaker_id": getattr(speaker, "id", None),
            "speaker_meta": dict(speaker.meta) if speaker is not None and speaker.meta else {},
            "conversation_id": utterance.conversation_id,
            "reply_to": utterance.reply_to,
            "timestamp": utterance.timestamp,
            "text": utterance.text,
            "meta": meta,
            "conversation_meta": conversation_meta,
            "corpus": corpus_name,
        }

    # --- map -------------------------------------------------------------
    def to_record(self, raw: dict) -> Record | None:
        native_id = str(raw.get("id") or "").strip()
        if not native_id:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None

        text = raw.get("text") or ""
        meta = raw.get("meta") or {}
        conversation_meta = raw.get("conversation_meta") or {}
        reply_to = raw.get("reply_to")
        is_post = reply_to is None

        # Deleted/removed bodies carry no information: drop them, but count them.
        if is_deleted_text(text):
            self.drop(DropReason.DELETED_TEXT, native_id)
            return None

        # Link posts have an empty selftext and carry the claim in the title.
        title = conversation_meta.get("title") if is_post else None
        include_title = sources_config().get(self.name, {}).get("include_title_in_post_text", True)
        if is_post and include_title and title:
            text = f"{title}\n\n{text}".strip()

        if not text.strip():
            self.drop(DropReason.EMPTY_TEXT, native_id)
            return None

        timestamp = self._to_datetime(raw.get("timestamp"))
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, native_id)
            return None

        # Author deleted but content kept: keep the record, flag the author.
        # Losing the text would lose evidence; pretending the author is usable
        # would corrupt the coordination graph. The sentinel says which.
        speaker_id = str(raw.get("speaker_id") or "").strip()
        if not speaker_id or is_deleted_text(speaker_id):
            self.note("author_deleted", native_id)  # kept as evidence, flagged
            speaker_id = DELETED_AUTHOR
            handle = None
        else:
            handle = speaker_id

        fields = build_text_fields(
            text, structured_urls=meta.get("url") or conversation_meta.get("url")
        )

        subreddit = meta.get("subreddit") or conversation_meta.get("subreddit") or raw.get("corpus")

        return Record(
            native_id=native_id,
            source="reddit",
            source_detail=str(subreddit or "unknown"),
            content_type="post" if is_post else "comment",
            author_id=make_id("reddit", speaker_id),
            author_handle=handle,
            timestamp=timestamp,
            parent_id=make_id("reddit", reply_to) if reply_to else None,
            conversation_id=(
                make_id("reddit", raw["conversation_id"]) if raw.get("conversation_id") else None
            ),
            engagement=EngagementMetrics(
                likes=_as_int(meta.get("score")),
                # Reddit exposes neither shares nor views in these corpora, and
                # replies are only countable at conversation level. Null, not 0.
                shares=None,
                replies=_as_int(conversation_meta.get("num_comments")) if is_post else None,
                views=None,
            ),
            media_urls=[],
            raw={
                "corpus": raw.get("corpus"),
                "meta": meta,
                "conversation_meta": conversation_meta,
                "speaker_meta": raw.get("speaker_meta") or {},
                "title": title,
            },
            **fields,
        )

    def to_author(self, raw: dict, record: Record) -> Author | None:
        if record.author_is_deleted:
            return None
        speaker_meta = raw.get("speaker_meta") or {}
        return Author(
            author_id=record.author_id,
            source="reddit",
            handle=record.author_handle,
            created_at=None,  # ConvoKit corpora do not carry account creation dates
            followers=None,  # Reddit has no follower count in these corpora
            following=None,
            post_count=1,
            first_seen=record.timestamp,
            last_seen=record.timestamp,
            raw={k: v for k, v in speaker_meta.items() if k in {"num_posts", "num_comments"}},
        )

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _to_datetime(value: Any) -> datetime | None:
        """ConvoKit timestamps are Unix ints. Reject anything we cannot trust."""
        if value is None or value == "":
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if seconds <= 0:
            return None
        if seconds > 1e12:  # milliseconds sneaking in from a re-exported corpus
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
