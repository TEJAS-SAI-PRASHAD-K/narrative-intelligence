"""The adapter contract.

An adapter implements exactly two things:

* ``fetch()`` -- yield raw payload dicts, handling pagination, rate limits and
  checkpointing as it goes.
* ``to_record(raw)`` -- map one raw dict to one validated ``Record``, or return
  ``None`` after calling :meth:`BaseSource.drop` with a reason code.

Everything else -- buffering, dedupe, parquet writes, drop accounting, the
manifest entry -- lives here so the six adapters cannot drift apart.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ingest.checkpoint import Checkpoint, QuotaExhausted
from ingest.config import Settings, get_settings
from ingest.schema import Author, DropReason, Record
from ingest.store import ParquetStore


@dataclass
class RunResult:
    """What one adapter run did. Printed by the CLI, asserted on by tests."""

    source: str
    fetched: int = 0
    written: int = 0
    duplicates: int = 0
    dropped: Counter = field(default_factory=Counter)
    authors: int = 0
    skipped_reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())

    def as_row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fetched": self.fetched,
            "written": self.written,
            "duplicates": self.duplicates,
            "dropped": self.total_dropped,
            "status": ("error" if self.error else ("skipped" if self.skipped_reason else "ok")),
            "note": self.error or self.skipped_reason or "",
        }


class SourceUnavailable(RuntimeError):
    """The adapter cannot run: missing credentials or missing optional package.

    Not an error condition. ``fetch-all`` catches this, logs a clear warning and
    moves on -- graceful degradation is an acceptance criterion, not a nicety.
    """


class BaseSource(ABC):
    #: adapter name, e.g. "reddit_convokit"; used for checkpoints and the registry
    name: str = "base"
    #: the schema ``source`` value this adapter emits, e.g. "reddit"
    source: str = "reddit"
    #: pip extra that provides this adapter's client library, if any
    requires_package: str | None = None

    def __init__(
        self,
        settings: Settings | None = None,
        store: ParquetStore | None = None,
        limit: int | None = None,
        **options: Any,
    ):
        self.settings = settings or get_settings()
        self.store = store or ParquetStore(self.settings)
        self.limit = limit
        self.options = options
        self.settings.ensure_dirs()
        self.checkpoint = Checkpoint(self.name, self.settings.checkpoint_dir)
        self.log = logging.getLogger(f"ingest.{self.name}")
        self.dropped: Counter = Counter()
        self._authors: dict[str, Author] = {}

    # --- to implement ----------------------------------------------------
    @abstractmethod
    def fetch(self) -> Iterator[dict]:
        """Yield raw payload dicts. Paginate, rate-limit and checkpoint in here."""

    @abstractmethod
    def to_record(self, raw: dict) -> Record | None:
        """Map one raw payload to a ``Record``, or drop it with a reason code."""

    # --- optional hooks --------------------------------------------------
    def to_author(self, raw: dict, record: Record) -> Author | None:
        """Optional per-record author roll-up. Override where the API gives one."""
        return None

    def preflight(self) -> None:
        """Raise :class:`SourceUnavailable` when the adapter cannot run.

        Default: check the credential gate and the optional import.
        """
        if not self.settings.has_credentials(self.name):
            raise SourceUnavailable(
                f"{self.name}: credentials absent; set the relevant key in .env "
                "(see .env.example). Skipping this source."
            )
        if self.requires_package:
            import importlib.util

            if importlib.util.find_spec(self.requires_package.split(".")[0]) is None:
                raise SourceUnavailable(
                    f"{self.name}: python package {self.requires_package!r} is not installed. "
                    'Install it with `pip install -e ".[sources]"`. Skipping this source.'
                )

    # --- drop accounting -------------------------------------------------
    def drop(self, reason: DropReason | str, detail: str = "") -> None:
        """Record a dropped item. Every drop is counted; none are silent."""
        code = reason.value if isinstance(reason, DropReason) else str(reason)
        self.dropped[code] += 1
        # Log the first few of each kind in full, then only the running count:
        # a corpus with 300k deleted comments should not produce 300k log lines.
        if self.dropped[code] <= 3:
            self.log.info("dropped (%s): %s", code, detail or "-")
        elif self.dropped[code] % 5000 == 0:
            self.log.info("dropped (%s): %d so far", code, self.dropped[code])

    def note_author(self, author: Author | None) -> None:
        """Merge an author roll-up into this run's accumulator."""
        if author is None:
            return
        existing = self._authors.get(author.author_id)
        if existing is None:
            self._authors[author.author_id] = author
            return
        existing.post_count += author.post_count
        existing.first_seen = min(existing.first_seen, author.first_seen)
        existing.last_seen = max(existing.last_seen, author.last_seen)
        for attr in ("handle", "created_at", "followers", "following"):
            value = getattr(author, attr)
            if value is not None:
                setattr(existing, attr, value)
        if author.raw:
            existing.raw = author.raw

    # --- the run loop ----------------------------------------------------
    def run(self, flush_every: int = 1000) -> RunResult:
        """Fetch -> map -> validate -> write, in bounded batches.

        Batched flushing is what makes a kill mid-run survivable: work already
        written stays written, and the id-level dedupe in the store means the
        resumed run cannot double-count it.
        """
        result = RunResult(source=self.name)
        try:
            self.preflight()
        except SourceUnavailable as exc:
            self.log.warning("SKIP %s", exc)
            result.skipped_reason = str(exc)
            return result

        buffer: list[Record] = []

        def flush() -> None:
            if not buffer:
                return
            written = self.store.write_records(buffer)
            result.written += written["written"]
            result.duplicates += written["duplicates"]
            buffer.clear()

        try:
            for raw in self.fetch():
                result.fetched += 1
                try:
                    record = self.to_record(raw)
                except ValidationError as exc:
                    self.drop(DropReason.VALIDATION_ERROR, _first_error(exc))
                    continue
                except Exception as exc:  # a bad payload must not kill the run
                    self.drop(DropReason.VALIDATION_ERROR, f"{type(exc).__name__}: {exc}")
                    continue
                if record is None:
                    continue
                buffer.append(record)
                self.note_author(self.to_author(raw, record))
                if len(buffer) >= flush_every:
                    flush()
                if self.limit is not None and result.written + len(buffer) >= self.limit:
                    self.log.info("reached --limit %d; stopping cleanly", self.limit)
                    break
        except QuotaExhausted as exc:
            # Expected, not exceptional: stop early with everything kept.
            self.log.warning("%s", exc)
            result.skipped_reason = str(exc)
        except KeyboardInterrupt:
            self.log.warning("interrupted; flushing %d buffered records", len(buffer))
            flush()
            raise
        except Exception as exc:
            self.log.exception("%s failed: %s", self.name, exc)
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            flush()
            self.checkpoint.save()

        if self._authors:
            result.authors = self.store.write_authors(self._authors.values(), self.source)

        result.dropped = self.dropped.copy()
        self.log.info(
            "%s: fetched=%d written=%d duplicates=%d dropped=%d %s",
            self.name,
            result.fetched,
            result.written,
            result.duplicates,
            result.total_dropped,
            dict(result.dropped) or "",
        )
        return result

    # --- helpers for adapters -------------------------------------------
    def record_manifest(self, key: str, **kwargs: Any) -> None:
        self.store.manifest.record_artifact(f"{self.name}:{key}", **kwargs)

    def http(self, rate_per_sec: float | None = None):
        from ingest.ratelimit import RateLimitedSession

        return RateLimitedSession(
            user_agent=self.settings.user_agent,
            rate_per_sec=rate_per_sec or self.settings.http_rate_limit_per_sec,
        )


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:  # pragma: no cover
        return str(exc)
    first = errors[0]
    return f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
