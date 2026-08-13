"""Reddit via static Kaggle dumps (and, optionally, Academic Torrents files).

Secondary source. ConvoKit stays primary because it ships threading; these
dumps usually do not.

**Read this before using Kaggle-sourced Reddit data for anything structural:**
these are flat CSV/JSON exports with inconsistent columns and, in almost every
case, no reply pointers. ``parent_id`` and ``conversation_id`` are therefore set
to ``None`` honestly rather than fabricated, which means this data is usable for
text/narrative work and *not* usable for the Phase 2 coordination graph. The
README says the same thing in the same words.

Each dataset gets an explicit column map in ``configs/sources.yaml``. One
heuristic parser across many datasets half-works on all of them, which is worse
than not working: it fails silently and the damage shows up three weeks later.

Academic Torrents path: torrent downloads are deliberately not automated here.
Point ``--path`` at an already-downloaded directory; the loader streams
line-delimited JSON, including zstd-compressed dumps, because ``json.load()`` on
a 40GB file is not a strategy.
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingest.config import sources_config
from ingest.normalize import build_text_fields, is_deleted_text
from ingest.schema import DELETED_AUTHOR, DropReason, EngagementMetrics, Record, make_id
from ingest.sources.base import BaseSource, SourceUnavailable

#: Fields a dataset's column_map may bind. Anything else is ignored.
MAPPABLE = (
    "native_id",
    "text",
    "body",
    "author",
    "timestamp",
    "score",
    "num_comments",
    "url",
    "subreddit",
    "parent_id",
    "conversation_id",
    "title",
)

CSV_FIELD_LIMIT = 10 * 1024 * 1024


class RedditKaggleSource(BaseSource):
    name = "reddit_kaggle"
    source = "reddit"
    requires_package = None  # the kaggle CLI is only needed to *download*

    def preflight(self) -> None:
        """Kaggle credentials are needed to *download*, not to read a local dump.

        The Academic Torrents path hands us an already-downloaded directory via
        ``--path``. Gating that behind a Kaggle key would make the documented
        bulk-import workflow impossible on a machine with no Kaggle account.
        """
        if self.options.get("path"):
            return
        super().preflight()

    # --- fetch -----------------------------------------------------------
    def fetch(self) -> Iterator[dict]:
        config = sources_config().get(self.name, {})
        datasets: list[dict] = self.options.get("datasets") or config.get("datasets") or []
        max_rows = int(self.options.get("max_rows") or config.get("max_rows_per_dataset", 100_000))

        if self.options.get("path"):
            # Academic Torrents / manual path: an already-downloaded directory.
            yield from self._read_directory(
                Path(self.options["path"]).expanduser(),
                spec=self.options.get("spec") or {"slug": "local", "column_map": {}},
                max_rows=max_rows,
            )
            return

        if not datasets:
            self.log.warning(
                "no kaggle datasets configured. add slugs with an explicit column_map under "
                "reddit_kaggle.datasets in configs/sources.yaml; see the commented example there."
            )
            return

        for spec in datasets:
            slug = spec.get("slug")
            if not slug:
                self.log.warning("dataset entry without a slug: %s", spec)
                continue
            if self.checkpoint.is_done(slug) and not self.options.get("force"):
                self.log.info("skipping %s: already ingested", slug)
                continue
            target = self._download(slug)
            if target is None:
                continue
            yield from self._read_directory(target, spec, max_rows)
            self.checkpoint.mark_done(slug)

    def _download(self, slug: str) -> Path | None:
        """``kaggle datasets download -d <slug> -p <dir> --unzip``.

        Authentication is the kaggle CLI's own concern (``~/.kaggle/kaggle.json``
        or ``KAGGLE_USERNAME``/``KAGGLE_KEY``); we do not re-implement it.
        """
        target = self.settings.raw_dir_for(self.name) / slug.replace("/", "__")
        if target.exists() and any(target.iterdir()):
            self.log.info("reusing existing download at %s", target)
            return target
        target.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(target),
            "--unzip",
        ]
        self.log.info("downloading kaggle dataset %s", slug)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.log.warning("kaggle download failed for %s: %s", slug, exc)
            return None
        if result.returncode != 0:
            self.log.warning(
                "kaggle download failed for %s (exit %d): %s",
                slug,
                result.returncode,
                (result.stderr or result.stdout or "").strip()[:400],
            )
            return None

        digest_rows = sum(1 for _ in target.rglob("*") if _.is_file())
        self.record_manifest(
            slug,
            path=target,
            url=f"https://www.kaggle.com/datasets/{slug}",
            rows=None,
            extra={"kind": "kaggle-dataset", "files": digest_rows},
        )
        return target

    def _read_directory(self, directory: Path, spec: dict, max_rows: int) -> Iterator[dict]:
        if not directory.exists():
            raise SourceUnavailable(f"{directory} does not exist")
        patterns = spec.get("files") or ["*.csv", "*.json", "*.jsonl", "*.ndjson", "*.zst"]
        files = sorted({p for pattern in patterns for p in directory.rglob(pattern) if p.is_file()})
        if not files:
            self.log.warning("no matching files under %s (patterns=%s)", directory, patterns)
            return

        emitted = 0
        for path in files:
            self.log.info("reading %s", path)
            for row in self._read_file(path):
                if emitted >= max_rows:
                    self.log.info("hit max_rows_per_dataset=%d", max_rows)
                    return
                emitted += 1
                yield {"_spec": spec, "_file": path.name, **row}
                if emitted % 10_000 == 0:
                    self.checkpoint.set(f"{spec.get('slug')}.emitted", emitted)
        self.checkpoint.set(f"{spec.get('slug')}.emitted", emitted)

    def _read_file(self, path: Path) -> Iterator[dict]:
        suffixes = {s.lower() for s in path.suffixes}
        try:
            if ".zst" in suffixes:
                yield from self._read_zstd_jsonl(path)
            elif suffixes & {".jsonl", ".ndjson"}:
                yield from self._read_jsonl(path)
            elif ".json" in suffixes:
                yield from self._read_json(path)
            else:
                yield from self._read_csv(path)
        except OSError as exc:
            self.log.warning("could not read %s: %s", path, exc)

    # --- readers ---------------------------------------------------------
    def _read_csv(self, path: Path) -> Iterator[dict]:
        # Reddit bodies routinely exceed the default 128KB csv field limit.
        csv.field_size_limit(CSV_FIELD_LIMIT)
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                yield dict(row)

    def _read_jsonl(self, path: Path) -> Iterator[dict]:
        loads = _json_loads()
        with path.open("rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield loads(line)
                except Exception:
                    self.drop(DropReason.VALIDATION_ERROR, f"{path.name}: unparseable json line")

    def _read_json(self, path: Path) -> Iterator[dict]:
        loads = _json_loads()
        payload = loads(path.read_bytes())
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        yield from (row for row in rows if isinstance(row, dict))

    def _read_zstd_jsonl(self, path: Path) -> Iterator[dict]:
        """Stream a zstd-compressed ndjson dump.

        Academic Torrents Reddit dumps are tens of gigabytes compressed. They are
        streamed line by line; decompressing to memory is not an option.
        """
        try:
            import zstandard
        except ImportError as exc:
            raise SourceUnavailable(
                'zstandard is required for .zst dumps; install with `pip install -e ".[sources]"`'
            ) from exc
        loads = _json_loads()
        decompressor = zstandard.ZstdDecompressor(max_window_size=2**31)
        with path.open("rb") as fh, decompressor.stream_reader(fh) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield loads(line)
                except Exception:
                    self.drop(DropReason.VALIDATION_ERROR, f"{path.name}: unparseable json line")

    # --- map -------------------------------------------------------------
    def to_record(self, raw: dict) -> Record | None:
        spec = raw.get("_spec") or {}
        column_map = {k: v for k, v in (spec.get("column_map") or {}).items() if k in MAPPABLE}
        get = lambda field, default=None: _lookup(raw, column_map, field, default)  # noqa: E731

        native_id = str(get("native_id") or raw.get("id") or "").strip()
        if not native_id:
            self.drop(DropReason.MISSING_ID, str(raw)[:120])
            return None

        title = str(get("title") or "").strip()
        body = str(get("text") or get("body") or "").strip()
        if is_deleted_text(body) and not title:
            self.drop(DropReason.DELETED_TEXT, native_id)
            return None
        if is_deleted_text(body):
            body = ""
        text = "\n\n".join(part for part in (title, body) if part)
        if not text:
            self.drop(DropReason.EMPTY_TEXT, native_id)
            return None

        timestamp = _parse_timestamp(get("timestamp"), spec.get("timestamp_format"))
        if timestamp is None:
            self.drop(DropReason.MISSING_TIMESTAMP, native_id)
            return None

        author = str(get("author") or "").strip()
        if not author or is_deleted_text(author):
            self.note("author_deleted", native_id)
            author, handle = DELETED_AUTHOR, None
        else:
            handle = author

        url = get("url")
        fields = build_text_fields(text, structured_urls=[url] if url else None)

        # Threading: only claim it when the dataset actually carries it. A
        # fabricated parent chain would silently corrupt every coordination
        # metric computed downstream.
        parent = _clean_id(get("parent_id"))
        conversation = _clean_id(get("conversation_id"))
        if parent is None and conversation is None:
            self.note("no_threading_in_dataset", native_id)

        return Record(
            native_id=native_id,
            source="reddit",
            source_detail=str(get("subreddit") or spec.get("slug") or "unknown"),
            content_type=spec.get("content_type", "post"),
            author_id=make_id("reddit", author),
            author_handle=handle,
            timestamp=timestamp,
            parent_id=make_id("reddit", parent) if parent else None,
            conversation_id=make_id("reddit", conversation) if conversation else None,
            engagement=EngagementMetrics(
                likes=_as_int(get("score")),
                shares=None,
                replies=_as_int(get("num_comments")),
                views=None,
            ),
            raw={
                "dataset": spec.get("slug"),
                "file": raw.get("_file"),
                "columns": {k: v for k, v in raw.items() if not k.startswith("_")},
                "threading_available": bool(parent or conversation),
            },
            **fields,
        )


# --- module helpers -------------------------------------------------------


def _json_loads():
    try:
        import orjson

        return orjson.loads
    except ImportError:  # pragma: no cover - orjson is a core dep
        import json

        return json.loads


def _lookup(raw: dict, column_map: dict, field: str, default: Any = None) -> Any:
    """Resolve a schema field through the dataset's explicit column map."""
    column = column_map.get(field)
    if column and column in raw:
        return raw[column]
    if column is None and field in raw:
        return raw[field]
    return default


def _clean_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", ""}:
        return None
    return text


def _parse_timestamp(value: Any, fmt: str | None = None) -> datetime | None:
    """Epoch seconds, epoch millis, ISO8601, or an explicit strptime format."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()

    if fmt and fmt not in {"epoch", "iso"}:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    if fmt != "iso":
        try:
            seconds = float(text)
        except ValueError:
            seconds = None
        if seconds is not None:
            if seconds <= 0:
                return None
            if seconds > 1e12:
                seconds /= 1000.0
            try:
                return datetime.fromtimestamp(seconds, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
