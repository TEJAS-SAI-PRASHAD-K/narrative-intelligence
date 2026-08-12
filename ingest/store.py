"""Parquet corpus: partitioned writes, id-level dedupe, and the artifact manifest.

Layout::

    data/normalized/source=<source>/date=<YYYY-MM-DD>/part-<uuid>.parquet
    data/manifest.json

Parquet rather than CSV because the schema has nested and typed fields (an
engagement struct, five list columns, tz-aware timestamps) and CSV loses all of
them. ``raw`` is stored as a JSON string, not as a struct: raw payloads differ
per source and per API version, and letting them into the Arrow schema would
make every source's partition mutually unreadable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ingest.config import Settings, get_settings
from ingest.schema import Record, utcnow

log = logging.getLogger(__name__)

ENGAGEMENT_TYPE = pa.struct(
    [
        pa.field("likes", pa.int64()),
        pa.field("shares", pa.int64()),
        pa.field("replies", pa.int64()),
        pa.field("views", pa.int64()),
    ]
)

#: The physical schema. Explicit, not inferred: inference from a batch that
#: happens to contain only nulls silently produces a null-typed column, and the
#: next batch then fails to merge.
ARROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("native_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_detail", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("lang", pa.string()),
        pa.field("author_id", pa.string(), nullable=False),
        pa.field("author_handle", pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("parent_id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("engagement", ENGAGEMENT_TYPE),
        pa.field("urls", pa.list_(pa.string())),
        pa.field("domains", pa.list_(pa.string())),
        pa.field("media_urls", pa.list_(pa.string())),
        pa.field("hashtags", pa.list_(pa.string())),
        pa.field("mentions", pa.list_(pa.string())),
        pa.field("simhash", pa.uint64()),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("raw", pa.string()),  # json blob, deliberately opaque
    ]
)

AUTHOR_SCHEMA = pa.schema(
    [
        pa.field("author_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("handle", pa.string()),
        pa.field("created_at", pa.timestamp("us", tz="UTC")),
        pa.field("followers", pa.int64()),
        pa.field("following", pa.int64()),
        pa.field("post_count", pa.int64()),
        pa.field("first_seen", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("last_seen", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("raw", pa.string()),
    ]
)


def _dumps(obj: Any) -> str:
    """JSON-serialize a raw payload, tolerating datetimes and odd objects."""
    try:
        import orjson

        return orjson.dumps(obj, default=str).decode("utf-8")
    except ImportError:  # pragma: no cover - orjson is a core dep
        return json.dumps(obj, default=str, ensure_ascii=False)


def record_to_row(record: Record) -> dict[str, Any]:
    row = record.model_dump(mode="python")
    row["raw"] = _dumps(row.get("raw") or {})
    return row


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int]:
    """Checksum a directory: sha256 over (relative path, file digest) pairs.

    ConvoKit and Kaggle both hand us directories, not single files, and the
    manifest promises one checksum per artifact.
    """
    digest = hashlib.sha256()
    total = 0
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(file).encode("ascii"))
        total += file.stat().st_size
    return digest.hexdigest(), total


class Manifest:
    """``data/manifest.json``: every raw artifact with url, sha256, bytes, rows.

    This is half of the reproducibility story (the other half is ``make data``).
    It is a build product, so it lives under the gitignored ``data/`` tree; its
    contents are what you paste into a paper appendix.
    """

    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                self._entries = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("manifest at %s is corrupt; starting a fresh one", path)

    def record_artifact(
        self,
        key: str,
        *,
        path: Path | None = None,
        url: str | None = None,
        rows: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "url": url,
            "path": str(path) if path else None,
            "sha256": None,
            "bytes": None,
            "rows": rows,
            "fetched_at": utcnow().isoformat(),
        }
        if path is not None and path.exists():
            if path.is_dir():
                entry["sha256"], entry["bytes"] = sha256_tree(path)
                entry["checksum_kind"] = "sha256-tree"
            else:
                entry["sha256"] = sha256_file(path)
                entry["bytes"] = path.stat().st_size
                entry["checksum_kind"] = "sha256-file"
        if extra:
            entry.update(extra)
        self._entries[key] = entry
        self.save()
        return entry

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, json.dumps(self._entries, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so a kill mid-write cannot corrupt state."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ParquetStore:
    """Write and read the normalized corpus."""

    def __init__(self, settings: Settings | None = None, root: Path | None = None):
        self.settings = settings or get_settings()
        self.root = Path(root) if root else self.settings.normalized_dir
        self.manifest = Manifest(self.settings.manifest_path)

    # --- writing ---------------------------------------------------------
    def write_records(self, records: Iterable[Record], *, dedupe: bool = True) -> dict[str, int]:
        """Append records to their ``source=/date=`` partitions.

        Returns ``{"written": n, "duplicates": n}``. Dedupe is on ``id`` against
        both the incoming batch and what is already on disk, which is what makes
        "kill it mid-run and start again" safe.
        """
        buckets: dict[tuple[str, str], list[Record]] = defaultdict(list)
        batch_ids: set[str] = set()
        duplicates = 0
        for record in records:
            if dedupe and record.id in batch_ids:
                duplicates += 1
                continue
            batch_ids.add(record.id)
            buckets[(record.source, record.date_partition)].append(record)

        written = 0
        for (source, date), bucket in sorted(buckets.items()):
            partition = self.root / f"source={source}" / f"date={date}"
            if dedupe:
                existing = self.existing_ids(source=source, date=date)
                fresh = [r for r in bucket if r.id not in existing]
                duplicates += len(bucket) - len(fresh)
            else:
                fresh = bucket
            if not fresh:
                continue
            partition.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist([record_to_row(r) for r in fresh], schema=ARROW_SCHEMA)
            out = partition / f"part-{uuid.uuid4().hex[:12]}.parquet"
            pq.write_table(table, out, compression="zstd")
            written += table.num_rows
            log.debug("wrote %d rows -> %s", table.num_rows, out)

        if duplicates:
            log.info("skipped %d duplicate record ids", duplicates)
        return {"written": written, "duplicates": duplicates}

    def write_authors(self, authors: Iterable[Any], source: str) -> int:
        """Author roll-ups, one parquet file per source (rewritten, not appended).

        Roll-ups are aggregates: merging on rewrite is correct, appending is not.
        """
        rows = []
        for author in authors:
            row = author.model_dump(mode="python")
            row["raw"] = _dumps(row.get("raw") or {})
            rows.append(row)
        if not rows:
            return 0
        merged: dict[str, dict[str, Any]] = {}
        existing = self.read_authors(source)
        for row in existing + rows:
            prior = merged.get(row["author_id"])
            if prior is None:
                merged[row["author_id"]] = row
                continue
            prior["post_count"] = (prior.get("post_count") or 0) + (row.get("post_count") or 0)
            prior["first_seen"] = min(prior["first_seen"], row["first_seen"])
            prior["last_seen"] = max(prior["last_seen"], row["last_seen"])
            for key in ("handle", "created_at", "followers", "following", "raw"):
                if row.get(key) is not None:
                    prior[key] = row[key]
        out_dir = self.settings.data_dir / "authors" / f"source={source}"
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(list(merged.values()), schema=AUTHOR_SCHEMA)
        pq.write_table(table, out_dir / "authors.parquet", compression="zstd")
        return table.num_rows

    def read_authors(self, source: str) -> list[dict[str, Any]]:
        path = self.settings.data_dir / "authors" / f"source={source}" / "authors.parquet"
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()

    # --- reading ---------------------------------------------------------
    def dataset(self, source: str | None = None) -> ds.Dataset | None:
        root = self.root / f"source={source}" if source else self.root
        if not root.exists() or not any(root.rglob("*.parquet")):
            return None
        partitioning = "hive"
        try:
            return ds.dataset(root, format="parquet", partitioning=partitioning)
        except Exception as exc:  # pragma: no cover - malformed partition dirs
            log.warning("could not open dataset at %s: %s", root, exc)
            return None

    def existing_ids(self, source: str | None = None, date: str | None = None) -> set[str]:
        """Ids already on disk, scoped as narrowly as possible.

        Scanning one ``source=/date=`` partition keeps resume cheap; scanning the
        whole corpus on every batch would not.
        """
        root = self.root
        if source:
            root = root / f"source={source}"
        if source and date:
            root = root / f"date={date}"
        if not root.exists():
            return set()
        ids: set[str] = set()
        for file in root.rglob("*.parquet"):
            try:
                ids.update(pq.read_table(file, columns=["id"]).column("id").to_pylist())
            except Exception as exc:  # pragma: no cover - partially written file
                log.warning("unreadable parquet %s: %s", file, exc)
        return ids

    def read_all(self, source: str | None = None):
        """The corpus as a pandas DataFrame. Used by the CLI and the EDA notebook."""
        dataset = self.dataset(source)
        if dataset is None:
            import pandas as pd

            return pd.DataFrame(columns=[f.name for f in ARROW_SCHEMA])
        return dataset.to_table().to_pandas()

    def iter_records(self, source: str | None = None, batch_size: int = 4096) -> Iterator[dict]:
        """Stream rows without materializing the corpus. For large local runs."""
        dataset = self.dataset(source)
        if dataset is None:
            return
        for batch in dataset.to_batches(batch_size=batch_size):
            yield from batch.to_pylist()

    # --- summaries -------------------------------------------------------
    def stats(self) -> list[dict[str, Any]]:
        """Per-source summary. Backs ``python -m ingest.cli stats``."""
        out: list[dict[str, Any]] = []
        if not self.root.exists():
            return out
        for source_dir in sorted(self.root.glob("source=*")):
            source = source_dir.name.split("=", 1)[1]
            dataset = self.dataset(source)
            if dataset is None:
                continue
            table = dataset.to_table(
                columns=["id", "author_id", "timestamp", "parent_id", "text", "lang"]
            )
            timestamps = [t for t in table.column("timestamp").to_pylist() if t is not None]
            parents = table.column("parent_id").to_pylist()
            texts = table.column("text").to_pylist()
            out.append(
                {
                    "source": source,
                    "records": table.num_rows,
                    "authors": len(set(table.column("author_id").to_pylist())),
                    "first": min(timestamps).isoformat() if timestamps else None,
                    "last": max(timestamps).isoformat() if timestamps else None,
                    "threaded_pct": (
                        round(100 * sum(p is not None for p in parents) / len(parents), 1)
                        if parents
                        else 0.0
                    ),
                    "median_chars": _median([len(t) for t in texts]) if texts else 0,
                    "langs": len({v for v in table.column("lang").to_pylist() if v}),
                    "partitions": len(list(source_dir.glob("date=*"))),
                }
            )
        return out


def _median(values: list[int]) -> int:
    if not values:
        return 0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) // 2


def partition_path(root: Path, source: str, when: datetime) -> Path:
    return root / f"source={source}" / f"date={when.astimezone(timezone.utc):%Y-%m-%d}"
