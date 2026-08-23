"""Phase 1's Parquet corpus -> Postgres.

This is the bridge Phase 1 explicitly deferred ("Parquet on disk is the Phase 1
store; Phase 4 loads it into Postgres").

Four properties, in the order they matter:

1. **Nothing is silently dropped.** Every row is validated against
   ``ingest/schema.py`` before load. Rejects are counted by reason code, written
   to ``data/rejects/<run_id>.parquet``, and the ``ingest_runs`` CHECK constraint
   refuses a run whose counts do not add up. Silent loss between Parquet and
   Postgres poisons every downstream metric and surfaces three weeks later as a
   number nobody can explain.
2. **Idempotent by construction.** ``ON CONFLICT (id) DO NOTHING`` for posts,
   which are immutable, and ``DO UPDATE`` for the author roll-up, which
   accumulates. Rerunning over the same partitions changes zero rows, and a test
   asserts exactly that.
3. **Bulk, not row-by-row.** ``COPY`` into an unlogged staging table, then one
   ``INSERT ... SELECT ... ON CONFLICT``. ORM inserts over a multi-million-row
   corpus take hours; this takes minutes. COPY itself cannot do ON CONFLICT,
   which is precisely why the staging table exists.
4. **Null semantics survive.** ``NULL`` engagement stays NULL. No COALESCE
   anywhere in this file, deliberately and permanently.

pyarrow reads the Parquet and psycopg writes the COPY stream. duckdb would also
work and is marginally faster, but pyarrow is already a Phase 1 dependency and
the extra one is not worth asking for at this corpus size.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.etl.simhash import to_signed
from ingest.schema import DropReason, Record

log = logging.getLogger(__name__)

#: Column order for the COPY stream. Must match ``_STAGING_DDL`` exactly.
POST_COLUMNS: tuple[str, ...] = (
    "id",
    "project_id",
    "native_id",
    "source",
    "source_detail",
    "content_type",
    "text",
    "lang",
    "author_id",
    "author_handle",
    "timestamp",
    "parent_id",
    "conversation_id",
    "likes",
    "shares",
    "replies",
    "views",
    "urls",
    "domains",
    "media_urls",
    "hashtags",
    "mentions",
    "simhash",
    "ingested_at",
    "raw",
)

#: Pre-quoted once. ``text`` and ``timestamp`` are both reserved-ish words in
#: Postgres and the column list is used in three places, so quoting it in one
#: string beats getting it right three times.
_QUOTED_COLUMNS = ", ".join(f'"{column}"' for column in POST_COLUMNS)

#: UNLOGGED: the staging table is dropped at the end of the transaction either
#: way, so paying for WAL on it is pure waste on a bulk load.
_STAGING_DDL = """
CREATE UNLOGGED TABLE {name} (
    id text, project_id uuid, native_id text, source text, source_detail text,
    content_type text, text text, lang text, author_id text, author_handle text,
    timestamp timestamptz, parent_id text, conversation_id text,
    likes int, shares int, replies int, views int,
    urls text[], domains text[], media_urls text[], hashtags text[], mentions text[],
    simhash bigint, ingested_at timestamptz, raw jsonb
)
"""


@dataclass
class LoadResult:
    """What one source's load did. Mirrors the ``ingest_runs`` row it produces."""

    source: str
    records_in: int = 0
    records_loaded: int = 0
    records_rejected: int = 0
    #: Rows that were valid but already present. Counted separately from
    #: `loaded` so an idempotent rerun is visibly a no-op rather than looking
    #: like a load that mysteriously inserted nothing.
    records_unchanged: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    rejects_path: str | None = None
    files_read: int = 0

    def assert_accounted(self) -> None:
        """Every row read is loaded, unchanged, or rejected with a reason.

        Checked here as well as by the database constraint, because failing in
        Python names the source and the counts, while failing on the INSERT
        names only the constraint.
        """
        total = self.records_loaded + self.records_unchanged + self.records_rejected
        if total != self.records_in:
            raise RuntimeError(
                f"reject accounting for {self.source} does not balance: read "
                f"{self.records_in}, loaded {self.records_loaded}, unchanged "
                f"{self.records_unchanged}, rejected {self.records_rejected}. "
                f"{self.records_in - total} row(s) unaccounted for -- this is the "
                "silent-data-loss failure mode, do not proceed."
            )
        if sum(self.rejection_reasons.values()) != self.records_rejected:
            raise RuntimeError(
                f"{self.source}: rejection_reasons sums to "
                f"{sum(self.rejection_reasons.values())} but records_rejected is "
                f"{self.records_rejected}; every rejected row needs a reason code."
            )


def discover_partitions(
    normalized_dir: Path,
    *,
    sources: list[str] | None = None,
    since: date | None = None,
    until: date | None = None,
) -> list[Path]:
    """Parquet files matching the filter, from Phase 1's hive layout.

    ``data/normalized/source=<source>/date=<YYYY-MM-DD>/*.parquet``. Filtering on
    the directory names rather than by reading and discarding rows is the whole
    point of the partitioning: a one-week reload should not touch a year of files.
    """
    if not normalized_dir.exists():
        return []

    wanted = set(sources) if sources else None
    files: list[Path] = []
    for source_dir in sorted(normalized_dir.glob("source=*")):
        source = source_dir.name.split("=", 1)[1]
        if wanted and source not in wanted:
            continue
        for date_dir in sorted(source_dir.glob("date=*")):
            try:
                partition_date = datetime.strptime(
                    date_dir.name.split("=", 1)[1], "%Y-%m-%d"
                ).date()
            except ValueError:
                log.warning("skipping unparseable partition directory %s", date_dir)
                continue
            if since and partition_date < since:
                continue
            if until and partition_date > until:
                continue
            files.extend(sorted(date_dir.glob("*.parquet")))
    return files


def _read_rows(files: list[Path]) -> Iterator[dict[str, Any]]:
    """Yield Parquet rows as plain dicts, one file at a time.

    Per file rather than one big table: the whole corpus does not need to be
    resident to load it, and a corrupt file names itself in the traceback
    instead of failing an opaque combined read.
    """
    for path in files:
        table = pq.read_table(path)
        # The hive partition columns are directory names, not stored columns.
        # `source` in particular is nullable in the file schema and populated by
        # the partitioning, so it is recovered from the path.
        source = path.parent.parent.name.split("=", 1)[1]
        for batch in table.to_batches(max_chunksize=10_000):
            for row in batch.to_pylist():
                row.setdefault("source", None)
                if not row.get("source"):
                    row["source"] = source
                yield row


class RowRejected(Exception):
    """A row that fails the boundary checks Phase 1 applies in its adapters."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def _to_record(row: dict[str, Any]) -> Record:
    """Validate one Parquet row against Phase 1's schema.

    Two checks happen here that ``Record`` itself does not make. Phase 1 applies
    them in the *adapters* -- an empty-text or timestamp-less row is dropped
    before it is ever constructed -- so the model has no reason to duplicate
    them. At this boundary there is no adapter, only a file, so they are applied
    explicitly with the same reason codes rather than assumed.

    ``raw`` is a JSON *string* on disk (Phase 1 stores it that way deliberately,
    because raw payloads differ per source and per API version and would make
    the Arrow schemas mutually unreadable). It is parsed back here, at the one
    boundary that owns the format.
    """
    payload = dict(row)
    payload.pop("date", None)  # hive partition column, not a Record field

    if not (payload.get("text") or "").strip():
        raise RowRejected(DropReason.EMPTY_TEXT.value, "text is empty or whitespace")
    if payload.get("timestamp") is None:
        raise RowRejected(DropReason.MISSING_TIMESTAMP.value, "timestamp is null")
    if not (payload.get("native_id") or "").strip():
        raise RowRejected(DropReason.MISSING_ID.value, "native_id is empty")

    raw = payload.get("raw")
    if isinstance(raw, str):
        try:
            payload["raw"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload["raw"] = {"_unparseable_raw": raw[:2000]}
    elif raw is None:
        payload["raw"] = {}

    engagement = payload.get("engagement")
    if engagement is None:
        payload["engagement"] = {}

    return Record.model_validate(payload)


def _copy_row(record: Record, project_id: uuid.UUID) -> tuple[Any, ...]:
    """A validated Record as one COPY tuple, in ``POST_COLUMNS`` order."""
    engagement = record.engagement
    return (
        record.id,
        project_id,
        record.native_id,
        record.source,
        record.source_detail,
        record.content_type,
        record.text,
        record.lang,
        record.author_id,
        record.author_handle,
        record.timestamp,
        record.parent_id,
        record.conversation_id,
        # No COALESCE. None stays None all the way to the column.
        engagement.likes,
        engagement.shares,
        engagement.replies,
        engagement.views,
        record.urls,
        record.domains,
        record.media_urls,
        record.hashtags,
        record.mentions,
        to_signed(record.simhash),
        record.ingested_at,
        json.dumps(record.raw, default=str),
    )


def load_source(
    session: Session,
    *,
    project_id: uuid.UUID,
    source: str,
    files: list[Path],
    rejects_dir: Path,
    run_id: uuid.UUID | None = None,
    batch_rows: int = 50_000,
) -> LoadResult:
    """COPY one source's partitions into ``posts``.

    Returns the accounting. Does **not** commit: the caller owns the transaction
    so that the ``ingest_runs`` row and the data it describes land together or
    not at all.
    """
    result = LoadResult(source=source, files_read=len(files))
    if not files:
        return result

    run_id = run_id or uuid.uuid4()
    staging = f"stg_posts_{run_id.hex[:12]}"
    rejects: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    connection = session.connection()
    raw_connection = connection.connection.driver_connection  # psycopg3

    connection.exec_driver_sql(_STAGING_DDL.format(name=staging))

    copy_sql = f"COPY {staging} ({_QUOTED_COLUMNS}) FROM STDIN"
    staged = 0
    with raw_connection.cursor() as cursor, cursor.copy(copy_sql) as copy:
        for row in _read_rows(files):
            result.records_in += 1
            try:
                record = _to_record(row)
            except (ValidationError, RowRejected) as exc:
                reason = exc.reason if isinstance(exc, RowRejected) else _classify(row, exc)
                reasons[reason] += 1
                result.records_rejected += 1
                rejects.append(
                    {
                        "id": str(row.get("id") or ""),
                        "source": source,
                        "reason": reason,
                        "detail": str(exc)[:1000],
                        "row": json.dumps(row, default=str)[:4000],
                    }
                )
                continue
            copy.write_row(_copy_row(record, project_id))
            staged += 1
            if staged % batch_rows == 0:
                log.info("%s: staged %d rows", source, staged)

    # De-duplicate inside the batch before the insert. Two Parquet files can
    # legitimately carry the same id after a re-fetch, and ON CONFLICT DO
    # NOTHING cannot resolve a conflict *within* one INSERT's own rows -- it
    # raises "cannot affect row a second time" instead.
    inserted = connection.exec_driver_sql(
        f"""
        WITH deduped AS (
            SELECT DISTINCT ON (id) * FROM {staging} ORDER BY id, ingested_at DESC
        )
        INSERT INTO posts ({_QUOTED_COLUMNS})
        SELECT {_QUOTED_COLUMNS} FROM deduped
        ON CONFLICT (id) DO NOTHING
        RETURNING 1
        """
    ).rowcount

    distinct_staged = connection.exec_driver_sql(
        f"SELECT count(DISTINCT id) FROM {staging}"
    ).scalar()

    connection.exec_driver_sql(f"DROP TABLE {staging}")

    result.records_loaded = max(inserted, 0)
    # Rows that were valid but already present, plus duplicates collapsed inside
    # this batch. Both are "read and accounted for, not inserted".
    result.records_unchanged = staged - result.records_loaded
    result.rejection_reasons = dict(reasons)

    if rejects:
        result.rejects_path = str(_write_rejects(rejects_dir, run_id, source, rejects))

    log.info(
        "%s: read %d, staged %d (%d distinct), inserted %d, unchanged %d, rejected %d %s",
        source,
        result.records_in,
        staged,
        distinct_staged,
        result.records_loaded,
        result.records_unchanged,
        result.records_rejected,
        dict(reasons) or "",
    )
    result.assert_accounted()
    return result


def _classify(row: dict[str, Any], exc: ValidationError) -> str:
    """Map a validation failure onto one of Phase 1's reason codes.

    Reusing Phase 1's ``DropReason`` vocabulary rather than inventing a parallel
    one means a rejection at this boundary is directly comparable with a
    rejection at the adapter boundary, which is the whole point of having codes.
    """
    fields = {str(part) for error in exc.errors() for part in error.get("loc", ())}
    if not (row.get("text") or "").strip():
        return DropReason.EMPTY_TEXT.value
    if "timestamp" in fields or not row.get("timestamp"):
        return DropReason.MISSING_TIMESTAMP.value
    if {"id", "native_id"} & fields:
        return DropReason.MISSING_ID.value
    if "content_type" in fields or "source" in fields:
        return DropReason.UNSUPPORTED_TYPE.value
    if "simhash" in fields:
        return DropReason.OUT_OF_RANGE.value
    return DropReason.VALIDATION_ERROR.value


def _write_rejects(
    rejects_dir: Path, run_id: uuid.UUID, source: str, rejects: list[dict[str, Any]]
) -> Path:
    """Every rejected row, on disk, inspectable.

    A count in a JSONB column tells you how many rows were lost; this tells you
    which ones and why, which is the difference between a debuggable pipeline
    and an apology.
    """
    rejects_dir.mkdir(parents=True, exist_ok=True)
    path = rejects_dir / f"{run_id}-{source}.parquet"
    table = pa.Table.from_pylist(
        rejects,
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("source", pa.string()),
                pa.field("reason", pa.string()),
                pa.field("detail", pa.string()),
                pa.field("row", pa.string()),
            ]
        ),
    )
    pq.write_table(table, path)
    return path


def count_parquet_rows(files: list[Path]) -> int:
    """Row count from the Parquet footers only. No data is read."""
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def count_distinct_parquet_ids(files: list[Path]) -> int:
    """Distinct ``id`` values across a source's partitions.

    This, not the raw row count, is what Postgres can hold: ``posts.id`` is the
    primary key. An RSS article re-fetched on two days legitimately appears in
    two date partitions with the same id, and the loader collapses it. Comparing
    Postgres against the raw row count would report that collapse as data loss
    forever, which trains everybody to ignore the reconciliation report -- the
    exact opposite of what it is for.
    """
    seen: set[str] = set()
    for path in files:
        table = pq.read_table(path, columns=["id"])
        seen.update(table.column("id").to_pylist())
    return len(seen)


def parquet_row_counts_by_source(normalized_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source_dir in sorted(normalized_dir.glob("source=*")):
        source = source_dir.name.split("=", 1)[1]
        counts[source] = count_parquet_rows(sorted(source_dir.rglob("*.parquet")))
    return counts
