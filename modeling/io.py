"""Reading Phase 1's corpus and writing Phase 2's scored tables.

The output contract lives here, as explicit Arrow schemas. Phase 4 loads these
files; if a column's type drifts, this is the file that should have stopped it.

Three properties this module is responsible for, all of which are acceptance
criteria:

* **Joinability.** Every scored row's ``record_id`` / ``author_id`` exists in
  Phase 1. :func:`assert_joinable` enforces it on write, not in a notebook.
* **Idempotency.** Rerunning ``score`` over unchanged inputs with unchanged
  model versions writes zero rows. Row identity is (key columns), and row
  *content* is hashed excluding the ``scored_at`` / ``generated_at`` clock so a
  second run is a genuine no-op rather than a timestamp churn.
* **Resumability.** :func:`already_scored` returns the keys already carrying the
  current model versions, so a killed run resumes instead of recomputing.

Nulls are load-bearing. ``null`` means "this model did not run on this row" and
Phase 4 renders it as "not assessed". Never substitute a default; a fabricated
0.0 is indistinguishable from a confident negative.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from modeling.config import ModelingSettings, get_settings, manifest_hash

log = logging.getLogger(__name__)

EMOTIONS: tuple[str, ...] = ("fear", "anger", "disgust", "joy", "surprise", "sadness", "neutral")

_MODEL_VERSIONS_TYPE = pa.map_(pa.string(), pa.string())
_EMOTION_TYPE = pa.struct([pa.field(name, pa.float32()) for name in EMOTIONS])
_TOP_FEATURE_TYPE = pa.list_(
    pa.struct([pa.field("name", pa.string()), pa.field("contribution", pa.float32())])
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoredTable:
    """One table in the output contract."""

    name: str
    schema: pa.Schema
    #: Columns that identify a row. Idempotency and merge are defined on these.
    keys: tuple[str, ...]
    #: Column that must exist in Phase 1, if any. Enforced by assert_joinable.
    foreign_key: str | None = None
    #: Hive partition columns. Empty means one file.
    partition_by: tuple[str, ...] = ()
    #: Columns excluded from the content hash: clocks change on every run and
    #: would defeat idempotency if they counted as content.
    volatile: tuple[str, ...] = ("scored_at", "generated_at")


RECORD_SCORES = ScoredTable(
    name="record_scores",
    keys=("record_id",),
    foreign_key="record_id",
    partition_by=("source",),
    schema=pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),  # partition key only
            pa.field("misinfo_prob", pa.float32()),
            pa.field("stance", pa.string()),
            pa.field("stance_conf", pa.float32()),
            pa.field("toxicity", pa.float32()),
            pa.field("sentiment", pa.string()),
            pa.field("sentiment_score", pa.float32()),
            pa.field("emotion", _EMOTION_TYPE),
            pa.field("anomaly_score", pa.float32()),
            pa.field("skip_reasons", pa.list_(pa.string())),
            pa.field("model_versions", _MODEL_VERSIONS_TYPE),
            pa.field("scored_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
)

NARRATIVES = ScoredTable(
    name="narratives",
    keys=("narrative_id",),
    schema=pa.schema(
        [
            pa.field("narrative_id", pa.string(), nullable=False),
            pa.field("label", pa.string()),
            pa.field("label_source", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("size", pa.int64()),
            pa.field("author_count", pa.int64()),
            pa.field("first_seen", pa.timestamp("us", tz="UTC")),
            pa.field("last_seen", pa.timestamp("us", tz="UTC")),
            pa.field("platforms", pa.list_(pa.string())),
            pa.field("top_domains", pa.list_(pa.string())),
            pa.field("top_hashtags", pa.list_(pa.string())),
            pa.field("centroid", pa.list_(pa.float32())),
            pa.field("velocity", pa.float32()),
            pa.field("severity", pa.float32()),
            pa.field("coherence", pa.float32()),
            pa.field("model_versions", _MODEL_VERSIONS_TYPE),
            pa.field("generated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
)

NARRATIVE_MEMBERSHIP = ScoredTable(
    name="narrative_membership",
    keys=("record_id", "narrative_id"),
    foreign_key="record_id",
    schema=pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("narrative_id", pa.string(), nullable=False),
            pa.field("membership_prob", pa.float32()),
            pa.field("is_representative", pa.bool_()),
            pa.field("generated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
)

AUTHOR_SCORES = ScoredTable(
    name="author_scores",
    keys=("author_id",),
    foreign_key="author_id",
    partition_by=("source",),
    schema=pa.schema(
        [
            pa.field("author_id", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("bot_prob", pa.float32()),
            pa.field("bot_top_features", _TOP_FEATURE_TYPE),
            pa.field("coordination_score", pa.float32()),
            pa.field("community_id", pa.string()),
            pa.field("community_size", pa.int64()),
            pa.field("anomalous", pa.float32()),
            pa.field("toxicity_mean", pa.float32()),
            pa.field("dominant_sentiment", pa.string()),
            pa.field("dominant_emotion", pa.string()),
            pa.field("narratives_touched", pa.list_(pa.string())),
            pa.field("skip_reasons", pa.list_(pa.string())),
            pa.field("model_versions", _MODEL_VERSIONS_TYPE),
            pa.field("scored_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
)

COORDINATION_EDGES = ScoredTable(
    name="coordination_edges",
    keys=("src_author_id", "dst_author_id", "evidence"),
    schema=pa.schema(
        [
            pa.field("src_author_id", pa.string(), nullable=False),
            pa.field("dst_author_id", pa.string(), nullable=False),
            pa.field("weight", pa.float32()),
            pa.field("evidence", pa.string(), nullable=False),
            pa.field("observations", pa.int64()),
            pa.field("window_start", pa.timestamp("us", tz="UTC")),
            pa.field("window_end", pa.timestamp("us", tz="UTC")),
            pa.field("generated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
)

MEDIA_SCORES = ScoredTable(
    name="media_scores",
    keys=("record_id", "media_url"),
    foreign_key="record_id",
    schema=pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("media_url", pa.string(), nullable=False),
            pa.field("deepfake_prob", pa.float32()),
            pa.field("manipulation_type", pa.string()),
            pa.field("frames_analyzed", pa.int64()),
            pa.field("face_detected", pa.bool_()),
            pa.field("explanation", pa.string()),
            pa.field("model_versions", _MODEL_VERSIONS_TYPE),
            pa.field("scored_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
)

#: Author cohort schema, deliberately unpopulated. The 135-cohort ideological
#: taxonomy is a labeling project of its own and is out of scope for Phase 2
#: (see the anti-goals). The schema is written down so Phase 4 can plan against
#: a shape rather than inventing one later.
AUTHOR_COHORTS = ScoredTable(
    name="author_cohorts",
    keys=("author_id", "cohort_id"),
    foreign_key="author_id",
    schema=pa.schema(
        [
            pa.field("author_id", pa.string(), nullable=False),
            pa.field("cohort_id", pa.string(), nullable=False),
            pa.field("cohort_label", pa.string()),
            pa.field("confidence", pa.float32()),
            pa.field("evidence", pa.string()),
            pa.field("generated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
)

TABLES: dict[str, ScoredTable] = {
    t.name: t
    for t in (
        RECORD_SCORES,
        NARRATIVES,
        NARRATIVE_MEMBERSHIP,
        AUTHOR_SCORES,
        COORDINATION_EDGES,
        MEDIA_SCORES,
        AUTHOR_COHORTS,
    )
}


# ---------------------------------------------------------------------------
# reading Phase 1
# ---------------------------------------------------------------------------
class CorpusReader:
    """Read-only view of Phase 1's corpus.

    Phase 2 never writes under ``data/normalized/`` or ``data/authors/``.
    """

    def __init__(self, settings: ModelingSettings | None = None, root: Path | None = None):
        self.settings = settings or get_settings()
        self.root = Path(root) if root else self.settings.normalized_dir
        self.authors_root = (
            Path(root).parent / "authors" if root else self.settings.authors_dir
        )

    def available_sources(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name.split("=", 1)[1] for p in self.root.glob("source=*") if p.is_dir())

    def records(
        self,
        *,
        sources: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """The corpus as a DataFrame, filtered.

        ``start``/``end`` are ``YYYY-MM-DD`` and filter on the hive ``date=``
        partition, so an out-of-range partition is never read off disk.
        """
        if not self.root.exists():
            log.warning("no Phase 1 corpus at %s; returning empty frame", self.root)
            return pd.DataFrame(columns=columns or [])

        wanted = set(sources) if sources else None
        frames: list[pd.DataFrame] = []
        for source_dir in sorted(self.root.glob("source=*")):
            source = source_dir.name.split("=", 1)[1]
            if wanted and source not in wanted:
                continue
            for date_dir in sorted(source_dir.glob("date=*")):
                date = date_dir.name.split("=", 1)[1]
                if start and date < start:
                    continue
                if end and date > end:
                    continue
                files = sorted(date_dir.glob("*.parquet"))
                if not files:
                    continue
                # Read file-by-file rather than handing pyarrow the whole list.
                # Phase 1's partitions are not byte-uniform: some were written
                # with `source` as a plain string and others as a dictionary
                # column, and pyarrow refuses to merge those two encodings.
                # Per-file reads sidestep the merge entirely; pandas then
                # reconciles the dtypes on concat.
                for file in files:
                    try:
                        frame = pq.read_table(file, columns=columns).to_pandas()
                    except Exception as exc:  # pragma: no cover - corrupt file
                        log.warning("unreadable parquet %s: %s", file, exc)
                        continue
                    if "source" not in frame.columns:
                        frame["source"] = source
                    else:
                        frame["source"] = frame["source"].astype(str)
                    frames.append(frame)
                if limit and sum(len(f) for f in frames) >= limit:
                    break
            if limit and sum(len(f) for f in frames) >= limit:
                break

        if not frames:
            return pd.DataFrame(columns=columns or [])
        out = pd.concat(frames, ignore_index=True)
        # Deterministic order: two runs must produce byte-identical scored rows.
        if "id" in out.columns:
            out = out.sort_values("id", kind="stable").reset_index(drop=True)
            # Defensive dedupe on id.
            #
            # Phase 1 dedupes within a source=/date= partition, which is what
            # keeps resume cheap. An article whose timestamp shifts between two
            # fetches lands in two partitions and survives as a genuine
            # duplicate id -- 2 of 4190 in the current news corpus. Phase 2
            # cannot fix that (ingest/ is read-only here), but it must not
            # propagate it: `record_scores` is keyed on record_id, so a
            # duplicate silently collapses at write time and the row counts stop
            # reconciling. Collapsing here, loudly, keeps the arithmetic honest.
            duplicated = out["id"].duplicated()
            if duplicated.any():
                examples = out.loc[duplicated, "id"].head(3).tolist()
                log.warning(
                    "%d duplicate record id(s) in the Phase 1 corpus; keeping the first of "
                    "each (e.g. %s). These are ids that appear under more than one date= "
                    "partition, which Phase 1's per-partition dedupe cannot see.",
                    int(duplicated.sum()),
                    examples,
                )
                out = out.loc[~duplicated].reset_index(drop=True)
        if limit:
            out = out.head(limit)
        return out

    def authors(self, sources: Iterable[str] | None = None) -> pd.DataFrame:
        """Phase 1's author roll-ups."""
        if not self.authors_root.exists():
            return pd.DataFrame()
        wanted = set(sources) if sources else None
        frames = []
        for source_dir in sorted(self.authors_root.glob("source=*")):
            source = source_dir.name.split("=", 1)[1]
            if wanted and source not in wanted:
                continue
            path = source_dir / "authors.parquet"
            if not path.exists():
                continue
            frame = pq.read_table(path).to_pandas()
            if "source" not in frame.columns:
                frame["source"] = source
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            "author_id", kind="stable"
        ).reset_index(drop=True)

    def record_ids(self, sources: Iterable[str] | None = None) -> set[str]:
        frame = self.records(sources=sources, columns=["id"])
        return set(frame["id"].tolist()) if len(frame) else set()

    def author_ids(self, sources: Iterable[str] | None = None) -> set[str]:
        frame = self.authors(sources=sources)
        return set(frame["author_id"].tolist()) if len(frame) else set()


# ---------------------------------------------------------------------------
# joinability
# ---------------------------------------------------------------------------
class OrphanRowsError(ValueError):
    """A scored row references a record/author that Phase 1 never emitted."""


def assert_joinable(
    frame: pd.DataFrame,
    table: ScoredTable,
    known_keys: set[str] | None,
    *,
    sample: int = 5,
) -> None:
    """Zero orphan rows. An acceptance criterion, enforced at write time.

    ``known_keys=None`` disables the check -- used only on the demo path, where
    the "corpus" is a fixture and the caller passes its own key set.
    """
    if known_keys is None or table.foreign_key is None or table.foreign_key not in frame.columns:
        return
    present = set(frame[table.foreign_key].dropna().astype(str).tolist())
    orphans = present - known_keys
    if orphans:
        raise OrphanRowsError(
            f"{table.name}: {len(orphans)} orphan {table.foreign_key} value(s) not present in "
            f"Phase 1 (e.g. {sorted(orphans)[:sample]}). Scored tables must join cleanly."
        )


# ---------------------------------------------------------------------------
# writing scored tables
# ---------------------------------------------------------------------------
def _canonical(value: Any) -> Any:
    """Normalize a cell so a freshly-computed row and the same row read back
    from Arrow hash identically.

    This is what makes idempotency real rather than nominal. Arrow round-trips
    change representation without changing meaning: a ``map<string,string>``
    comes back as a list of ``(k, v)`` tuples, a struct comes back as a dict of
    numpy scalars, an empty list column comes back as ``None``. Hashing the raw
    representation would mark every row "updated" on every rerun, and the
    acceptance criterion would be silently unmet.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (str, bytes)):
        return value if isinstance(value, str) else value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple)):
        items = list(value)
        # Arrow map -> list of 2-tuples. Restore the dict so it matches the
        # freshly-built row, which is a real dict.
        if items and all(isinstance(i, tuple) and len(i) == 2 for i in items):
            return {str(k): _canonical(v) for k, v in sorted(items, key=lambda kv: str(kv[0]))}
        return [_canonical(v) for v in items]
    if hasattr(value, "tolist"):  # numpy array
        return _canonical(value.tolist())
    if hasattr(value, "item"):  # numpy scalar
        return _canonical(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # An Arrow int64 column that contains nulls comes back from pandas as
        # float64, so a freshly-computed `community_size` of 5 and the same
        # value read from disk as 5.0 must hash identically. Without this, every
        # table with a nullable integer column reported all its rows as
        # "updated" on every rerun.
        if value.is_integer():
            return int(value)
        # Compare at *storage* precision, not at some arbitrary decimal place.
        #
        # Every float column in the contract is float32. A freshly-computed
        # float64 prediction and the same value read back from Parquet differ in
        # the low bits, so rounding to a fixed number of decimals still reports a
        # difference for values near 1 -- which made every scored row look
        # "updated" on a rerun even though nothing had changed. Casting through
        # float32 asks the only question that matters: would writing this produce
        # different bytes?
        import numpy as np

        return float(np.float32(value))
    return value


def _row_content_hash(row: dict[str, Any], volatile: tuple[str, ...]) -> str:
    payload = {k: _canonical(v) for k, v in sorted(row.items()) if k not in volatile}
    # Absent, null and empty all mean the same thing: nothing was recorded here.
    #
    # Dropping them also reconciles the representations Arrow round-trips
    # through: an empty `map<string,string>` comes back as `[]` while a freshly
    # built one is `{}`, and treating those as different made every row with an
    # empty model_versions look "updated" on every rerun.
    payload = {
        key: value
        for key, value in payload.items()
        if value is not None and value != [] and value != {}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _coerce_to_schema(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Build an Arrow table with the declared schema, adding missing columns as
    nulls and refusing unexpected ones.

    Missing-as-null is the honest default here: a scorer that did not run leaves
    its column absent, and absent must become ``null``, not a fabricated value.
    """
    known = {f.name for f in schema}
    unexpected = set(frame.columns) - known
    if unexpected:
        raise ValueError(
            f"unexpected column(s) {sorted(unexpected)} for this table; the output contract is "
            "closed. Put module-specific extras in an eval artifact, not in a scored table."
        )
    rows = frame.to_dict(orient="records")
    filled: list[dict[str, Any]] = []
    for row in rows:
        filled.append({name: _null_if_na(row.get(name, None)) for name in known})
    return pa.Table.from_pylist(filled, schema=schema)


def _null_if_na(value: Any) -> Any:
    """Turn pandas' NA sentinels back into ``None`` before Arrow sees them.

    A column that is null for every row becomes float64 NaN in pandas, and
    Arrow then rejects it against a string field with "Expected bytes, got a
    'float' object". The scorers are *meant* to produce all-null columns when a
    model did not run, so this is the normal case, not an edge case.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if value is pd.NaT:
        return None
    try:
        if value is pd.NA:
            return None
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return value


class ScoredStore:
    """Writer/reader for ``data/scored/``."""

    def __init__(self, settings: ModelingSettings | None = None, root: Path | None = None):
        self.settings = settings or get_settings()
        self.root = Path(root) if root else self.settings.scored_dir

    def path_for(self, table: ScoredTable, partition: dict[str, str] | None = None) -> Path:
        path = self.root / table.name
        for col in table.partition_by:
            value = (partition or {}).get(col, "unknown")
            path = path / f"{col}={value}"
        return path / "part-000.parquet"

    def read(self, name: str) -> pd.DataFrame:
        """Read a scored table back. Empty frame when it does not exist yet."""
        table = TABLES[name]
        base = self.root / name
        if not base.exists():
            return pd.DataFrame(columns=[f.name for f in table.schema])
        files = sorted(base.rglob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=[f.name for f in table.schema])
        frames = []
        for file in files:
            try:
                frames.append(pq.read_table(file).to_pandas())
            except Exception as exc:  # pragma: no cover
                log.warning("unreadable scored file %s: %s", file, exc)
        if not frames:
            return pd.DataFrame(columns=[f.name for f in table.schema])
        return pd.concat(frames, ignore_index=True)

    def already_scored(self, name: str, model_versions: dict[str, str]) -> set[tuple[str, ...]]:
        """Keys already carrying exactly these model versions.

        This is what makes scoring resumable *and* idempotent: the scorer skips
        computing them, and the writer then has nothing new to write. Rows
        scored by an older model version are deliberately *not* returned, so a
        retrain re-scores them.
        """
        table = TABLES[name]
        existing = self.read(name)
        if not len(existing) or "model_versions" not in existing.columns:
            return set()
        wanted = {str(k): str(v) for k, v in model_versions.items()}
        out: set[tuple[str, ...]] = set()
        for row in existing.to_dict(orient="records"):
            have = _as_version_dict(row.get("model_versions"))
            # Subset test, not equality: a row scored by aux+misinfo is still
            # current for an aux-only rerun.
            if all(have.get(k) == v for k, v in wanted.items()):
                out.add(tuple(str(row[k]) for k in table.keys))
        return out

    def write(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        known_keys: set[str] | None = None,
        merge: bool = True,
    ) -> dict[str, int]:
        """Write (or merge into) a scored table.

        Returns ``{"written", "updated", "unchanged"}``. When nothing changed,
        no file is touched at all -- that is the acceptance criterion "rerunning
        writes zero new rows", and touching mtimes would technically satisfy it
        while being useless in practice.
        """
        table = TABLES[name]
        if not len(frame):
            log.info("%s: nothing to write", name)
            return {"written": 0, "updated": 0, "unchanged": 0}

        frame = frame.copy()
        for col in table.keys:
            if col not in frame.columns:
                raise KeyError(f"{name}: key column {col!r} missing")
            frame[col] = frame[col].astype(str)
        assert_joinable(frame, table, known_keys)

        # Existing rows are always read, in both modes.
        #
        # `merge` controls what happens to rows the caller did *not* supply, not
        # whether the previous state is consulted. Replace mode (merge=False)
        # drops them -- a narrative that no longer exists must disappear rather
        # than linger as a stale row -- but it still compares the rows it does
        # supply against what is on disk. Skipping that comparison would make
        # every replace-mode rerun report its whole table as newly written, and
        # the idempotency guarantee would be nominal rather than real.
        existing = self.read(name)
        stamp = "generated_at" if "generated_at" in {f.name for f in table.schema} else "scored_at"
        if stamp not in frame.columns:
            frame[stamp] = utcnow()

        new_rows = {
            tuple(str(row[k]) for k in table.keys): row for row in frame.to_dict(orient="records")
        }
        old_rows: dict[tuple[str, ...], dict[str, Any]] = {}
        if len(existing):
            for row in existing.to_dict(orient="records"):
                old_rows[tuple(str(row[k]) for k in table.keys)] = row

        # Merge is column-wise, not row-wise.
        #
        # Stages fill different columns of the same table: the aux pass writes
        # toxicity and sentiment, the misinfo stage writes misinfo_prob. A
        # row-wise merge lets whichever stage ran last blank every column it does
        # not know about -- so each run wiped and re-added misinfo_prob, churning
        # the table forever and leaving a window where the column was null.
        # Columns the incoming frame does not carry keep their prior value.
        incoming_columns = set(frame.columns)

        written = updated = unchanged = 0
        merged: dict[tuple[str, ...], dict[str, Any]] = dict(old_rows) if merge else {}
        for key, row in new_rows.items():
            prior = old_rows.get(key)
            if prior is not None and merge:
                candidate = dict(prior)
                candidate.update({k: v for k, v in row.items() if k in incoming_columns})
                if "model_versions" in incoming_columns:
                    # Union, not overwrite: a row is current for every scorer
                    # that has ever produced a value for it.
                    candidate["model_versions"] = {
                        **_as_version_dict(prior.get("model_versions")),
                        **_as_version_dict(row.get("model_versions")),
                    }
                row = candidate
            if prior is None:
                merged[key] = row
                written += 1
            elif _row_content_hash(prior, table.volatile) == _row_content_hash(row, table.volatile):
                # Identical content: keep the *older* row so scored_at reflects
                # when the score was actually produced, not when it was re-checked.
                merged[key] = prior
                unchanged += 1
            else:
                merged[key] = row
                updated += 1

        dropped = 0 if merge else len(set(old_rows) - set(new_rows))
        if written == 0 and updated == 0 and dropped == 0 and len(old_rows):
            log.info("%s: %d rows unchanged, nothing written", name, unchanged)
            return {"written": 0, "updated": 0, "unchanged": unchanged}
        if dropped:
            log.info("%s: %d row(s) no longer produced and were dropped", name, dropped)

        out_frame = pd.DataFrame(list(merged.values()))
        # Deterministic row order on disk.
        out_frame = out_frame.sort_values(list(table.keys), kind="stable").reset_index(drop=True)

        base = self.root / name
        if base.exists():
            for file in base.rglob("*.parquet"):
                file.unlink()
        if table.partition_by:
            for partition_values, group in out_frame.groupby(
                list(table.partition_by), dropna=False, sort=True
            ):
                if not isinstance(partition_values, tuple):
                    partition_values = (partition_values,)
                partition = dict(
                    zip(table.partition_by, [str(v) for v in partition_values], strict=True)
                )
                path = self.path_for(table, partition)
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(_coerce_to_schema(group, table.schema), path, compression="zstd")
        else:
            path = self.path_for(table)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(_coerce_to_schema(out_frame, table.schema), path, compression="zstd")

        log.info(
            "%s: +%d new, ~%d updated, =%d unchanged (%d rows on disk)",
            name,
            written,
            updated,
            unchanged,
            len(out_frame),
        )
        return {"written": written, "updated": updated, "unchanged": unchanged}

    # --- manifest --------------------------------------------------------
    def update_manifest(
        self,
        *,
        table: str,
        rows: int,
        model_versions: dict[str, str],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Mirror Phase 1's manifest pattern: table -> rows, versions, provenance."""
        path = self.settings.scored_manifest_path
        payload: dict[str, Any] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("scored manifest at %s is corrupt; starting fresh", path)
        payload[table] = {
            "rows": rows,
            "model_versions": dict(model_versions),
            "generated_at": utcnow().isoformat(),
            "input_manifest_hash": manifest_hash(),
            **(extra or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def manifest(self) -> dict[str, Any]:
        path = self.settings.scored_manifest_path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def summary(self) -> list[dict[str, Any]]:
        out = []
        for name in TABLES:
            base = self.root / name
            if not base.exists():
                continue
            frame = self.read(name)
            out.append(
                {
                    "table": name,
                    "rows": len(frame),
                    "files": len(list(base.rglob("*.parquet"))),
                }
            )
        return out


def _as_version_dict(value: Any) -> dict[str, str]:
    """Normalize a ``model_versions`` cell read back from Arrow.

    Arrow ``map<string,string>`` round-trips through pandas as a list of
    ``(key, value)`` tuples, not a dict. Callers should not have to know that.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    try:
        return {str(k): str(v) for k, v in value}
    except (TypeError, ValueError):
        return {}


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def as_list(value) -> list:
    """Coerce a Parquet list cell to a plain Python list.

    Arrow list columns arrive as numpy arrays, and `value or []` on a numpy
    array raises "truth value of an array ... is ambiguous" -- or worse, on a
    one-element array, silently succeeds with the wrong semantics. Every read of
    a list-typed column goes through here.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    try:
        return list(value)
    except TypeError:
        return []


def empty_emotion() -> dict[str, float]:
    """All-zero emotion vector. Only for rows where the model *ran* and found
    nothing; a row the model skipped gets ``None`` for the whole struct."""
    return dict.fromkeys(EMOTIONS, 0.0)
