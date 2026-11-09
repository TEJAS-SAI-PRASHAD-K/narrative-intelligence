"""Three-way comparison on the Indian fact-check set: the evaluation harness.

The harness consumes scores computed elsewhere (a Colab GPU, a CPU worker). It
loads no model and needs no GPU. It reads one Parquet file per model and turns
the rows into the numbers that decide whether a model ships.

Read ``docs/roadmap.md`` ("How to report the fact-check evaluation") and
``docs/laya-finetune.md`` step 6 first. The rules that shape the code:

**Recall alone means nothing.** The fact-check set holds only claims that a
publisher chose to debunk, all of them false. A model that flags everything has
recall 1.0. Recall is therefore always reported with the flag rate on the
unlabelled corpus at the same threshold, and as their ratio, the lift.

**A null score is not "not flagged".** A model that cannot score a row produced no
answer. Treating the missing answer as "not flagged" lowers recall and the flag
rate in a way that looks like a conservative model. Null scores leave the
denominator, and each one is counted under its ``skip_reason``.

**The threshold belongs to the model.** It is read off each model's own
validation rows at a precision target. A threshold shared between models belongs
to none of them.

**Every comparison is on the same rows.** The paired bootstrap resamples one set
of row indices and applies it to both models.

This module imports nothing from ``ingest`` at import time. The claim table
(``ingest.factcheck.read_deduplicated_factchecks``) is read in
``load_claim_universe`` and the rest of the module works on plain ids, so the
statistics stay testable without any corpus on disk.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: The parts a score file may hold, and what each one is for.
PARTS = ("val", "test", "claims", "corpus_title", "corpus_text")
#: Parts that carry a 0/1 ``label``. Every other part must have a null label.
LABELLED_PARTS = ("val", "test")
CORPUS_PARTS = ("corpus_title", "corpus_text")
REQUIRED_COLUMNS = ("part", "id", "lang", "label", "score", "source_detail", "skip_reason")

#: Languages reported on their own. ``all`` is reported as well and counts every row.
LANGUAGES = ("en", "hi")
ALL = "all"

#: Reason code for a null score that arrives with no ``skip_reason``.
UNSPECIFIED_SKIP = "unspecified"


class EvalInputError(ValueError):
    """An input the harness refuses to work with. Never caught and defaulted."""


class ScoreFileError(EvalInputError):
    """A score file that does not match the input schema."""


class SplitMismatch(EvalInputError):
    """The models do not share the same ``test`` rows."""


class ManifestMismatch(EvalInputError):
    """The frozen evaluation set on disk and the current fact-check rows disagree."""


# ---------------------------------------------------------------------------
# Loading and validating one model's score file
# ---------------------------------------------------------------------------
@dataclass
class ModelScores:
    """One validated score file."""

    name: str
    path: Path | None
    sha256: str | None
    frame: pd.DataFrame
    #: Columns present in the file that the harness does not read.
    ignored_columns: list[str] = field(default_factory=list)

    def part(self, name: str) -> pd.DataFrame:
        return self.frame[self.frame["part"] == name]


def _is_string_type(arrow_type: Any) -> bool:
    import pyarrow as pa

    return bool(
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_null(arrow_type)
    )


def _check_arrow_schema(schema: Any, source: str) -> None:
    """Reject a column with a type the harness would have to guess about."""
    import pyarrow as pa

    names = set(schema.names)
    missing = [column for column in REQUIRED_COLUMNS if column not in names]
    if missing:
        raise ScoreFileError(
            f"{source}: missing column(s) {missing}; need {list(REQUIRED_COLUMNS)}"
        )
    for column in ("part", "id", "lang", "source_detail", "skip_reason"):
        if not _is_string_type(schema.field(column).type):
            raise ScoreFileError(
                f"{source}: column {column!r} must be a string, is {schema.field(column).type}"
            )
    label_type = schema.field("label").type
    if not (
        pa.types.is_integer(label_type)
        or pa.types.is_boolean(label_type)
        or pa.types.is_floating(label_type)
        or pa.types.is_null(label_type)
    ):
        raise ScoreFileError(f"{source}: column 'label' must be numeric 0/1, is {label_type}")
    score_type = schema.field("score").type
    if not (pa.types.is_floating(score_type) or pa.types.is_null(score_type)):
        raise ScoreFileError(f"{source}: column 'score' must be a float, is {score_type}")


def validate_frame(frame: pd.DataFrame, name: str) -> ModelScores:
    """Check the values of a score table and return it in normal form.

    Normal form: ``score`` and ``label`` are float64 with NaN for null, and a row
    with no score has a ``skip_reason`` (``unspecified`` when the file gave none).
    Raises ``ScoreFileError`` on the first class of problem found, naming it.
    """
    source = f"score file for model {name!r}"
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ScoreFileError(f"{source}: missing column(s) {missing}")
    ignored = [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    if ignored:
        log.warning("%s: ignoring unknown column(s) %s", source, ignored)

    frame = frame[list(REQUIRED_COLUMNS)].copy()
    if not len(frame):
        raise ScoreFileError(f"{source}: the file has no rows")

    if frame["part"].isna().any():
        raise ScoreFileError(f"{source}: {int(frame['part'].isna().sum())} row(s) with a null part")
    unknown = sorted(set(frame["part"]) - set(PARTS))
    if unknown:
        raise ScoreFileError(f"{source}: unknown part value(s) {unknown}; allowed {list(PARTS)}")

    ids = frame["id"]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise ScoreFileError(f"{source}: null or empty id")
    duplicated = frame[frame.duplicated(["part", "id"], keep=False)]
    if len(duplicated):
        example = duplicated.iloc[0]
        raise ScoreFileError(
            f"{source}: {len(duplicated)} row(s) share a (part, id) pair, for example "
            f"part={example['part']!r} id={example['id']!r}. One score per row is required."
        )

    score = pd.to_numeric(frame["score"], errors="coerce").astype("float64")
    if np.isinf(score.to_numpy()).any():
        raise ScoreFileError(f"{source}: an infinite score")
    valid = score.dropna()
    if len(valid) and ((valid < 0) | (valid > 1)).any():
        raise ScoreFileError(
            f"{source}: scores must be probabilities in [0, 1], found range "
            f"[{valid.min():.4g}, {valid.max():.4g}]"
        )
    # NaN and null both mean "no score". A NaN must never reach a comparison,
    # where `nan >= threshold` is False and reads as "not flagged".
    frame["score"] = score

    has_score = frame["score"].notna()
    skip = frame["skip_reason"].where(frame["skip_reason"].notna(), None)
    if (has_score & skip.notna()).any():
        row = frame[has_score & skip.notna()].iloc[0]
        raise ScoreFileError(
            f"{source}: a row has both a score and a skip_reason "
            f"(part={row['part']!r} id={row['id']!r}); one of the two is wrong"
        )
    frame["skip_reason"] = skip.where(has_score | skip.notna(), UNSPECIFIED_SKIP)

    label = pd.to_numeric(frame["label"], errors="coerce").astype("float64")
    labelled = frame["part"].isin(LABELLED_PARTS)
    if label[labelled].isna().any():
        raise ScoreFileError(f"{source}: a val or test row has a null label")
    if (~label[labelled].isin([0.0, 1.0])).any():
        raise ScoreFileError(f"{source}: val and test labels must be 0 or 1")
    if label[~labelled].notna().any():
        raise ScoreFileError(
            f"{source}: label must be null outside val and test; unlabelled parts have no label"
        )
    frame["label"] = label

    for part in LABELLED_PARTS:
        if not (frame["part"] == part).any():
            raise ScoreFileError(
                f"{source}: no {part!r} rows. Each model needs its own val and test."
            )
    overlap = set(frame.loc[frame["part"] == "val", "id"]) & set(
        frame.loc[frame["part"] == "test", "id"]
    )
    if overlap:
        raise ScoreFileError(
            f"{source}: {len(overlap)} id(s) appear in both val and test (leakage), "
            f"for example {sorted(overlap)[0]!r}"
        )

    frame = frame.reset_index(drop=True)
    return ModelScores(name=name, path=None, sha256=None, frame=frame, ignored_columns=ignored)


def load_score_file(path: Path, name: str | None = None) -> ModelScores:
    """Read and validate ``<model>.parquet``. A malformed file raises, loudly."""
    import pyarrow.parquet as pq

    path = Path(path)
    name = name or path.stem
    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is the same refusal
        raise ScoreFileError(f"{path}: not a readable Parquet file ({exc})") from exc
    _check_arrow_schema(table.schema, str(path))
    scores = validate_frame(table.to_pandas(), name)
    scores.path = path
    scores.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return scores


def load_score_dir(directory: Path) -> dict[str, ModelScores]:
    """Every ``*.parquet`` in a directory, keyed by file stem, in name order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise EvalInputError(f"{directory} is not a directory")
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise EvalInputError(f"no <model>.parquet files in {directory}")
    return {file.stem: load_score_file(file) for file in files}


def check_shared_test_split(models: dict[str, ModelScores]) -> None:
    """Refuse unless every model has the same test ids, labels and languages.

    A comparison on different rows is not a comparison. The same id with a
    different label means the files came from different splits.
    """
    reference_name = next(iter(models))
    reference = models[reference_name].part("test").set_index("id")
    for name, scores in models.items():
        test = scores.part("test").set_index("id")
        if set(test.index) != set(reference.index):
            only_here = sorted(set(test.index) - set(reference.index))[:3]
            only_there = sorted(set(reference.index) - set(test.index))[:3]
            raise SplitMismatch(
                f"models {reference_name!r} and {name!r} do not have the same test ids "
                f"({len(reference)} vs {len(test)} rows; only in {name}: {only_here}, "
                f"only in {reference_name}: {only_there})"
            )
        aligned = test.loc[reference.index]
        if not np.array_equal(aligned["label"].to_numpy(), reference["label"].to_numpy()):
            raise SplitMismatch(
                f"models {reference_name!r} and {name!r} give different labels "
                "for the same test ids"
            )


# ---------------------------------------------------------------------------
# The claim set, and freezing it
# ---------------------------------------------------------------------------
#: Reason codes for a claim id that is in a score file and is left out of recall.
CLAIM_COLLAPSED_DUPLICATE = "collapsed_duplicate"
CLAIM_NOT_KNOWN_FALSE = "label_not_false"
CLAIM_NOT_IN_FROZEN_SET = "not_in_frozen_set"
#: The id is in no fact-check row that the reader returned, kept or collapsed.
CLAIM_UNKNOWN_ID = "unknown_claim_id"

MANIFEST_COLUMNS = ("record_id", "url", "claim_sha256")


def default_manifest_path() -> Path:
    """``configs/factcheck_eval_manifest.csv``. Absent until someone freezes the set."""
    from modeling.config import CONFIG_DIR

    return CONFIG_DIR / "factcheck_eval_manifest.csv"


def claim_hash(claim: str) -> str:
    """SHA-256 of the claim text exactly as stored. The manifest keeps this, not the text."""
    return hashlib.sha256(claim.encode("utf-8")).hexdigest()


def _is_false(row: dict[str, Any]) -> bool:
    return str(row.get("label") or "").strip().lower() == "false"


def build_manifest(kept_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The manifest rows for a deduplicated read: the known-false claims, by record id.

    The frozen set is exactly what the harness scores recall on, so only rows
    labelled ``false`` go in. Sorted by ``record_id`` so the file is stable.
    """
    rows: list[dict[str, str]] = []
    for item in kept_rows:
        if not _is_false(item):
            continue
        if not item.get("claim"):
            raise EvalInputError(
                f"fact-check row {item.get('record_id')!r} has no claim text; cannot hash it"
            )
        rows.append(
            {
                "record_id": str(item["record_id"]),
                "url": str(item.get("url") or ""),
                "claim_sha256": claim_hash(str(item["claim"])),
            }
        )
    rows.sort(key=lambda r: r["record_id"])
    return rows


def write_manifest(rows: list[dict[str, str]], path: Path, *, overwrite: bool = False) -> Path:
    """Write the manifest CSV. Refuses to replace a frozen set unless told to.

    Replacing the file silently would make "frozen" mean nothing, so an existing
    file is an error unless ``overwrite`` is set, which is a deliberate act.
    """
    import csv

    path = Path(path)
    if not rows:
        raise EvalInputError("no known-false claims to freeze; the manifest would be empty")
    if path.exists() and not overwrite:
        raise EvalInputError(
            f"{path} already exists. The evaluation set is frozen; pass overwrite to replace it "
            "on purpose, and say why in the commit."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_manifest(path: Path) -> list[dict[str, str]] | None:
    """The manifest rows, or ``None`` when no manifest exists (the set is not frozen)."""
    import csv

    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ManifestMismatch(
                f"{path}: header must be {list(MANIFEST_COLUMNS)}, is {reader.fieldnames}"
            )
        rows = [dict(r) for r in reader]
    if not rows:
        raise ManifestMismatch(f"{path} has no rows; delete it or rebuild it")
    ids = [r["record_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ManifestMismatch(f"{path}: a record_id appears more than once")
    for r in rows:
        digest = r["claim_sha256"] or ""
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ManifestMismatch(f"{path}: record {r['record_id']!r} has no valid SHA-256")
    return rows


def verify_manifest(
    manifest: list[dict[str, str]],
    kept_rows: list[dict[str, Any]],
    collapsed: list[tuple[str, str, int]],
) -> None:
    """Stop unless every manifest row is present and unchanged in the current rows.

    Checks, for each manifest row: the record exists as a kept row (a row that is
    now a collapsed duplicate counts as missing, because the set it was frozen in
    has changed), the URL is the same, and the claim hash is the same. Raises
    ``ManifestMismatch`` listing every problem category with examples.
    """
    kept = {str(r["record_id"]): r for r in kept_rows}
    collapsed_into = {record_id: rep for record_id, rep, _ in collapsed}
    missing: list[str] = []
    now_duplicate: list[str] = []
    hash_changed: list[str] = []
    url_changed: list[str] = []
    for entry in manifest:
        record_id = entry["record_id"]
        row = kept.get(record_id)
        if row is None:
            if record_id in collapsed_into:
                rep = collapsed_into[record_id]
                now_duplicate.append(f"{record_id} (now a duplicate of {rep})")
            else:
                missing.append(record_id)
            continue
        if claim_hash(str(row.get("claim") or "")) != entry["claim_sha256"]:
            hash_changed.append(record_id)
        if str(row.get("url") or "") != entry["url"]:
            url_changed.append(record_id)

    problems: list[str] = []
    for label, items in (
        ("missing from the fact-check rows", missing),
        ("no longer kept after claim deduplication", now_duplicate),
        ("claim hash differs", hash_changed),
        ("URL differs", url_changed),
    ):
        if items:
            more = f", and {len(items) - 5} more" if len(items) > 5 else ""
            shown = ", ".join(items[:5]) + more
            problems.append(f"{len(items)} row(s) {label}: {shown}")
    if problems:
        raise ManifestMismatch(
            "The frozen evaluation set does not match the fact-check data on disk. "
            + "; ".join(problems)
            + ". Do not report numbers from a changed set. Restore the data, or rebuild the "
            "manifest on purpose and record why."
        )


@dataclass
class ClaimUniverse:
    """Which claim ids count towards recall, and why the others do not."""

    #: Known-false claim ids that recall is computed on.
    eval_ids: frozenset[str]
    #: Ids left out on purpose, mapped to their reason code.
    excluded: dict[str, str]
    frozen: bool
    manifest_path: Path | None = None
    manifest_rows: int = 0
    #: Counts from the deduplicating reader, for the report.
    dedup_counts: dict[str, int] = field(default_factory=dict)

    def reason_for_exclusion(self, claim_id: str) -> str | None:
        """The reason code if ``claim_id`` is not an evaluation claim, else ``None``."""
        if claim_id in self.eval_ids:
            return None
        return self.excluded.get(claim_id, CLAIM_UNKNOWN_ID)


def build_claim_universe(
    kept_rows: list[dict[str, Any]],
    collapsed: list[tuple[str, str, int]],
    *,
    manifest: list[dict[str, str]] | None = None,
    manifest_path: Path | None = None,
    dedup_counts: dict[str, int] | None = None,
) -> ClaimUniverse:
    """Decide which ids are evaluation claims.

    Without a manifest the set is the deduplicated rows labelled ``false``. With
    one, the manifest is verified first (a mismatch raises) and the set is
    exactly the manifest ids: a row added to the data after freezing is not part
    of the frozen set and is counted as ``not_in_frozen_set``.
    """
    if manifest is not None:
        verify_manifest(manifest, kept_rows, collapsed)
    frozen_ids = {entry["record_id"] for entry in manifest} if manifest is not None else None

    eval_ids: set[str] = set()
    excluded: dict[str, str] = {}
    for record_id, _, _ in collapsed:
        excluded[record_id] = CLAIM_COLLAPSED_DUPLICATE
    for item in kept_rows:
        record_id = str(item["record_id"])
        if not _is_false(item):
            excluded[record_id] = CLAIM_NOT_KNOWN_FALSE
        elif frozen_ids is not None and record_id not in frozen_ids:
            excluded[record_id] = CLAIM_NOT_IN_FROZEN_SET
        else:
            eval_ids.add(record_id)
    return ClaimUniverse(
        eval_ids=frozenset(eval_ids),
        excluded=excluded,
        frozen=manifest is not None,
        manifest_path=manifest_path,
        manifest_rows=len(manifest) if manifest is not None else 0,
        dedup_counts=dict(dedup_counts or {}),
    )


def load_claim_universe(data_dir: Path, manifest_path: Path | None = None) -> ClaimUniverse:
    """Read ``factchecks.parquet`` through the deduplicating reader and build the set.

    ``ingest`` is imported here and nowhere else in this module. Uses the manifest
    at ``manifest_path`` (default ``configs/factcheck_eval_manifest.csv``) when it
    exists, and raises ``ManifestMismatch`` if the data disagrees with it.
    """
    from ingest.config import Settings
    from ingest.factcheck import read_deduplicated_factchecks

    result = read_deduplicated_factchecks(Settings(data_dir=Path(data_dir), _env_file=None))
    path = Path(manifest_path) if manifest_path else default_manifest_path()
    return build_claim_universe(
        result.kept,
        result.collapsed,
        manifest=read_manifest(path),
        manifest_path=path,
        dedup_counts=result.counts,
    )


def freeze_manifest_from_data(
    data_dir: Path, path: Path | None = None, *, overwrite: bool = False
) -> tuple[Path, int]:
    """Write the manifest for the deduplicated known-false claims on disk."""
    from ingest.config import Settings
    from ingest.factcheck import read_deduplicated_factchecks

    result = read_deduplicated_factchecks(Settings(data_dir=Path(data_dir), _env_file=None))
    rows = build_manifest(result.kept)
    target = write_manifest(rows, path or default_manifest_path(), overwrite=overwrite)
    return target, len(rows)
