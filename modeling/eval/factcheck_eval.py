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
import json
import logging
from collections.abc import Collection
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


class InputsMismatch(EvalInputError):
    """The models were not scored on the same claim rows or the same corpus rows."""


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


#: Parts that carry no label, so a model cannot show a difference in rows through them.
SHARED_INPUT_PARTS = ("claims", "corpus_title", "corpus_text")


def check_shared_inputs(models: dict[str, ModelScores]) -> None:
    """Refuse unless every model scored the same claim ids and the same corpus ids.

    A flag rate on a corpus that one model saw in full and another saw in part is
    not a comparison, and a lift divides by it. The test rows have their own check.
    """
    reference_name = next(iter(models))
    for part in SHARED_INPUT_PARTS:
        reference = set(models[reference_name].part(part)["id"])
        for name, scores in models.items():
            ids = set(scores.part(part)["id"])
            if ids == reference:
                continue
            only_here = sorted(ids - reference)[:3]
            only_there = sorted(reference - ids)[:3]
            raise InputsMismatch(
                f"models {reference_name!r} and {name!r} were not scored on the same {part} "
                f"rows ({len(reference)} vs {len(ids)} ids; only in {name}: {only_here}, "
                f"only in {reference_name}: {only_there}). Score every model from the same "
                "frozen inputs."
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

MANIFEST_COLUMNS = ("record_id", "url", "claim_sha256", "verdict")


def default_manifest_path() -> Path:
    """``configs/factcheck_eval_manifest.csv``. Absent until someone freezes the set."""
    from modeling.config import CONFIG_DIR

    return CONFIG_DIR / "factcheck_eval_manifest.csv"


def claim_hash(claim: str) -> str:
    """SHA-256 of the claim text exactly as stored. The manifest keeps this, not the text."""
    return hashlib.sha256(claim.encode("utf-8")).hexdigest()


def _verdict(row: dict[str, Any]) -> str:
    """The claim's verdict label as the manifest stores it: stripped and lower case."""
    return str(row.get("label") or "").strip().lower()


def _is_false(row: dict[str, Any]) -> bool:
    return _verdict(row) == "false"


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
                "verdict": _verdict(item),
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
        header = tuple(reader.fieldnames or ())
        if header == MANIFEST_COLUMNS[:-1]:
            raise ManifestMismatch(
                f"{path}: this manifest has no verdict column, so a claim whose label changed "
                "would drop out of the evaluation set unnoticed. Rebuild it on purpose and "
                "record why."
            )
        if header != MANIFEST_COLUMNS:
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
        if not (r["verdict"] or "").strip():
            raise ManifestMismatch(f"{path}: record {r['record_id']!r} has no verdict")
    return rows


def verify_manifest(
    manifest: list[dict[str, str]],
    kept_rows: list[dict[str, Any]],
    collapsed: list[tuple[str, str, int]],
) -> None:
    """Stop unless every manifest row is present and unchanged in the current rows.

    Checks, for each manifest row: the record exists as a kept row (a row that is
    now a collapsed duplicate counts as missing, because the set it was frozen in
    has changed), the URL is the same, the claim hash is the same, and the verdict
    is the same. Raises
    ``ManifestMismatch`` listing every problem category with examples.
    """
    kept = {str(r["record_id"]): r for r in kept_rows}
    collapsed_into = {record_id: rep for record_id, rep, _ in collapsed}
    missing: list[str] = []
    now_duplicate: list[str] = []
    hash_changed: list[str] = []
    url_changed: list[str] = []
    verdict_changed: list[str] = []
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
        if _verdict(row) != entry["verdict"]:
            verdict_changed.append(f"{record_id} ({entry['verdict']!r} -> {_verdict(row)!r})")

    problems: list[str] = []
    for label, items in (
        ("missing from the fact-check rows", missing),
        ("no longer kept after claim deduplication", now_duplicate),
        ("claim hash differs", hash_changed),
        ("URL differs", url_changed),
        ("verdict differs", verdict_changed),
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


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
DEFAULT_BASELINE = "tfidf"
#: Models that are a reference point and never a candidate. The zero-shot base
#: checkpoint is measured so that the fine-tuned models have something to beat,
#: and a number that clears the ship rule for it must not read as a model to ship.
CONTROL_MODELS = frozenset({"laya_multi_base"})
CONTROL_REASON = (
    "control model, not a candidate: its numbers are a baseline for the fine-tuned models "
    "and it is never shipped"
)
DEFAULT_PRECISION_TARGET = 0.85
#: Resamples for every interval in this module.
DEFAULT_N_BOOTSTRAP = 2000
ALPHA = 0.05
#: Hindi scores enter fusion only with this many Hindi claims ...
HINDI_MIN_CLAIMS = 50
#: ... and a lift interval whose lower limit is above this.
HINDI_LIFT_FLOOR = 1.0

SHIP_RULE = (
    "A transformer ships only if the lower limit of the 95% paired-bootstrap interval of "
    "(its macro-F1 minus the baseline macro-F1) is above 0."
)
EVALUATION_SET_STATEMENT = (
    "The fact-check set contains only the claims that publishers chose to debunk. It is not a "
    "sample of false claims in general, and every claim in it is false. Recall on it measures "
    "detection of publicly debunked claims and is never reported without the flag rate on the "
    "unlabelled corpus."
)
#: Shown beside every corpus_text flag rate and lift. It states a difference in what
#: the models read, and it changes no number.
CORPUS_TEXT_NOTICE = (
    "Full-text flag rates and lifts are not like for like across model families. Laya "
    "scores long text with predict_long, which takes the maximum over windows of the whole "
    "text. The roberta classifier truncates the text at 256 tokens. A model that reads more "
    "of a long article has more chances to cross its threshold. Compare models on "
    "corpus_title, the primary figure, and read corpus_text as a second value."
)
NOT_FROZEN_NOTICE = (
    "WARNING: evaluation set not frozen. No manifest exists, so the claim set is whatever "
    "factchecks.parquet holds today and can change between runs. Freeze it with "
    "`python -m modeling.cli freeze-factchecks` before citing any number from this report."
)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from modeling.eval.metrics import macro_f1_binary

    return macro_f1_binary(y_true, y_pred)


def _interval_dict(interval: Any) -> dict[str, Any]:
    """An ``Interval`` at full precision, with NaN written as ``None``."""
    out = {
        "value": _finite_or_none(interval.value),
        "ci_low": _finite_or_none(interval.low),
        "ci_high": _finite_or_none(interval.high),
        "n_bootstrap": int(interval.n_bootstrap),
    }
    for extra in ("n_paired", "n_single_class_resamples"):
        if hasattr(interval, extra):
            out[extra] = int(getattr(interval, extra))
    return out


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _skip_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Null-score rows counted per ``skip_reason``, in a stable order."""
    skipped = frame[frame["score"].isna()]
    counts = skipped["skip_reason"].fillna(UNSPECIFIED_SKIP).value_counts()
    return {str(reason): int(count) for reason, count in sorted(counts.items())}


def _describe_skips(skips: dict[str, int]) -> str:
    return ", ".join(f"{reason}={count}" for reason, count in skips.items()) or "none"


def operating_point(scores: ModelScores, precision_target: float) -> dict[str, Any]:
    """The model's threshold from its own validation rows at a precision target.

    Null-score validation rows are left out and counted. When the scored rows
    hold only one class there is no precision-recall curve, and the model gets no
    operating point rather than a made-up one.
    """
    from modeling.eval.metrics import threshold_at_precision

    val = scores.part("val")
    scored = val[val["score"].notna()]
    out: dict[str, Any] = {
        "precision_target": precision_target,
        "n_val_rows": int(len(val)),
        "n_val_scored": int(len(scored)),
        "n_val_skipped": int(len(val) - len(scored)),
        "skips": _skip_counts(val),
    }
    y = scored["label"].astype(int).to_numpy()
    if not len(scored):
        return {**out, "status": "no_operating_point", "reason": "no scored validation rows"}
    if y.min() == y.max():
        return {
            **out,
            "status": "no_operating_point",
            "reason": f"the scored validation rows hold only class {int(y[0])}",
        }
    chosen = threshold_at_precision(y, scored["score"].to_numpy(), precision_target)
    return {
        **out,
        "status": "ok",
        "threshold": float(chosen["threshold"]),
        "val_precision": float(chosen["precision"]),
        "val_recall": float(chosen["recall"]),
        "target_met": bool(chosen["target_met"]),
    }


def _test_summary(
    scores: ModelScores, op: dict[str, Any], n_bootstrap: int, seed: int
) -> dict[str, Any]:
    from modeling.eval.metrics import bootstrap_metric

    test = scores.part("test")
    scored = test[test["score"].notna()].sort_values("id")
    out: dict[str, Any] = {
        "n_test_rows": int(len(test)),
        "n_scored": int(len(scored)),
        "n_skipped": int(len(test) - len(scored)),
        "skips": _skip_counts(test),
    }
    if op["status"] != "ok":
        return {**out, "macro_f1": None, "reason": f"no operating point: {op['reason']}"}
    if not len(scored):
        return {**out, "macro_f1": None, "reason": "no scored test rows"}
    y = scored["label"].astype(int).to_numpy()
    pred = (scored["score"].to_numpy() >= op["threshold"]).astype(int)
    interval = bootstrap_metric(y, pred, macro_f1, n_bootstrap=n_bootstrap, seed=seed)
    out["macro_f1"] = _interval_dict(interval)
    out["n_flagged"] = int(pred.sum())
    if interval.n_bootstrap == 0 or not np.isfinite(interval.low):
        out["reason"] = "no interval: fewer than 10 scored test rows or too few usable resamples"
    return out


def paired_comparison(
    candidate: ModelScores,
    baseline: ModelScores,
    candidate_op: dict[str, Any],
    baseline_op: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Paired macro-F1 difference of a candidate against the baseline, with the ship decision.

    Uses the test ids scored by *both* models. A row one model could not score is
    not a row where it was wrong, and is left out of the pair rather than
    counted as a miss. The number left out is reported.
    """
    from modeling.eval.metrics import paired_bootstrap_difference

    out: dict[str, Any] = {
        "model": candidate.name,
        "baseline": baseline.name,
        "rule": SHIP_RULE,
        "difference_definition": f"macro-F1({candidate.name}) - macro-F1({baseline.name})",
    }
    for op, name in ((candidate_op, candidate.name), (baseline_op, baseline.name)):
        if op["status"] != "ok":
            return {
                **out,
                "ship": False,
                "difference": None,
                "reason": f"{name} has no operating point: {op['reason']}",
            }

    def scored_test(scores: ModelScores) -> pd.DataFrame:
        test = scores.part("test")
        return test[test["score"].notna()].set_index("id")

    cand_rows, base_rows = scored_test(candidate), scored_test(baseline)
    common = sorted(set(cand_rows.index) & set(base_rows.index))
    n_test = len(candidate.part("test"))
    out["n_test_rows"] = int(n_test)
    out["n_paired"] = len(common)
    out["n_left_out"] = int(n_test - len(common))
    if not common:
        return {**out, "ship": False, "difference": None, "reason": "no test row scored by both"}

    y = cand_rows.loc[common, "label"].astype(int).to_numpy()
    pred_c = (cand_rows.loc[common, "score"].to_numpy() >= candidate_op["threshold"]).astype(int)
    pred_b = (base_rows.loc[common, "score"].to_numpy() >= baseline_op["threshold"]).astype(int)
    interval = paired_bootstrap_difference(
        y, pred_c, pred_b, macro_f1, n_bootstrap=n_bootstrap, seed=seed
    )
    out["difference"] = _interval_dict(interval)
    if not np.isfinite(interval.low):
        return {
            **out,
            "ship": False,
            "reason": f"no interval: {len(common)} paired rows is fewer than 10",
        }
    ship = bool(interval.low > 0)
    return {
        **out,
        "ship": ship,
        "reason": (
            f"lower limit {interval.low:.4f} is above 0"
            if ship
            else f"lower limit {interval.low:.4f} is not above 0"
        ),
    }


def percentile_with_undefined(finite: np.ndarray, n_undefined: int, q: float) -> float:
    """Percentile ``q`` of a sample in which ``n_undefined`` values are +infinity.

    Same interpolation as ``numpy.percentile`` (linear). The undefined values sort
    above every finite one. The result is ``inf`` when the percentile falls on or
    between an undefined value, which says the limit cannot be bounded. It is
    never a made-up finite number, and it is equal to ``numpy.percentile`` when
    ``n_undefined`` is 0.
    """
    ordered = np.sort(np.asarray(finite, dtype=float))
    total = len(ordered) + n_undefined
    if total == 0:
        return float("nan")
    position = q / 100 * (total - 1)
    low_rank = int(np.floor(position))
    high_rank = int(np.ceil(position))
    fraction = position - low_rank

    def at(rank: int) -> float:
        return float(ordered[rank]) if rank < len(ordered) else float("inf")

    low_value, high_value = at(low_rank), at(high_rank)
    if fraction == 0 or low_value == high_value:
        return low_value
    if not np.isfinite(high_value):
        return float("inf")
    return low_value + (high_value - low_value) * fraction


def _bootstrap_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent generators for the claims resample and the corpus resample."""
    claims, corpus = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(claims), np.random.default_rng(corpus)


def _count_draws(k: int, n: int, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    """Flag counts of ``n_bootstrap`` bootstrap resamples of ``n`` rows with ``k`` flagged.

    Resampling ``n`` rows with replacement and counting the flagged ones is a
    Binomial(n, k/n) draw. That is the same distribution as resampling the rows
    one by one, and it costs one number per resample instead of ``n``, which
    matters for a corpus of millions of rows.
    """
    return rng.binomial(n, k / n, size=n_bootstrap)


def _rate_estimate(k: int, n: int, *, n_bootstrap: int, seed: int, stream: int) -> dict[str, Any]:
    """A rate with its bootstrap interval. ``stream`` 0 is claims, 1 is corpus.

    Uses the same generators as ``lift_estimate``, so the recall and flag-rate
    intervals printed beside a lift are the very draws that lift was built from.
    """
    rng = _bootstrap_streams(seed)[stream]
    draws = _count_draws(k, n, n_bootstrap, rng) / n
    low, high = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {
        "status": "ok",
        "value": k / n,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_bootstrap": n_bootstrap,
        "reason": None,
    }


def lift_estimate(
    k_claims: int,
    n_claims: int,
    k_corpus: int,
    n_corpus: int,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int,
) -> dict[str, Any]:
    """Lift = recall / flag rate, with a bootstrap interval from independent resamples.

    The claims and the corpus are resampled independently, ``n_bootstrap`` times.
    A resample with no flagged corpus row has an undefined lift (a division by
    zero). Such resamples are counted and reported, and they are not dropped from
    the interval: they rank above every finite lift, so when more than 2.5% of
    resamples are undefined the upper limit itself is undefined and is reported
    as "no value" with the reason. Dropping them would shrink the interval exactly
    when the flag rate is too small to measure.

    When the flag rate of the data is 0, the lift is undefined: no value, no
    interval, a reason, and never infinity.
    """
    recall = k_claims / n_claims
    flag_rate = k_corpus / n_corpus
    out: dict[str, Any] = {
        "recall": recall,
        "flag_rate": flag_rate,
        "n_claims": n_claims,
        "n_corpus": n_corpus,
        "n_bootstrap": n_bootstrap,
    }
    if k_corpus == 0:
        return {
            **out,
            "status": "undefined",
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "n_zero_flag_resamples": n_bootstrap,
            "reason": (
                f"flag rate is 0 at the validation threshold (0 of {n_corpus} corpus rows "
                "flagged), so the ratio is undefined"
            ),
        }

    rng_claims, rng_corpus = _bootstrap_streams(seed)
    recall_draws = _count_draws(k_claims, n_claims, n_bootstrap, rng_claims) / n_claims
    corpus_counts = _count_draws(k_corpus, n_corpus, n_bootstrap, rng_corpus)
    zero = corpus_counts == 0
    n_zero = int(zero.sum())
    lifts = recall_draws[~zero] / (corpus_counts[~zero] / n_corpus)

    low = percentile_with_undefined(lifts, n_zero, 100 * ALPHA / 2)
    high = percentile_with_undefined(lifts, n_zero, 100 * (1 - ALPHA / 2))
    reasons: list[str] = []
    if not np.isfinite(low) or not np.isfinite(high):
        reasons.append(
            f"{n_zero} of {n_bootstrap} resamples had flag rate 0, so the "
            + ("upper limit" if not np.isfinite(high) and np.isfinite(low) else "interval")
            + " cannot be bounded"
        )
    elif n_zero:
        reasons.append(
            f"{n_zero} of {n_bootstrap} resamples had flag rate 0; they are counted in the "
            "percentiles as undefined and are not dropped"
        )
    return {
        **out,
        "status": "ok",
        "value": recall / flag_rate,
        "ci_low": _finite_or_none(low),
        "ci_high": _finite_or_none(high),
        "n_zero_flag_resamples": n_zero,
        "reason": "; ".join(reasons) or None,
    }


# ---------------------------------------------------------------------------
# Claims, corpus and the cells of the report
# ---------------------------------------------------------------------------
def load_factcheck_publishers(path: Path | None = None) -> frozenset[str]:
    """Publisher domains from ``configs/factcheck.yaml``. An empty list refuses.

    With no publishers to exclude, the flag rate would include the articles the
    claims came from, which measures the model on its own answer key.
    """
    import yaml

    from modeling.config import CONFIG_DIR

    path = Path(path) if path else CONFIG_DIR / "factcheck.yaml"
    if not path.exists():
        raise EvalInputError(f"{path} not found; cannot tell which corpus rows are fact-checkers")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    domains = frozenset(_normalise_domain(d) for d in (data.get("publishers") or {}))
    if not domains:
        raise EvalInputError(f"{path} lists no publishers; refusing to compute a flag rate")
    return domains


def _normalise_domain(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text[4:] if text.startswith("www.") else text


def _split_claims(
    scores: ModelScores, universe: ClaimUniverse
) -> tuple[pd.DataFrame, dict[str, int], int]:
    """Claim rows that count towards recall, the exclusion counts, and kept ids with no row.

    A claim id that is not an evaluation claim is excluded under a reason code
    (collapsed duplicate, not labelled false, outside the frozen set, or unknown).
    The input part may or may not be limited already, so none of this is assumed.
    """
    claims = scores.part("claims")
    reasons = claims["id"].map(universe.reason_for_exclusion)
    kept = claims[reasons.isna()]
    counts = reasons.dropna().value_counts()
    excluded = {str(reason): int(count) for reason, count in sorted(counts.items())}
    absent = len(universe.eval_ids - set(claims["id"]))
    return kept, excluded, absent


def _split_corpus(
    scores: ModelScores, part: str, publishers: frozenset[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Corpus rows without fact-check publishers, and what was removed or unclear."""
    frame = scores.part(part)
    domains = frame["source_detail"].map(_normalise_domain)
    is_publisher = domains.isin(publishers)
    kept = frame[~is_publisher]
    info = {
        "n_rows_in_file": int(len(frame)),
        "n_excluded_fact_check_publisher": int(is_publisher.sum()),
        "n_without_source_detail": int((domains == "").sum()),
    }
    return kept, info


def _language_subset(frame: pd.DataFrame, language: str) -> pd.DataFrame:
    if language == ALL:
        return frame
    return frame[frame["lang"] == language]


def _counts(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    scored = frame[frame["score"].notna()]
    return {
        "n_rows": int(len(frame)),
        "n_scored": int(len(scored)),
        "n_skipped": int(len(frame) - len(scored)),
        "skips": _skip_counts(frame),
        "n_flagged": int((scored["score"].to_numpy() >= threshold).sum()),
    }


def _no_value(reason: str, status: str = "no_score") -> dict[str, Any]:
    return {"status": status, "value": None, "ci_low": None, "ci_high": None, "reason": reason}


def _no_score_reason(counts: dict[str, Any], what: str) -> str:
    if counts["n_rows"] == 0:
        return f"no {what} rows in this cell"
    return (
        f"no valid score for any of {counts['n_rows']} {what} rows "
        f"(skip_reason: {_describe_skips(counts['skips'])})"
    )


def _cell(
    claims: pd.DataFrame,
    corpus: dict[str, pd.DataFrame],
    language: str,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    claim_counts = _counts(_language_subset(claims, language), threshold)
    cell: dict[str, Any] = {"claims": claim_counts}
    if claim_counts["n_scored"]:
        claim_counts["recall"] = _rate_estimate(
            claim_counts["n_flagged"],
            claim_counts["n_scored"],
            n_bootstrap=n_bootstrap,
            seed=seed,
            stream=0,
        )
    else:
        claim_counts["recall"] = _no_value(_no_score_reason(claim_counts, "claim"))

    for part, key, label in (
        ("corpus_title", "title", "corpus title"),
        ("corpus_text", "text", "corpus text"),
    ):
        counts = _counts(_language_subset(corpus[part], language), threshold)
        cell[f"corpus_{key}"] = counts
        if counts["n_scored"]:
            counts["flag_rate"] = _rate_estimate(
                counts["n_flagged"],
                counts["n_scored"],
                n_bootstrap=n_bootstrap,
                seed=seed,
                stream=1,
            )
        else:
            counts["flag_rate"] = _no_value(_no_score_reason(counts, label))

        if claim_counts["n_scored"] and counts["n_scored"]:
            cell[f"lift_{key}"] = lift_estimate(
                claim_counts["n_flagged"],
                claim_counts["n_scored"],
                counts["n_flagged"],
                counts["n_scored"],
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        else:
            reasons = []
            if not claim_counts["n_scored"]:
                reasons.append("recall: " + claim_counts["recall"]["reason"])
            if not counts["n_scored"]:
                reasons.append("flag rate: " + counts["flag_rate"]["reason"])
            cell[f"lift_{key}"] = _no_value("; ".join(reasons))
    return cell


def _unavailable_cell(reason: str) -> dict[str, Any]:
    entry = _no_value(reason)
    return {
        "claims": {"recall": entry},
        "corpus_title": {"flag_rate": entry},
        "corpus_text": {"flag_rate": entry},
        "lift_title": entry,
        "lift_text": entry,
    }


def hindi_gate(model: str, cell: dict[str, Any] | None) -> dict[str, Any]:
    """Hindi scores enter fusion only with enough claims and a lift that clears 1.

    Needs at least ``HINDI_MIN_CLAIMS`` Hindi claims *with a valid score* and a
    lower interval limit above ``HINDI_LIFT_FLOOR`` on the primary (title) lift.
    Both numbers are returned, so a failure shows which condition failed.
    """
    out: dict[str, Any] = {
        "model": model,
        "min_claims": HINDI_MIN_CLAIMS,
        "lift_floor": HINDI_LIFT_FLOOR,
        "lift_basis": "corpus_title",
    }
    claims = (cell or {}).get("claims", {})
    n = int(claims.get("n_scored", 0))
    lift = (cell or {}).get("lift_title") or {}
    low = lift.get("ci_low")
    out.update(
        n_hindi_claims_scored=n,
        n_hindi_claims_rows=int(claims.get("n_rows", 0)),
        lift=lift.get("value"),
        lift_ci_low=low,
        lift_ci_high=lift.get("ci_high"),
        claims_condition_met=n >= HINDI_MIN_CLAIMS,
        lift_condition_met=bool(low is not None and low > HINDI_LIFT_FLOOR),
    )
    out["pass"] = bool(out["claims_condition_met"] and out["lift_condition_met"])
    failures = []
    if not out["claims_condition_met"]:
        failures.append(f"{n} Hindi claims with a score, {HINDI_MIN_CLAIMS} needed")
    if not out["lift_condition_met"]:
        failures.append(
            "no lift interval: " + str(lift.get("reason") or "not computed")
            if low is None
            else f"lift lower limit {low:.3f} is not above {HINDI_LIFT_FLOOR:g}"
        )
    out["reason"] = "; ".join(failures) or (
        f"{n} Hindi claims and lift lower limit {low:.3f} above {HINDI_LIFT_FLOOR:g}"
    )
    return out


def apply_control(
    comparison: dict[str, Any] | None = None, gate: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Force a control model's decision to "do not ship" or "FAIL (control)".

    The numbers stay in the record. ``by_numbers`` keeps what the rule would have
    decided without the control marker, so the baseline remains readable.
    """
    if comparison is not None:
        return {
            **comparison,
            "control": True,
            "ship_by_numbers": comparison["ship"],
            "ship": False,
            "reason": f"{CONTROL_REASON}. By the numbers alone: {comparison['reason']}",
        }
    assert gate is not None
    return {
        **gate,
        "control": True,
        "pass_by_numbers": gate["pass"],
        "pass": False,
        "reason": f"{CONTROL_REASON}. By the numbers alone: {gate['reason']}",
    }


def gate_text(gate: dict[str, Any]) -> str:
    """``PASS``, ``FAIL`` or ``FAIL (control)``."""
    if gate.get("control"):
        return "FAIL (control)"
    return "PASS" if gate["pass"] else "FAIL"


def check_claim_languages(models: dict[str, ModelScores]) -> None:
    """Refuse when two models give one claim id different languages.

    Per-language recall would then be computed on different claims per model.
    """
    seen: dict[str, tuple[str, Any]] = {}
    for name, scores in models.items():
        for claim_id, lang in zip(
            scores.part("claims")["id"], scores.part("claims")["lang"], strict=True
        ):
            lang = None if pd.isna(lang) else lang
            if claim_id in seen and seen[claim_id][1] != lang:
                raise SplitMismatch(
                    f"claim {claim_id!r} has lang {seen[claim_id][1]!r} in model "
                    f"{seen[claim_id][0]!r} and {lang!r} in model {name!r}"
                )
            seen.setdefault(claim_id, (name, lang))


def evaluate(
    models: dict[str, ModelScores],
    *,
    universe: ClaimUniverse,
    publishers: frozenset[str],
    baseline: str = DEFAULT_BASELINE,
    precision_target: float = DEFAULT_PRECISION_TARGET,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int | None = None,
    versions: dict[str, str] | None = None,
    split_description: str | None = None,
    controls: Collection[str] | None = None,
) -> dict[str, Any]:
    """Run the whole comparison and return a JSON-ready result.

    Raises ``EvalInputError`` (or a subclass) for anything it must not work
    around: a missing baseline, test splits that differ, no publishers to exclude.
    Output depends only on the inputs and ``seed``, never on row order.
    """
    from modeling.config import get_settings

    seed = get_settings().seed if seed is None else seed
    if baseline not in models:
        raise EvalInputError(
            f"baseline model {baseline!r} has no score file; found {sorted(models)}"
        )
    if not publishers:
        raise EvalInputError("no fact-check publishers given; refusing to compute a flag rate")
    controls = CONTROL_MODELS if controls is None else frozenset(controls)
    if baseline in controls:
        raise EvalInputError(f"baseline model {baseline!r} is marked as a control model")
    check_shared_test_split(models)
    check_shared_inputs(models)
    check_claim_languages(models)
    versions = versions or {}
    ordered = sorted(models)

    notices: list[str] = []
    if not universe.frozen:
        notices.append(NOT_FROZEN_NOTICE)
    for name in sorted(set(models) & controls):
        notices.append(f"{name}: {CONTROL_REASON}")
    for name, scores in models.items():
        if scores.ignored_columns:
            notices.append(f"{name}: ignored unknown column(s) {scores.ignored_columns}")

    ops = {name: operating_point(models[name], precision_target) for name in ordered}
    for name, op in ops.items():
        if op["status"] != "ok":
            notices.append(f"{name}: no operating point ({op['reason']}); every cell is no score")
        elif not op["target_met"]:
            notices.append(
                f"{name}: no validation threshold reaches precision {precision_target}; the "
                f"best available is {op['val_precision']:.3f}. The flags below use that threshold."
            )

    model_out: dict[str, Any] = {}
    cells: dict[str, dict[str, Any]] = {}
    for name in ordered:
        scores, op = models[name], ops[name]
        claims, excluded, absent = _split_claims(scores, universe)
        corpus: dict[str, pd.DataFrame] = {}
        corpus_info: dict[str, Any] = {}
        for part in CORPUS_PARTS:
            corpus[part], corpus_info[part] = _split_corpus(scores, part, publishers)
        if absent:
            notices.append(
                f"{name}: {absent} evaluation claim(s) have no row in the score file; they are "
                "not in any denominator"
            )
        if any(info["n_without_source_detail"] for info in corpus_info.values()):
            notices.append(
                f"{name}: corpus rows with no source_detail cannot be checked against the "
                "fact-check publisher list and stay in the flag rate"
            )

        model_out[name] = {
            "version": versions.get(name),
            "control": name in controls,
            "input": {
                "file": scores.path.name if scores.path else None,
                "sha256": scores.sha256,
                "n_rows": int(len(scores.frame)),
            },
            "operating_point": op,
            "test": _test_summary(scores, op, n_bootstrap, seed),
            "claims_excluded": excluded,
            "evaluation_claims_missing_from_scores": absent,
            "corpus": corpus_info,
        }
        if op["status"] == "ok":
            cells[name] = {
                lang: _cell(claims, corpus, lang, op["threshold"], n_bootstrap, seed)
                for lang in (*LANGUAGES, ALL)
            }
        else:
            reason = f"no operating point: {op['reason']}"
            cells[name] = {lang: _unavailable_cell(reason) for lang in (*LANGUAGES, ALL)}

    comparisons = {
        name: paired_comparison(
            models[name],
            models[baseline],
            ops[name],
            ops[baseline],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for name in ordered
        if name != baseline
    }
    comparisons = {
        name: apply_control(comparison) if name in controls else comparison
        for name, comparison in comparisons.items()
    }
    for name, comparison in comparisons.items():
        if comparison.get("n_left_out"):
            notices.append(
                f"{name}: {comparison['n_left_out']} test row(s) were not scored by both {name} "
                f"and {baseline}; the paired comparison uses the {comparison['n_paired']} rows "
                "both scored"
            )

    hindi_gates = {}
    for name in ordered:
        gate = hindi_gate(name, cells[name].get("hi"))
        hindi_gates[name] = apply_control(gate=gate) if name in controls else gate

    n_val = {name: ops[name]["n_val_rows"] for name in ordered}
    n_test = len(models[baseline].part("test"))
    return {
        "schema_version": 1,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "precision_target": precision_target,
        "baseline": baseline,
        "languages": [*LANGUAGES, ALL],
        "split": {
            "description": split_description
            or (
                "val and test rows are used as supplied in each score file. This harness does "
                "not re-split and cannot verify that the producer split by group (author, source "
                "video, speaker)."
            ),
            "n_test_shared": int(n_test),
            "n_val_per_model": n_val,
        },
        "evaluation_set": {
            "frozen": universe.frozen,
            "manifest": str(universe.manifest_path) if universe.manifest_path else None,
            "manifest_rows": universe.manifest_rows,
            "n_evaluation_claims": len(universe.eval_ids),
            "dedup_counts": universe.dedup_counts,
            "statement": EVALUATION_SET_STATEMENT,
            "corpus_text_notice": CORPUS_TEXT_NOTICE,
        },
        "notices": notices,
        "models": model_out,
        "comparisons": comparisons,
        "cells": cells,
        "controls": sorted(set(ordered) & controls),
        "hindi_gate": hindi_gates,
    }


# ---------------------------------------------------------------------------
# Output: artifacts/eval/factcheck/<label>/{factcheck_eval.json,factcheck_eval.md}
# ---------------------------------------------------------------------------
EVAL_MODULE = "factcheck"
JSON_NAME = "factcheck_eval.json"
MARKDOWN_NAME = "factcheck_eval.md"


def _cell_text(estimate: dict[str, Any] | None, digits: int = 3) -> str:
    """One table cell: ``value [low, high]``, or the status with its reason."""
    if not estimate:
        return "n/a"
    if estimate["status"] != "ok":
        label = "no score" if estimate["status"] == "no_score" else "undefined"
        return f"{label}: {estimate['reason']}"

    def fmt(value: float | None) -> str:
        return "no value" if value is None else f"{value:.{digits}f}"

    text = f"{fmt(estimate['value'])} [{fmt(estimate['ci_low'])}, {fmt(estimate['ci_high'])}]"
    if estimate.get("reason"):
        text += f" ({estimate['reason']})"
    return text


def _escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _interval_text(interval: dict[str, Any] | None) -> str:
    if not interval or interval.get("value") is None:
        return "no value"
    low, high = interval.get("ci_low"), interval.get("ci_high")
    if low is None or high is None:
        return f"{interval['value']:.3f} [no interval]"
    return f"{interval['value']:.3f} [{low:.3f}, {high:.3f}]"


def render_markdown(result: dict[str, Any], fingerprint: dict[str, Any] | None = None) -> str:
    """The report table set. Every number shown has its sample size beside it."""
    lines: list[str] = ["# Fact-check evaluation: three-way comparison", ""]
    evaluation_set = result["evaluation_set"]
    if not evaluation_set["frozen"]:
        lines += [f"> **{NOT_FROZEN_NOTICE}**", ""]
    lines += [f"> {evaluation_set['statement']}", ""]
    lines += [
        f"- Baseline: `{result['baseline']}`. Precision target: {result['precision_target']}. "
        f"Seed: `{result['seed']}`. Resamples: {result['n_bootstrap']}. Intervals are 95% "
        "percentile bootstrap intervals.",
        f"- Split: {result['split']['description']}",
        f"- Evaluation claims: {evaluation_set['n_evaluation_claims']}"
        + (
            f" (frozen, {evaluation_set['manifest_rows']} manifest rows)"
            if evaluation_set["frozen"]
            else " (not frozen)"
        ),
        "",
    ]

    notices = [n for n in result["notices"] if n != NOT_FROZEN_NOTICE]
    if notices:
        lines += ["## Notices", ""] + [f"- {_escape(n)}" for n in notices] + [""]

    lines += [
        "## Models and operating points",
        "",
        "The threshold of each model comes from its own validation rows. Nothing is shared.",
        "",
        "| model | version | threshold | val precision | val recall | target met | "
        "val scored/rows | test scored/rows |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, info in result["models"].items():
        op, test = info["operating_point"], info["test"]
        if op["status"] == "ok":
            op_cells = (
                f"{op['threshold']:.4f} | {op['val_precision']:.3f} | {op['val_recall']:.3f} | "
                f"{'yes' if op['target_met'] else '**no**'}"
            )
        else:
            op_cells = f"no operating point: {_escape(op['reason'])} | n/a | n/a | n/a"
        lines.append(
            f"| {name}{' (control)' if info.get('control') else ''} | "
            f"{info['version'] or 'not supplied'} | {op_cells} | "
            f"{op['n_val_scored']}/{op['n_val_rows']} | {test['n_scored']}/{test['n_test_rows']} |"
        )
    lines.append("")

    lines += [
        "## Shared test split: macro-F1 at each model's own threshold",
        "",
        "| model | macro-F1 [95% CI] | scored/rows | flagged | skips |",
        "|---|---|---|---|---|",
    ]
    for name, info in result["models"].items():
        test = info["test"]
        if test.get("macro_f1"):
            f1 = _interval_text(test["macro_f1"])
        else:
            f1 = f"no score: {_escape(test.get('reason'))}"
        lines.append(
            f"| {name} | {f1} | {test['n_scored']}/{test['n_test_rows']} | "
            f"{test.get('n_flagged', 'n/a')} | {_describe_skips(test['skips'])} |"
        )
    lines += [""]

    lines += [
        "### Ship decision against the baseline",
        "",
        SHIP_RULE,
        "",
        "| model | macro-F1 difference [95% CI] | paired rows | left out | decision | reason |",
        "|---|---|---|---|---|---|",
    ]
    for name, comparison in result["comparisons"].items():
        lines.append(
            f"| {name} | {_interval_text(comparison.get('difference'))} | "
            f"{comparison.get('n_paired', 'n/a')} | {comparison.get('n_left_out', 'n/a')} | "
            f"**{'ship' if comparison['ship'] else 'do not ship'}"
            f"{' (control)' if comparison.get('control') else ''}** | "
            f"{_escape(comparison['reason'])} |"
        )
    lines += [
        "",
        "The difference is the candidate minus the baseline, on test rows scored by both, with "
        "the same resampled row indices for both models.",
        "",
    ]

    for key, heading, corpus_key in (
        ("title", "Recall, flag rate and lift: corpus titles (primary)", "corpus_title"),
        ("text", "Recall, flag rate and lift: corpus full text (second value)", "corpus_text"),
    ):
        lines += [
            f"## {heading}",
            "",
            "Lift = recall on known-false claims / flag rate on the unlabelled corpus, at the "
            "same threshold. Fact-check publishers are excluded from the corpus.",
            "",
        ]
        if key == "text":
            lines += [f"> **Not like for like.** {CORPUS_TEXT_NOTICE}", ""]
        lines += [
            "| model | language | claims scored/rows | recall [95% CI] | corpus scored/rows | "
            "flag rate [95% CI] | lift [95% CI] | zero-flag resamples |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, by_language in result["cells"].items():
            for language in result["languages"]:
                cell = by_language[language]
                claims, corpus = cell["claims"], cell[corpus_key]
                lift = cell[f"lift_{key}"]
                zero = lift.get("n_zero_flag_resamples")
                mark = " (not like for like)" if key == "text" else ""
                lines.append(
                    f"| {name} | {language} | "
                    f"{claims.get('n_scored', 0)}/{claims.get('n_rows', 0)} | "
                    f"{_escape(_cell_text(claims.get('recall')))} | "
                    f"{corpus.get('n_scored', 0)}/{corpus.get('n_rows', 0)} | "
                    f"{_escape(_cell_text(corpus.get('flag_rate')))}{mark} | "
                    f"{_escape(_cell_text(lift))}{mark} | "
                    f"{zero if zero is not None else 'n/a'} of {result['n_bootstrap']} |"
                )
        lines.append("")

    lines += [
        "## Hindi gate",
        "",
        f"Hindi scores enter fusion only with at least {HINDI_MIN_CLAIMS} Hindi claims and a "
        f"title lift interval whose lower limit is above {HINDI_LIFT_FLOOR:g}.",
        "",
        "| model | Hindi claims scored | lift [95% CI] | result | reason |",
        "|---|---|---|---|---|",
    ]
    for name, gate in result["hindi_gate"].items():
        interval = {
            "value": gate["lift"],
            "ci_low": gate["lift_ci_low"],
            "ci_high": gate["lift_ci_high"],
        }
        lines.append(
            f"| {name} | {gate['n_hindi_claims_scored']} of {gate['n_hindi_claims_rows']} | "
            f"{_interval_text(interval) if gate['lift'] is not None else 'no value'} | "
            f"**{gate_text(gate)}** | {_escape(gate['reason'])} |"
        )
    lines.append("")

    lines += [
        "## Skipped and excluded rows",
        "",
        "A null score is never counted as not flagged. Each one is left out of the "
        "denominator and counted here.",
        "",
        "| model | part | rows | no score | skip_reason counts |",
        "|---|---|---|---|---|",
    ]
    for name, by_language in result["cells"].items():
        cell = by_language[ALL]
        for part, key in (("claims", "claims"), ("corpus_title", "corpus_title"),
                          ("corpus_text", "corpus_text")):
            counts = cell[key]
            if "n_rows" in counts:
                lines.append(
                    f"| {name} | {part} | {counts['n_rows']} | {counts['n_skipped']} | "
                    f"{_describe_skips(counts['skips'])} |"
                )
    lines += [
        "",
        "| model | claim ids excluded (reason=count) | kept claims absent from scores | "
        "corpus rows from fact-check publishers (title/text) | corpus rows without source_detail |",
        "|---|---|---|---|---|",
    ]
    for name, info in result["models"].items():
        title, text = info["corpus"]["corpus_title"], info["corpus"]["corpus_text"]
        lines.append(
            f"| {name} | {_describe_skips(info['claims_excluded'])} | "
            f"{info['evaluation_claims_missing_from_scores']} | "
            f"{title['n_excluded_fact_check_publisher']}/{text['n_excluded_fact_check_publisher']}"
            f" | {title['n_without_source_detail']}/{text['n_without_source_detail']} |"
        )
    lines.append("")

    lines += ["## Inputs", "", "| model | file | rows | sha256 |", "|---|---|---|---|"]
    for name, info in result["models"].items():
        source = info["input"]
        lines.append(f"| {name} | {source['file']} | {source['n_rows']} | `{source['sha256']}` |")
    lines.append("")
    if fingerprint:
        lines += [
            "## Reproducibility",
            "",
            f"- seed: `{fingerprint['seed']}`",
            f"- input manifest hash: `{fingerprint['input_manifest_hash']}`",
            "- regenerate by rerunning `python -m modeling.cli evaluate-factchecks` on the same "
            "score files and the same seed",
            "",
        ]
    return "\n".join(lines)


def write_factcheck_report(
    result: dict[str, Any], *, label: str = "comparison", settings: Any = None
) -> list[Path]:
    """Write the JSON result and the Markdown report under ``artifacts/eval/factcheck/<label>/``."""
    from datetime import datetime, timezone

    from modeling.config import run_fingerprint
    from modeling.eval.report import eval_dir

    target = eval_dir(EVAL_MODULE, label, settings)
    fingerprint = run_fingerprint(result["seed"])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": fingerprint,
        **result,
    }
    json_path = target / JSON_NAME
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown_path = target / MARKDOWN_NAME
    markdown_path.write_text(render_markdown(result, fingerprint), encoding="utf-8")
    return [json_path, markdown_path]


def run_comparison(
    score_dir: Path,
    *,
    data_dir: Path | None = None,
    manifest_path: Path | None = None,
    publishers_path: Path | None = None,
    baseline: str = DEFAULT_BASELINE,
    precision_target: float = DEFAULT_PRECISION_TARGET,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int | None = None,
    versions: dict[str, str] | None = None,
    split_description: str | None = None,
    label: str = "comparison",
    settings: Any = None,
    write: bool = True,
    controls: Collection[str] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Load, verify, evaluate and write. Any refusal raises before a file is written.

    The manifest check runs while the claim set is built, before any statistic,
    so a changed evaluation set stops the run with nothing written.
    """
    from modeling.config import get_settings

    settings = settings or get_settings()
    models = load_score_dir(score_dir)
    universe = load_claim_universe(data_dir or settings.data_dir, manifest_path)
    result = evaluate(
        models,
        universe=universe,
        publishers=load_factcheck_publishers(publishers_path),
        baseline=baseline,
        precision_target=precision_target,
        n_bootstrap=n_bootstrap,
        seed=seed,
        versions=versions,
        split_description=split_description,
        controls=controls,
    )
    paths = write_factcheck_report(result, label=label, settings=settings) if write else []
    return result, paths
