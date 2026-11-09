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
