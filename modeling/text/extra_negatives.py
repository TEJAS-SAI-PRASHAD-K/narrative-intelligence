"""Extra negatives for the ``misinfo`` training frame: articles that report on a false claim.

``misinfo`` v0.2.0 reacts to the subject of an article. A news report or a rebuttal names the
false claim, and the model scores it as if it pushed the claim. This module lets a training
run add such articles as label 0 rows.

The input is a Parquet file with the columns ``text`` and ``group_id``, and an optional
``domain``. ``modeling.cli reporting-negatives`` builds it from the corpus on disk.

Two rules keep the benchmark comparison fair:

* The benchmark rows are split first, with the same call and seed as a run without extra
  negatives. Their train, validation and test assignment does not depend on the extra rows.
  A split of the union would move the benchmark rows, because the shuffle sees more groups.
* The extra rows are split on their own, by group, with the same splitter and seed. A group
  sits on one side. The test side of the extra rows never enters the benchmark test set. The
  training run reports it as a separate section.

Every refusal and every dropped row carries a reason code. The codes are in
:class:`ExtraNegativesReason`.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.datasets.splits import group_train_val_test

log = logging.getLogger(__name__)

SOURCE_DATASET = "reporting"
GROUP_PREFIX = "reporting:"
REQUIRED_COLUMNS = ("text", "group_id")
#: A grouped three-way split needs at least this many groups.
MIN_GROUPS = 3


class ExtraNegativesReason:
    """Reason codes of the extra-negatives path."""

    FILE_MISSING = "extra_negatives_file_missing"
    UNREADABLE = "extra_negatives_unreadable"
    MISSING_COLUMNS = "extra_negatives_missing_columns"
    NO_ROWS = "extra_negatives_no_rows"
    EMPTY_TEXT = "extra_negatives_empty_text"
    EMPTY_GROUP = "extra_negatives_empty_group_id"
    TOO_FEW_GROUPS = "extra_negatives_too_few_groups"
    NO_ROWS_LEFT = "extra_negatives_no_rows_left"
    OUTPUT_EXISTS = "output_version_exists"
    BAD_OUTPUT_VERSION = "output_version_invalid"
    WRONG_MODULE = "extra_negatives_wrong_module"
    #: A row dropped because its text occurs twice in the extra file.
    DUPLICATE = "extra_negatives_duplicate_text"
    #: A row dropped because a benchmark row has the same text and label 0.
    DUPLICATE_OF_BENCHMARK = "extra_negatives_duplicate_of_benchmark_text"
    #: A row dropped because a benchmark row has the same text and label 1.
    CONFLICT_WITH_BENCHMARK = "extra_negatives_conflict_with_benchmark_text"


class ExtraNegativesRefused(ValueError):
    """The extra-negatives input or the run options are refused. ``reason`` is the code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass
class ExtraNegatives:
    """The validated extra rows and the facts that the eval report records."""

    frame: pd.DataFrame
    path: Path
    sha256: str
    rows_in_file: int
    groups_in_file: int
    dropped: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """The record that goes into the eval report and the checkpoint metadata."""
        return {
            "file": self.path.name,
            "sha256": self.sha256,
            "rows": self.rows_in_file,
            "groups": self.groups_in_file,
            "rows_used": int(len(self.frame)),
            "groups_used": int(self.frame["group_id"].nunique()) if len(self.frame) else 0,
            "dropped": {k: v for k, v in self.dropped.items() if v},
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_text(values: pd.Series) -> pd.Series:
    """The key for text equality: the same one that ``build_training_frame`` uses."""
    return values.astype(str).str.strip().str.lower()


def load_extra_negatives(path: Path) -> ExtraNegatives:
    """Read and validate the file. Raise :class:`ExtraNegativesRefused` on a bad input."""
    path = Path(path)
    if not path.is_file():
        raise ExtraNegativesRefused(ExtraNegativesReason.FILE_MISSING, f"{path} does not exist")
    try:
        raw = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - any parser error is one refusal
        raise ExtraNegativesRefused(
            ExtraNegativesReason.UNREADABLE, f"{path} is not readable Parquet ({exc})"
        ) from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ExtraNegativesRefused(
            ExtraNegativesReason.MISSING_COLUMNS,
            f"{path.name} lacks column(s) {missing}. It has {list(raw.columns)}.",
        )
    if not len(raw):
        raise ExtraNegativesRefused(ExtraNegativesReason.NO_ROWS, f"{path.name} has no rows")

    text = raw["text"]
    empty_text = text.isna() | text.astype(str).str.strip().eq("")
    if empty_text.any():
        raise ExtraNegativesRefused(
            ExtraNegativesReason.EMPTY_TEXT,
            f"{int(empty_text.sum())} row(s) of {path.name} have empty text",
        )
    group = raw["group_id"]
    empty_group = group.isna() | group.astype(str).str.strip().eq("")
    if empty_group.any():
        raise ExtraNegativesRefused(
            ExtraNegativesReason.EMPTY_GROUP,
            f"{int(empty_group.sum())} row(s) of {path.name} have an empty group_id",
        )

    domain = raw["domain"].astype(str).to_numpy() if "domain" in raw.columns else SOURCE_DATASET
    frame = pd.DataFrame(
        {
            "text": text.astype(str).to_numpy(),
            "label": 0,
            "source_dataset": SOURCE_DATASET,
            "group_id": GROUP_PREFIX + group.astype(str).str.strip().to_numpy(),
            "domain": domain,
        }
    )
    return ExtraNegatives(
        frame=frame,
        path=path,
        sha256=file_sha256(path),
        rows_in_file=int(len(frame)),
        groups_in_file=int(frame["group_id"].nunique()),
    )


def drop_overlaps(extra: ExtraNegatives, benchmark: pd.DataFrame) -> ExtraNegatives:
    """Drop extra rows whose text equals a benchmark text, and count each reason.

    A benchmark text with label 1 and an extra row with label 0 is an annotation conflict.
    The benchmark wins, as it does in ``build_training_frame``. Duplicates inside the extra
    rows are dropped by the splitter, and ``split_extra_negatives`` counts them.
    """
    frame = extra.frame
    key = normalise_text(frame["text"])
    bench_key = normalise_text(benchmark["text"])
    positive = set(bench_key[benchmark["label"].astype(int).to_numpy() == 1])
    negative = set(bench_key[benchmark["label"].astype(int).to_numpy() == 0])
    conflict = key.isin(positive).to_numpy()
    same = key.isin(negative).to_numpy() & ~conflict
    dropped = dict(extra.dropped)
    dropped[ExtraNegativesReason.CONFLICT_WITH_BENCHMARK] = int(conflict.sum())
    dropped[ExtraNegativesReason.DUPLICATE_OF_BENCHMARK] = int(same.sum())
    kept = frame.loc[~(conflict | same)].reset_index(drop=True)
    for reason, count in dropped.items():
        if count:
            log.warning("extra negatives: dropped %d row(s), reason=%s", count, reason)
    return ExtraNegatives(
        frame=kept,
        path=extra.path,
        sha256=extra.sha256,
        rows_in_file=extra.rows_in_file,
        groups_in_file=extra.groups_in_file,
        dropped=dropped,
    )


def split_extra_negatives(
    extra: ExtraNegatives, *, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the extra rows by group with the splitter and seed of the benchmark split.

    Returns the train, validation and test parts. The duplicate count is added to
    ``extra.dropped``.
    """
    if not len(extra.frame):
        raise ExtraNegativesRefused(
            ExtraNegativesReason.NO_ROWS_LEFT,
            f"no row of {extra.path.name} is left after the overlap check",
        )
    n_groups = extra.frame["group_id"].nunique()
    if n_groups < MIN_GROUPS:
        raise ExtraNegativesRefused(
            ExtraNegativesReason.TOO_FEW_GROUPS,
            f"{extra.path.name} has {n_groups} group(s) after the overlap check. "
            f"A grouped split needs at least {MIN_GROUPS}.",
        )
    work, split = group_train_val_test(
        extra.frame, group_col="group_id", label_col="label", seed=seed
    )
    extra.dropped[ExtraNegativesReason.DUPLICATE] = int(
        split.dropped.get("exact_text_duplicate", 0)
    )
    if extra.dropped[ExtraNegativesReason.DUPLICATE]:
        log.warning(
            "extra negatives: dropped %d row(s), reason=%s",
            extra.dropped[ExtraNegativesReason.DUPLICATE],
            ExtraNegativesReason.DUPLICATE,
        )
    extra.frame = work
    return work.iloc[split.train], work.iloc[split.val], work.iloc[split.test]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a rate. ``(nan, nan)`` for ``n == 0``."""
    if n <= 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def held_out_section(scores: np.ndarray, threshold: float, summary: dict[str, Any]) -> str:
    """Report text for the extra rows in the held-out part: the flag rate at the threshold."""
    n = int(len(scores))
    flagged = int((np.asarray(scores) >= threshold).sum())
    lines = [
        "Reporting articles in the held-out groups of the extra-negatives file. "
        "Every row is a true negative, so the flag rate is a false-positive rate. "
        "These rows are not in the benchmark test set above.",
        "",
        f"- extra-negatives file: `{summary['file']}`, sha256 `{summary['sha256']}`",
        f"- rows in file: {summary['rows']}, groups in file: {summary['groups']}",
        f"- rows used: {summary['rows_used']}, groups used: {summary['groups_used']}",
    ]
    if summary.get("dropped"):
        drops = ", ".join(f"{k}={v}" for k, v in summary["dropped"].items())
        lines.append(f"- dropped rows by reason: {drops}")
    if n:
        low, high = wilson_interval(flagged, n)
        lines.append(
            f"- held-out rows: {n}, flagged at {threshold:g}: {flagged} "
            f"({flagged / n:.3f}, 95% Wilson interval {low:.3f} to {high:.3f})"
        )
    else:
        lines.append("- held-out rows: 0, so no flag rate is reported")
    return "\n".join(lines)


def select_reporting_rows(
    records: pd.DataFrame, domains: frozenset[str], *, lang: str = "en"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pick the fact-check articles of the corpus and shape them as extra negatives.

    A record matches when ``normalise_domain(source_detail)`` is in ``domains``. That is the
    test the ``misinfo`` stage uses for ``fact_check_publisher``. The group is the domain.
    Returns the frame with ``text``, ``group_id`` and ``domain``, and the counts of rows
    dropped by reason.
    """
    from modeling.config import normalise_domain

    dropped = {"not_fact_check_domain": 0, "language_mismatch": 0, "empty_text": 0}
    if "source_detail" not in records.columns:
        raise ExtraNegativesRefused(
            "records_have_no_source_detail",
            "the corpus has no `source_detail` column, so no record matches a fact-checker",
        )
    domain = records["source_detail"].map(normalise_domain)
    is_fact_check = domain.isin(domains)
    dropped["not_fact_check_domain"] = int((~is_fact_check).sum())
    lang_ok = records["lang"].astype(str).str.strip().str.lower().eq(lang.lower())
    dropped["language_mismatch"] = int((is_fact_check & ~lang_ok).sum())
    has_text = records["text"].notna() & records["text"].astype(str).str.strip().ne("")
    dropped["empty_text"] = int((is_fact_check & lang_ok & ~has_text).sum())
    keep = is_fact_check & lang_ok & has_text
    frame = pd.DataFrame(
        {
            "text": records.loc[keep, "text"].astype(str).to_numpy(),
            "group_id": domain[keep].to_numpy(),
            "domain": domain[keep].to_numpy(),
        }
    )
    return frame, dropped
