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

* The calibrator and the early-stopping loss use the benchmark validation rows only. The
  extra rows would shift the base rate of the calibration set. The extra validation rows and
  the held-out rows are for reporting only.

The extra rows are articles of fact-checkers. Live scoring already skips those articles with
the reason ``fact_check_publisher``. A result on the held-out extra rows therefore says little
about other outlets that report on a false claim.

Every refusal and every dropped row carries a reason code. The codes are in
:class:`ExtraNegativesReason`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.datasets.splits import dedupe_near_duplicates, group_train_val_test

log = logging.getLogger(__name__)

SOURCE_DATASET = "reporting"
GROUP_PREFIX = "reporting:"
REQUIRED_COLUMNS = ("text", "group_id")
#: A grouped three-way split needs at least this many groups.
#: Below this, a run is refused and nothing trains. Three sides of a split with fewer
#: publishers leave a side with one publisher, and a rate on one publisher says nothing.
MIN_GROUPS = 6


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
    #: A row dropped because its simhash is near a benchmark row or an earlier extra row.
    NEAR_DUPLICATE = "extra_negatives_near_duplicate_simhash"
    OUTPUT_VERSION_REQUIRED = "output_version_required"
    HELDOUT_MISMATCH = "heldout_ids_mismatch"
    # builder reasons
    CLAIM_RECORD = "reporting_record_has_claim_row"
    FROZEN_CLAIM_TEXT = "reporting_text_contains_frozen_claim"
    LANGUAGE_MISMATCH = "reporting_language_mismatch"
    ROW_EMPTY_TEXT = "reporting_empty_text"
    MANIFEST_MISSING = "frozen_manifest_missing"
    NO_SOURCE_DETAIL = "records_have_no_source_detail"
    FROZEN_CLAIM_UNAVAILABLE = "frozen_claim_text_unavailable"
    #: A frozen claim has no row in factchecks.parquet, so its text and article are unknown.
    FROZEN_CLAIM_MISSING = "frozen_claim_missing_from_factchecks"
    #: The article text is within the configured simhash radius of a frozen claim.
    FROZEN_CLAIM_SIMHASH = "reporting_text_near_frozen_claim_simhash"
    HELDOUT_META_MISSING = "heldout_metadata_missing"


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
    #: Rows and groups on each side of the extra split, set by ``split_extra_negatives``.
    sides: dict[str, dict[str, int]] = field(default_factory=dict)

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
            "sides": {k: dict(v) for k, v in self.sides.items()},
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_text(values: pd.Series) -> pd.Series:
    """The key for text equality: ``build_training_frame``'s key with white space squashed."""
    return values.astype(str).str.lower().str.split().str.join(" ")


def _squash(text: str) -> str:
    """Text for the claim substring test.

    NFKC, lower case, every punctuation character (quote and dash variants included) turned
    into a space, and runs of white space made into one space.
    """
    folded = unicodedata.normalize("NFKC", str(text)).lower()
    spaced = "".join(" " if unicodedata.category(c).startswith("P") else c for c in folded)
    return " ".join(spaced.split())


def text_sha256(values: pd.Series) -> list[str]:
    """The row identity that a held-out list uses: the hash of the normalised text."""
    return [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in normalise_text(values)]


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
    frame["record_id"] = (
        raw["record_id"].astype(str).to_numpy() if "record_id" in raw.columns else ""
    )
    frame["text_sha256"] = text_sha256(frame["text"])
    return ExtraNegatives(
        frame=frame,
        path=path,
        sha256=file_sha256(path),
        rows_in_file=int(len(frame)),
        groups_in_file=int(frame["group_id"].nunique()),
    )


def _log_drops(dropped: dict[str, int]) -> None:
    for reason, count in dropped.items():
        if count:
            log.warning("extra negatives: dropped %d row(s), reason=%s", count, reason)


def drop_overlaps(extra: ExtraNegatives, benchmark: pd.DataFrame) -> ExtraNegatives:
    """Drop extra rows that repeat a benchmark text or an earlier extra row, and count them.

    Three checks run in this order, each under its own reason code:

    1. A benchmark row with label 1 and the same text: an annotation conflict. The benchmark
       wins, as it does in ``build_training_frame``.
    2. A benchmark row with label 0 and the same text, or an earlier extra row with the same
       text: an exact duplicate.
    3. A simhash within 3 bits of a benchmark row or an earlier extra row: a near duplicate.
       The check runs over the union of both sets, with the benchmark rows first, so a
       benchmark row is never the one that goes. Benchmark rows that repeat each other are not
       touched here, because that would change the benchmark split.
    """
    from ingest.normalize import simhash

    frame = extra.frame
    key = normalise_text(frame["text"])
    bench_key = normalise_text(benchmark["text"])
    labels = benchmark["label"].astype(int).to_numpy()
    positive = set(bench_key[labels == 1])
    negative = set(bench_key[labels == 0])
    conflict = key.isin(positive).to_numpy()
    same = key.isin(negative).to_numpy() & ~conflict
    repeat = key.duplicated(keep="first").to_numpy() & ~conflict & ~same
    dropped = dict(extra.dropped)
    dropped[ExtraNegativesReason.CONFLICT_WITH_BENCHMARK] = int(conflict.sum())
    dropped[ExtraNegativesReason.DUPLICATE_OF_BENCHMARK] = int(same.sum())
    dropped[ExtraNegativesReason.DUPLICATE] = int(repeat.sum())
    kept = frame.loc[~(conflict | same | repeat)].reset_index(drop=True)

    from modeling.eval.labelling import simhash_matches

    # An extra row is compared with every benchmark row, so a benchmark row that is itself a
    # near duplicate of an earlier one still counts. Extra rows are then compared with each
    # other, and the first of a pair stays.
    bench_hashes = [simhash(t) for t in benchmark["text"].astype(str)]
    extra_hashes = [simhash(t) for t in kept["text"]]
    ids = [str(i) for i in range(len(kept))]
    near_bench = set(simhash_matches(ids, extra_hashes, bench_hashes, 3))
    keep_mask = np.array([i not in near_bench for i in ids], dtype=bool)
    rest = kept.loc[keep_mask].reset_index(drop=True)
    rest["simhash"] = [h for h, k in zip(extra_hashes, keep_mask, strict=True) if k]
    rest, inner = dedupe_near_duplicates(
        rest, text_col="text", simhash_col="simhash", hamming_threshold=3
    )
    dropped[ExtraNegativesReason.NEAR_DUPLICATE] = int((~keep_mask).sum()) + int(
        inner.get("near_duplicate_simhash", 0)
    )
    kept = rest.drop(columns="simhash").reset_index(drop=True)
    _log_drops(dropped)
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

    Returns the train, validation and held-out parts. ``extra.sides`` gets the row and group
    count of each part. A file with fewer than ``MIN_GROUPS`` groups after the overlap check is
    refused, and the run trains nothing.
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
            f"A grouped split needs at least {MIN_GROUPS}. The run is refused and nothing "
            "trains.",
        )
    work, split = group_train_val_test(
        extra.frame, group_col="group_id", label_col="label", seed=seed
    )
    extra.dropped[ExtraNegativesReason.DUPLICATE] = extra.dropped.get(
        ExtraNegativesReason.DUPLICATE, 0
    ) + int(split.dropped.get("exact_text_duplicate", 0))
    extra.frame = work
    parts = (work.iloc[split.train], work.iloc[split.val], work.iloc[split.test])
    extra.sides = {
        name: {"rows": int(len(part)), "groups": int(part["group_id"].nunique())}
        for name, part in zip(("train", "val", "heldout"), parts, strict=True)
    }
    return parts


# ---------------------------------------------------------------------------
# held-out reporting
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a rate. ``(nan, nan)`` for ``n == 0``."""
    if n <= 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def group_bootstrap_rate(
    scores: np.ndarray, groups: np.ndarray, threshold: float, *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95 percent interval of the flag rate, resampling whole groups (publishers).

    Rows of one publisher are not independent, so the row-level Wilson interval is too narrow.
    ``(nan, nan)`` when there are fewer than two groups.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups).astype(str)
    names = np.unique(groups)
    if len(names) < 2:
        return (math.nan, math.nan)
    flagged = {g: int((scores[groups == g] >= threshold).sum()) for g in names}
    sizes = {g: int((groups == g).sum()) for g in names}
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        total = sum(sizes[g] for g in pick)
        rates.append(sum(flagged[g] for g in pick) / total)
    low, high = np.percentile(rates, [2.5, 97.5])
    return (float(low), float(high))


def precision_threshold(
    scores: np.ndarray, labels: np.ndarray, target: float = 0.85
) -> float | None:
    """The lowest score threshold whose precision on ``labels`` reaches ``target``.

    Read on the benchmark validation rows. ``None`` when no threshold reaches it.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    for threshold in sorted(set(scores.tolist())):
        flagged = scores >= threshold
        if flagged.sum() and labels[flagged].mean() >= target:
            return float(threshold)
    return None


#: A publisher bootstrap needs at least this many publishers to mean anything.
MIN_PUBLISHERS_FOR_INTERVAL = 5


def publisher_interval(
    scores: np.ndarray, groups: np.ndarray, threshold: float
) -> tuple[float, float] | None:
    """The publisher bootstrap interval, or ``None`` with fewer than 5 publishers."""
    if len(set(np.asarray(groups).astype(str))) < MIN_PUBLISHERS_FOR_INTERVAL:
        return None
    return group_bootstrap_rate(scores, groups, threshold)


def leave_one_publisher_out(
    scores: np.ndarray, groups: np.ndarray, threshold: float
) -> dict[str, float]:
    """Flag rate of the held-out rows with one publisher left out, for each publisher."""
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups).astype(str)
    out: dict[str, float] = {}
    for name in sorted(set(groups)):
        rest = scores[groups != name]
        if len(rest):
            out[name] = float((rest >= threshold).mean())
    return out


def _rate_line(name: str, scores: np.ndarray, groups: np.ndarray, threshold: float) -> str:
    n = len(scores)
    flagged = int((scores >= threshold).sum())
    wilson = wilson_interval(flagged, n)
    n_groups = len(set(np.asarray(groups).astype(str)))
    boot = publisher_interval(scores, groups, threshold)
    text = (
        f"- {name} (threshold {threshold:.4g}): {flagged}/{n} flagged, rate {flagged / n:.3f}; "
        f"row-level Wilson interval {wilson[0]:.3f} to {wilson[1]:.3f} "
        "(row-level only, too narrow, because rows of one publisher are not independent); "
    )
    if boot is None:
        text += f"publisher interval not meaningful ({n_groups} publishers)"
    else:
        text += f"publisher bootstrap interval {boot[0]:.3f} to {boot[1]:.3f}"
    loo = leave_one_publisher_out(scores, groups, threshold)
    if len(loo) > 1:
        text += "\n  - flag rate without one publisher: " + ", ".join(
            f"{k} {v:.3f}" for k, v in loo.items()
        )
    return text


def heldout_report(
    *,
    heldout_scores: np.ndarray,
    heldout_groups: np.ndarray,
    val_scores: np.ndarray,
    bench_val_scores: np.ndarray,
    bench_val_labels: np.ndarray,
    threshold: float,
    summary: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Text and a data record for the held-out reporting rows.

    The rates use the decision threshold, and the threshold at precision 0.85 on the benchmark
    validation rows. Calibration keeps the ranking, so the score quantiles and the ROC-AUC
    (held-out rows against the false-labelled benchmark validation rows) are the ranking
    view. The held-out rows are fact-checker articles. Live scoring already skips those, so
    this table does not measure other outlets that report on a false claim.
    """
    from sklearn.metrics import roc_auc_score

    p85 = precision_threshold(bench_val_scores, bench_val_labels, 0.85)
    held = np.asarray(heldout_scores, dtype=float)
    record: dict[str, Any] = {
        "n_heldout": int(len(held)),
        "n_heldout_groups": int(len(set(np.asarray(heldout_groups).astype(str)))),
        "threshold": threshold,
        "threshold_precision_085": p85,
        "benchmark_val_positive_rate": float(np.mean(bench_val_labels)),
    }
    lines = [
        "Scope. The extra negatives are articles of fact-checkers. Live scoring already skips "
        "those articles with the reason `fact_check_publisher`. This section does not measure "
        "other outlets that report on a false claim. That effect is measured only on the "
        "hand-labelled corpus sample, which has few such rows.",
        "",
        "Every held-out row is a true negative, so a flag rate is a false-positive rate. The "
        "held-out rows are not in the benchmark test set above.",
        "",
        f"- extra-negatives file: `{summary['file']}`, sha256 `{summary['sha256']}`",
        f"- rows in file: {summary['rows']}, publishers in file: {summary['groups']}",
        f"- rows used: {summary['rows_used']}, publishers used: {summary['groups_used']}",
    ]
    for name, side in (summary.get("sides") or {}).items():
        lines.append(f"- {name}: {side['rows']} rows, {side['groups']} publishers")
    if summary.get("dropped"):
        lines.append(
            "- dropped rows by reason: "
            + ", ".join(f"{k}={v}" for k, v in summary["dropped"].items())
        )
    lines.append(
        f"- benchmark validation positive rate: {record['benchmark_val_positive_rate']:.3f}. "
        "The calibrator was fitted on these rows only."
    )
    if len(held):
        lines.append(_rate_line("held-out", held, heldout_groups, threshold))
        record["flag_rate_threshold"] = float((held >= threshold).mean())
        record["without_one_publisher"] = leave_one_publisher_out(
            held, heldout_groups, threshold
        )
        if p85 is not None:
            lines.append(_rate_line("held-out", held, heldout_groups, p85))
            record["flag_rate_precision_085"] = float((held >= p85).mean())
        else:
            lines.append("- no threshold on the benchmark validation rows reaches precision 0.85")
        quantiles = np.percentile(held, [50, 90, 99])
        record["score_quantiles_p50_p90_p99"] = [float(q) for q in quantiles]
        lines.append(
            "- held-out score quantiles p50, p90, p99: " + ", ".join(f"{q:.3f}" for q in quantiles)
        )
        positives = np.asarray(bench_val_scores, dtype=float)[np.asarray(bench_val_labels) == 1]
        if len(positives):
            auc = roc_auc_score(
                np.r_[np.zeros(len(held)), np.ones(len(positives))], np.r_[held, positives]
            )
            record["auc_heldout_vs_false_benchmark_val"] = float(auc)
            lines.append(
                f"- ROC-AUC, held-out rows against false-labelled benchmark validation rows: "
                f"{auc:.3f}. A value near 1 means the ranking separates them."
            )
    else:
        lines.append("- held-out rows: 0, so no rate is reported")
    vals = np.asarray(val_scores, dtype=float)
    if len(vals):
        record["val_extra_flag_rate_threshold"] = float((vals >= threshold).mean())
        lines.append(
            f"- extra validation rows (reporting only, not in the calibration set): "
            f"{len(vals)} rows, {int((vals >= threshold).sum())} flagged at {threshold:.4g}"
        )
    return "\n".join(lines), record


def write_heldout(directory: Path, heldout: pd.DataFrame, summary: dict[str, Any]) -> list[Path]:
    """Save the held-out row ids, so any checkpoint can be scored on the same rows.

    The files hold ids and text hashes, never text.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids = heldout[["record_id", "group_id", "domain", "text_sha256"]].reset_index(drop=True)
    ids_path = directory / "reporting_heldout_ids.parquet"
    ids.to_parquet(ids_path, index=False)
    digest = hashlib.sha256("\n".join(sorted(ids["text_sha256"])).encode()).hexdigest()
    meta_path = directory / "reporting_heldout.json"
    meta_path.write_text(
        json.dumps(
            {
                "rows": int(len(ids)),
                "groups": int(ids["group_id"].nunique()),
                "ids_sha256": digest,
                "extra_negatives_file": summary["file"],
                "extra_negatives_sha256": summary["sha256"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [ids_path, meta_path]


def select_heldout(extra_path: Path, ids_path: Path) -> pd.DataFrame:
    """The rows of the extra-negatives file that a saved held-out list names.

    Refuses when a listed row is absent from the file or the file has another sha256 than the
    one that the list recorded.
    """
    ids_path = Path(ids_path)
    ids = pd.read_parquet(ids_path)
    meta_file = ids_path.with_name("reporting_heldout.json")
    if not meta_file.exists():
        raise ExtraNegativesRefused(
            ExtraNegativesReason.HELDOUT_META_MISSING,
            f"{meta_file} is missing, so nothing says which extra-negatives file made "
            f"{ids_path.name}",
        )
    recorded = json.loads(meta_file.read_text(encoding="utf-8"))["extra_negatives_sha256"]
    if recorded != file_sha256(extra_path):
        raise ExtraNegativesRefused(
            ExtraNegativesReason.HELDOUT_MISMATCH,
            f"{extra_path.name} has another sha256 than the file that made {ids_path.name}",
        )
    extra = load_extra_negatives(extra_path).frame
    wanted = set(ids["text_sha256"])
    rows = extra.loc[extra["text_sha256"].isin(wanted)].drop_duplicates("text_sha256")
    if len(rows) != len(wanted):
        raise ExtraNegativesRefused(
            ExtraNegativesReason.HELDOUT_MISMATCH,
            f"{len(wanted) - len(rows)} held-out row(s) are not in {extra_path.name}",
        )
    return rows.reset_index(drop=True)


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------
@dataclass
class LeakGuards:
    """What the builder must exclude so no article is both a negative and an evaluated claim."""

    record_ids: set[str]
    claim_texts: list[str]
    claim_simhashes: list[int]
    simhash_radius: int
    #: Manifest claims with no row in ``factchecks.parquet``, by claim sha256.
    missing_claims: list[str] = field(default_factory=list)


def select_reporting_rows(
    records: pd.DataFrame,
    domains: frozenset[str],
    *,
    lang: str = "en",
    guards: LeakGuards | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pick the fact-check articles of the corpus and shape them as extra negatives.

    A record matches when ``normalise_domain(source_detail)`` is in ``domains``. That is the
    test the ``misinfo`` stage uses for ``fact_check_publisher``, so it is the filter and not
    a drop. The group is the domain. Returns the frame with ``record_id``, ``text``,
    ``group_id`` and ``domain``, and the counts of rows dropped by reason.

    The fact-check evaluation scores the claim of an article as a positive, keyed by the
    record id. Training on the same article as a negative would leak the answer. So the
    builder drops, each under its own reason code:

    * every record in ``guards.record_ids`` (a claim row of any label, or a manifest entry),
    * every record whose normalised text contains a normalised frozen claim text,
    * every record whose simhash is within ``guards.simhash_radius`` bits of a frozen claim.
    """
    from ingest.normalize import simhash
    from modeling.config import normalise_domain
    from modeling.eval.labelling import simhash_matches

    guards = guards or LeakGuards(set(), [], [], 3)
    reasons = ExtraNegativesReason
    dropped = {
        reasons.LANGUAGE_MISMATCH: 0,
        reasons.ROW_EMPTY_TEXT: 0,
        reasons.CLAIM_RECORD: 0,
        reasons.FROZEN_CLAIM_TEXT: 0,
        reasons.FROZEN_CLAIM_SIMHASH: 0,
    }
    if "source_detail" not in records.columns:
        raise ExtraNegativesRefused(
            reasons.NO_SOURCE_DETAIL,
            "the corpus has no `source_detail` column, so no record matches a fact-checker",
        )
    domain = records["source_detail"].map(normalise_domain)
    is_fact_check = domain.isin(domains)
    lang_ok = records["lang"].astype(str).str.strip().str.lower().eq(lang.lower())
    has_text = records["text"].notna() & records["text"].astype(str).str.strip().ne("")
    dropped[reasons.LANGUAGE_MISMATCH] = int((is_fact_check & ~lang_ok).sum())
    dropped[reasons.ROW_EMPTY_TEXT] = int((is_fact_check & lang_ok & ~has_text).sum())
    keep = is_fact_check & lang_ok & has_text

    ids = records["id"].astype(str)
    has_claim = ids.isin(guards.record_ids)
    dropped[reasons.CLAIM_RECORD] = int((keep & has_claim).sum())
    keep &= ~has_claim

    claims = [c for c in (_squash(t) for t in guards.claim_texts) if c]
    if claims:
        squashed = records["text"].astype(str).map(_squash)
        contains = squashed.map(lambda text: any(c in text for c in claims))
        dropped[reasons.FROZEN_CLAIM_TEXT] = int((keep & contains).sum())
        keep &= ~contains

    if guards.claim_simhashes:
        candidates = records.loc[keep]
        if "simhash" in candidates.columns:
            hashes = [
                int(h) if h is not None and not pd.isna(h) else simhash(str(t))
                for h, t in zip(candidates["simhash"], candidates["text"], strict=True)
            ]
        else:
            hashes = [simhash(str(t)) for t in candidates["text"]]
        near = set(
            simhash_matches(
                list(ids[candidates.index]), hashes, guards.claim_simhashes, guards.simhash_radius
            )
        )
        is_near = ids.isin(near)
        dropped[reasons.FROZEN_CLAIM_SIMHASH] = int((keep & is_near).sum())
        keep &= ~is_near

    frame = pd.DataFrame(
        {
            "record_id": ids[keep].to_numpy(),
            "text": records.loc[keep, "text"].astype(str).to_numpy(),
            "group_id": domain[keep].to_numpy(),
            "domain": domain[keep].to_numpy(),
        }
    )
    return frame, dropped


def leak_guards(
    manifest_path: Path | None = None,
    *,
    ingest_settings: Any = None,
    allow_missing: bool = False,
    simhash_radius: int | None = None,
) -> LeakGuards:
    """The ids, claim texts and claim simhashes that the builder must exclude.

    The ids are every ``record_id`` of ``factchecks.parquet`` (all labels, kept and collapsed)
    and every ``record_id`` of ``configs/factcheck_eval_manifest.csv``. A frozen claim is a
    manifest entry. Its text and simhash come from the claim row with the same sha256. A
    frozen claim with no such row is counted. The call refuses when any exist, unless
    ``allow_missing`` is true, because the article of that claim is then excluded by id only.
    It refuses when the manifest is absent. ``simhash_radius`` defaults to
    ``claim_match.simhash_hamming`` of ``configs/labelling.yaml``.
    """
    import ingest.factcheck as factcheck
    from modeling.eval.factcheck_eval import claim_hash, default_manifest_path, read_manifest
    from modeling.eval.labelling import load_sampling_config

    path = Path(manifest_path) if manifest_path else default_manifest_path()
    manifest = read_manifest(path)
    if manifest is None:
        raise ExtraNegativesRefused(
            ExtraNegativesReason.MANIFEST_MISSING,
            f"{path} does not exist, so the frozen claim texts cannot be excluded",
        )
    rows = factcheck.read_factcheck_rows(ingest_settings)
    ids = {str(r["record_id"]) for r in rows} | {str(r["record_id"]) for r in manifest}
    wanted = {r["claim_sha256"] for r in manifest}
    texts: dict[str, int] = {}
    for row in rows:
        claim = row.get("claim")
        if claim and claim_hash(str(claim)) in wanted:
            texts[str(claim)] = int(row.get("claim_simhash") or 0)
    missing = sorted(wanted - {claim_hash(t) for t in texts})
    if missing:
        log.warning(
            "reporting-negatives: %d frozen claim(s) have no row in factchecks.parquet, "
            "reason=%s",
            len(missing),
            ExtraNegativesReason.FROZEN_CLAIM_MISSING,
        )
        if not allow_missing:
            raise ExtraNegativesRefused(
                ExtraNegativesReason.FROZEN_CLAIM_MISSING,
                f"{len(missing)} frozen claim(s) of {path.name} have no row in "
                "factchecks.parquet, so their text cannot be matched. Their articles are "
                "excluded by record id only. Pass --allow-missing-frozen-claims to accept.",
            )
    radius = simhash_radius
    if radius is None:
        radius = load_sampling_config().simhash_hamming
    return LeakGuards(
        record_ids=ids,
        claim_texts=list(texts),
        claim_simhashes=[h for h in texts.values() if h],
        simhash_radius=radius,
        missing_claims=missing,
    )


def heldout_rates(
    scores: np.ndarray, groups: np.ndarray, thresholds: list[float]
) -> list[dict[str, Any]]:
    """Flag rate at each threshold, with the row-level and the publisher interval."""
    out = []
    for threshold in thresholds:
        n = len(scores)
        flagged = int((np.asarray(scores) >= threshold).sum())
        wilson = wilson_interval(flagged, n)
        boot = publisher_interval(scores, groups, threshold)
        out.append(
            {
                "threshold": threshold,
                "n": n,
                "flagged": flagged,
                "rate": flagged / n if n else math.nan,
                "wilson_row_level": list(wilson),
                "bootstrap_by_publisher": None if boot is None else list(boot),
                "publishers": len(set(np.asarray(groups).astype(str))),
                "without_one_publisher": leave_one_publisher_out(scores, groups, threshold),
            }
        )
    return out
