"""Measure the tier 4 headline reader against hand labels.

Two steps. :func:`draw_sample` writes a label sheet and a key file. People fill in
the sheet. :func:`score_sheet` joins the filled sheet to the key and reports
precision and recall with Wilson intervals.

The sheet is blind: it holds the headline and the label columns and nothing that the
reader predicted. The key file holds the prediction, the pattern id, the pattern-set
hash and the sampling weight of each headline. Both files hold headline text, so
:func:`check_sheet_path` refuses a path inside the repository unless git ignores it.

A label is one of `false`, `true`, `middle` or `none`. `none` means the headline holds
no verdict. `middle` means a verdict that the label map drops, such as missing
context. The reader predicts the same four values, with `none` for a headline that
matches no pattern or that a guard cancelled.

The draw is stratified by language and by match or no match. Hindi headlines get at
least `MIN_HINDI` rows when the pool holds that many, and unmatched headlines get
about half of each language, so recall is measured on headlines that the reader did
not match. Each row carries the weight `population / sampled` of its stratum, and
every figure is weighted with it.

Only labels from `human` labellers give a measured error rate. See `MEASURED_KINDS`.
"""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ingest.config import load_yaml
from ingest.factcheck import DropReason, FactCheckExtractor, normalize_rating
from ingest.title_predicate import TitlePredicateReader, headline_lang, prepare_headline

DEFAULT_SEED = 20260813
DEFAULT_SAMPLE_SIZE = 100
MIN_HINDI = 30
Z_95 = 1.959963984540054

LABELS = ("false", "true", "middle", "none")
VERDICT_LABELS = ("false", "true")
#: Who filled in the labels. Only `human` supports a measured error rate. A model
#: label, reviewed or not, is a first pass.
LABELLER_KINDS = ("human", "model-reviewed", "model-unreviewed")
MEASURED_KINDS = ("human",)

#: The blind sheet. Nothing here shows what the reader predicted.
SHEET_COLUMNS = (
    "record_id",
    "url",
    "headline",
    "label_1",
    "label_2",
    "adjudicated_label",
    "labeller_1",
    "labeller_2",
    "labeller_kind",
)
KEY_SCHEMA_VERSION = 2


class SheetError(ValueError):
    """The sheet or key cannot be written or scored. The message says why."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_sheet_path(path: Path, repo_root: Path | None = None) -> Path:
    """Return the resolved path, or raise when it is inside the repository and not ignored."""
    root = (repo_root or _repo_root()).resolve()
    target = Path(path).expanduser().resolve()
    if target != root and root not in target.parents:
        return target
    try:
        ignored = (
            subprocess.run(
                ["git", "check-ignore", "-q", "--", str(target)],
                cwd=root,
                capture_output=True,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        ignored = False
    if not ignored:
        raise SheetError(
            f"{target} is inside the repository and git does not ignore it. The file holds "
            "headline text. Write it outside the repository or under the ignored data directory."
        )
    return target


def default_key_path(sheet: Path) -> Path:
    """``<sheet stem>.key.json`` beside the sheet."""
    sheet = Path(sheet)
    return sheet.with_name(f"{sheet.stem}.key.json")


def predicted_verdict(extractor: FactCheckExtractor, rating: str | None) -> str:
    """Map a pattern rating to `false`, `true` or `middle` with the config rating map."""
    if rating is None:
        return "none"
    mapped = extractor.ratings.get(normalize_rating(rating))
    if mapped in VERDICT_LABELS:
        return mapped
    return "middle"


def allocate(
    pools: dict[tuple[str, bool], int], n: int, min_hindi: int = MIN_HINDI
) -> dict[tuple[str, bool], int]:
    """Rows per stratum ``(language, matched)``.

    Hindi gets at least ``min_hindi`` rows when its pool allows, else all of it. The
    rest is shared by the pool sizes. Inside a language, matched headlines get at most
    half of the rows and the rest go to unmatched ones, with the spill going back to
    the other stratum when a pool is short.
    """
    pool = {lang: pools.get((lang, True), 0) + pools.get((lang, False), 0) for lang in ("en", "hi")}
    total = pool["en"] + pool["hi"]
    if total <= n:
        return {key: count for key, count in pools.items() if count}
    floor = min(min_hindi, math.ceil(0.3 * n), pool["hi"])
    n_hi = min(pool["hi"], max(floor, round(n * pool["hi"] / total)))
    n_en = min(pool["en"], n - n_hi)
    n_hi = min(pool["hi"], n - n_en)
    out: dict[tuple[str, bool], int] = {}
    for lang, n_lang in (("en", n_en), ("hi", n_hi)):
        matched, unmatched = pools.get((lang, True), 0), pools.get((lang, False), 0)
        take_matched = min(matched, math.ceil(n_lang / 2))
        take_unmatched = min(unmatched, n_lang - take_matched)
        take_matched = min(matched, n_lang - take_unmatched)
        for key, count in (((lang, True), take_matched), ((lang, False), take_unmatched)):
            if count:
                out[key] = count
    return out


def draw_sample(
    records: Iterable[dict[str, Any]],
    extractor: FactCheckExtractor,
    *,
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    min_hindi: int = MIN_HINDI,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Draw ``n`` headlines, stratified, and return the blind sheet rows and the key.

    The headline is prepared with the same function as the extractor, and the
    prediction comes from the same reader, so the sheet shows the text that the
    pipeline read. The draw depends on the seed and the set of ids and not on the
    order of the input. A record with no id or no title is not eligible and is counted
    by the extractor under its reason code.
    """
    reader = TitlePredicateReader(extractor.config)
    population: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("id") or "")
        title = prepare_headline(record.get("title"))
        if not record_id:
            extractor.drop(DropReason.NO_RECORD_ID, title)
            continue
        if not title:
            extractor.drop(DropReason.NO_CLAIM_TEXT, record_id)
            continue
        if record_id in population:
            continue
        match, guard_id = reader.classify(title)
        lang = headline_lang(title)
        population[record_id] = {
            "record": record,
            "headline": title,
            "lang": lang,
            "matched": match is not None,
            "match": match,
            "guard": guard_id,
        }
    strata: dict[tuple[str, bool], list[str]] = {}
    for record_id in sorted(population):
        item = population[record_id]
        strata.setdefault((item["lang"], item["matched"]), []).append(record_id)
    quotas = allocate({key: len(ids) for key, ids in strata.items()}, n, min_hindi)
    chosen: list[tuple[str, tuple[str, bool]]] = []
    for key, quota in sorted(quotas.items()):
        rng = random.Random(f"{seed}:{key[0]}:{int(key[1])}")
        chosen += [(rid, key) for rid in rng.sample(strata[key], quota)]
    chosen.sort()

    sheet: list[dict[str, str]] = []
    key_rows: list[dict[str, Any]] = []
    for record_id, stratum in chosen:
        item = population[record_id]
        match = item["match"]
        weight = len(strata[stratum]) / quotas[stratum]
        sheet.append(
            {
                "record_id": record_id,
                "url": str(item["record"].get("url") or ""),
                "headline": item["headline"],
                **{column: "" for column in SHEET_COLUMNS[3:]},
            }
        )
        key_rows.append(
            {
                "record_id": record_id,
                "publisher": str(item["record"].get("source_detail") or ""),
                "lang": item["lang"],
                "stratum": f"{stratum[0]}|{'matched' if stratum[1] else 'unmatched'}",
                "weight": weight,
                "predicted_verdict": predicted_verdict(extractor, match.rating if match else None),
                "predicted_rating": match.rating if match else "",
                "predicted_pattern": match.pattern_id if match else "",
                "guard": item["guard"] or "",
            }
        )
    key = {
        "schema_version": KEY_SCHEMA_VERSION,
        "seed": seed,
        "n_requested": n,
        "n_drawn": len(chosen),
        "pattern_hash": reader.pattern_hash,
        "strata": {
            f"{k[0]}|{'matched' if k[1] else 'unmatched'}": {
                "population": len(strata[k]),
                "sampled": quotas[k],
                "weight": len(strata[k]) / quotas[k],
            }
            for k in sorted(quotas)
        },
        "rows": key_rows,
    }
    return sheet, key


def write_sheet(
    rows: list[dict[str, str]],
    key: dict[str, Any],
    path: Path,
    key_path: Path | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the sheet CSV and the key JSON after the path checks.

    The sheet has a byte order mark, so Excel reads the Hindi text. Neither file is
    replaced without ``overwrite``, because a filled sheet is work that a person did.
    """
    target = check_sheet_path(path)
    key_target = check_sheet_path(key_path or default_key_path(target))
    for existing in (target, key_target):
        if existing.exists() and not overwrite:
            raise SheetError(f"{existing} exists. Pass overwrite to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    key_target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    key_target.write_text(
        json.dumps(key, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return target, key_target


# --- reading ---------------------------------------------------------------


def read_key(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SheetError(f"key file {path} does not exist")
    key = json.loads(path.read_text(encoding="utf-8"))
    if key.get("schema_version") != KEY_SCHEMA_VERSION or not key.get("rows"):
        raise SheetError(f"{path} is not a version {KEY_SCHEMA_VERSION} key file")
    return key


def read_sheet(path: Path) -> list[dict[str, str]]:
    """Read and check a filled sheet. Raises ``SheetError`` for every kind of gap.

    The adjudicated label may be empty where the two labels agree. Where they differ,
    an empty adjudicated label is allowed here and is counted by the scorer, which
    does not call such a result measured.
    """
    path = Path(path)
    if not path.exists():
        raise SheetError(f"{path} does not exist")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing = [c for c in SHEET_COLUMNS if c not in header]
        if missing:
            raise SheetError(f"{path}: missing column(s) {missing}")
        rows = [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]
    if not rows:
        raise SheetError(f"{path} has no rows")
    ids = [r["record_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SheetError(f"{path}: a record_id appears more than once")
    problems: list[str] = []
    for index, row in enumerate(rows, start=2):
        where = f"line {index} ({row['record_id'] or 'no id'})"
        for column in ("label_1", "label_2"):
            if not row[column]:
                problems.append(f"{where}: {column} is empty")
            elif row[column].casefold() not in LABELS:
                problems.append(f"{where}: {column} {row[column]!r} is not one of {list(LABELS)}")
        adjudicated = row["adjudicated_label"]
        if adjudicated and adjudicated.casefold() not in LABELS:
            problems.append(
                f"{where}: adjudicated_label {adjudicated!r} is not one of {list(LABELS)}"
            )
        for column in ("labeller_1", "labeller_2"):
            if not row[column]:
                problems.append(f"{where}: {column} is empty")
        if row["labeller_1"] and row["labeller_1"].casefold() == row["labeller_2"].casefold():
            problems.append(f"{where}: labeller_1 and labeller_2 are the same labeller")
        if row["labeller_kind"] not in LABELLER_KINDS:
            problems.append(
                f"{where}: labeller_kind {row['labeller_kind']!r} is not one of "
                f"{list(LABELLER_KINDS)}"
            )
    if problems:
        shown = "; ".join(problems[:8])
        more = f"; and {len(problems) - 8} more" if len(problems) > 8 else ""
        raise SheetError(f"{path}: the sheet is not fully labelled. {shown}{more}")
    for row in rows:
        for column in ("label_1", "label_2", "adjudicated_label", "labeller_1", "labeller_2"):
            row[column] = row[column].casefold()
    return rows


def join_sheet_and_key(rows: list[dict[str, str]], key: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach the key fields to each sheet row. The two files must hold the same ids."""
    by_id = {r["record_id"]: r for r in key["rows"]}
    sheet_ids, key_ids = {r["record_id"] for r in rows}, set(by_id)
    if sheet_ids != key_ids:
        raise SheetError(
            f"the sheet and the key hold different headlines: {len(sheet_ids - key_ids)} only in "
            f"the sheet, {len(key_ids - sheet_ids)} only in the key"
        )
    return [{**row, **by_id[row["record_id"]]} for row in rows]


# --- statistics --------------------------------------------------------------


def wilson(p: float, n_eff: float) -> tuple[float, float]:
    """Wilson score interval for a proportion ``p`` on ``n_eff`` effective observations."""
    z2 = Z_95**2
    centre = (p + z2 / (2 * n_eff)) / (1 + z2 / n_eff)
    half = Z_95 * math.sqrt(p * (1 - p) / n_eff + z2 / (4 * n_eff**2)) / (1 + z2 / n_eff)
    return max(0.0, centre - half), min(1.0, centre + half)


def _cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _estimate(
    hits: list[float], denominator: list[float], reason: str | None = None
) -> dict[str, Any]:
    """A weighted proportion with a Wilson interval on the Kish effective size."""
    out: dict[str, Any] = {
        "value": None,
        "ci_low": None,
        "ci_high": None,
        "n_effective": None,
        "numerator_rows": len(hits),
        "denominator_rows": len(denominator),
        "reason": reason,
    }
    if not denominator:
        out["reason"] = reason or "no row in the denominator"
        return out
    total = sum(denominator)
    p = min(1.0, sum(hits) / total)
    n_eff = total**2 / sum(w * w for w in denominator)
    out["value"] = p
    out["n_effective"] = n_eff
    out["ci_low"], out["ci_high"] = wilson(p, n_eff)
    return out


def _metric(
    rows: list[dict[str, Any]],
    labels: list[str | None],
    classes: tuple[str, ...],
    which: str,
    *,
    worst_case: bool = False,
) -> dict[str, Any]:
    """Weighted precision or recall of the reader for a class set.

    ``labels`` is the reference label of each row, or ``None`` for a row without one.
    With ``worst_case`` an unresolved row counts in the denominator and never as a hit,
    so every disagreement between the labellers is treated as a reader error.
    """
    hits: list[float] = []
    denominator: list[float] = []
    for row, label in zip(rows, labels, strict=True):
        weight = float(row["weight"])
        predicted = row["predicted_verdict"]
        if label is None:
            if not worst_case:
                continue
            in_scope = (
                predicted in classes
                if which == "precision"
                else (row["label_1"] in classes or row["label_2"] in classes)
            )
            if in_scope:
                denominator.append(weight)
            continue
        in_scope = predicted in classes if which == "precision" else label in classes
        if in_scope:
            denominator.append(weight)
            if predicted == label:
                hits.append(weight)
    reason = None if denominator else (
        f"no {'predicted' if which == 'precision' else 'labelled'} row in the class set"
    )
    return _estimate(hits, denominator, reason)


CLASS_SETS = {"verdict": VERDICT_LABELS, "false": ("false",), "true": ("true",)}


def _metrics_block(
    rows: list[dict[str, Any]], labels: list[str | None], *, worst_case: bool = False
) -> dict[str, Any]:
    return {
        name: {
            which: _metric(rows, labels, classes, which, worst_case=worst_case)
            for which in ("precision", "recall")
        }
        for name, classes in CLASS_SETS.items()
    }


def score_rows(
    joined: list[dict[str, Any]],
    key: dict[str, Any],
    *,
    current_pattern_hash: str | None = None,
    min_adjudicated_rows: int = 100,
    min_kappa: float = 0.6,
    min_language_rows: int = 30,
) -> dict[str, Any]:
    """Precision and recall of the reader, with Wilson intervals, from joined rows.

    ``joined`` is the output of :func:`join_sheet_and_key`.

    The reference label of a row is its adjudicated label. Where the two labellers
    agree and nobody adjudicated, the agreed label is used. A disagreement without an
    adjudicated label has no reference label: it is counted, left out of the main
    figure and treated as an error in the worst-case figure. Precision and recall
    against each labeller alone are reported too.
    """
    reference: list[str | None] = []
    for row in joined:
        if row["adjudicated_label"]:
            reference.append(row["adjudicated_label"])
        elif row["label_1"] == row["label_2"]:
            reference.append(row["label_1"])
        else:
            reference.append(None)
    unresolved = [
        r["record_id"] for r, label in zip(joined, reference, strict=True) if label is None
    ]
    kinds = Counter(r["labeller_kind"] for r in joined)
    kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    kappa = _cohen_kappa([r["label_1"] for r in joined], [r["label_2"] for r in joined])
    n_adjudicated = len(joined) - len(unresolved)
    by_language = Counter(r["lang"] for r in joined)
    pattern_hash = key.get("pattern_hash")

    reasons: list[str] = []
    if kind not in MEASURED_KINDS:
        reasons.append(f"labeller kind is {kind!r}, not 'human'")
    if unresolved:
        reasons.append(f"{len(unresolved)} row(s) have no adjudicated label")
    if n_adjudicated < min_adjudicated_rows:
        reasons.append(f"{n_adjudicated} adjudicated rows, fewer than {min_adjudicated_rows}")
    if kappa is None or kappa < min_kappa:
        reasons.append(f"Cohen kappa {kappa} is below {min_kappa}")
    for language in ("en", "hi"):
        if by_language.get(language, 0) < min_language_rows:
            reasons.append(
                f"{by_language.get(language, 0)} {language} rows, fewer than {min_language_rows}"
            )
    if current_pattern_hash is not None and pattern_hash != current_pattern_hash:
        reasons.append("the pattern set in the key differs from the config")

    per_language = {
        lang: _subset_metrics(joined, reference, lambda r, lang=lang: r["lang"] == lang)
        for lang in sorted(by_language)
    }
    per_publisher = {
        publisher: _subset_metrics(joined, reference, lambda r, p=publisher: r["publisher"] == p)
        for publisher in sorted({r["publisher"] for r in joined})
    }
    metrics = _metrics_block(joined, reference)
    result: dict[str, Any] = {
        "schema_version": 2,
        "interval": "Wilson 95%, weighted proportion on the Kish effective sample size",
        "draw_seed": key.get("seed"),
        "pattern_hash": pattern_hash,
        "current_pattern_hash": current_pattern_hash,
        "labeller_kind": kind,
        "labeller_kind_counts": dict(sorted(kinds.items())),
        "labellers": {
            "labeller_1": sorted({r["labeller_1"] for r in joined}),
            "labeller_2": sorted({r["labeller_2"] for r in joined}),
        },
        "is_measured_error_rate": not reasons,
        "not_measured_reasons": reasons,
        "n_rows": len(joined),
        "n_by_language": dict(sorted(by_language.items())),
        "n_adjudicated": n_adjudicated,
        "n_unresolved": len(unresolved),
        "unresolved_record_ids": sorted(unresolved),
        "agreement": sum(r["label_1"] == r["label_2"] for r in joined) / len(joined),
        "cohen_kappa": kappa,
        "floors": {
            "min_adjudicated_rows": min_adjudicated_rows,
            "min_kappa": min_kappa,
            "min_language_rows": min_language_rows,
        },
        "strata": key.get("strata"),
        "metrics": metrics,
        "worst_case_metrics": _metrics_block(joined, reference, worst_case=True),
        "against_labeller_1": _metrics_block(joined, [r["label_1"] for r in joined]),
        "against_labeller_2": _metrics_block(joined, [r["label_2"] for r in joined]),
        "per_language": per_language,
        "per_publisher": per_publisher,
    }
    false = metrics["false"]
    result["suggested_config_block"] = {
        "status": "measured" if not reasons else "not_measured",
        "reason": None if not reasons else "; ".join(reasons),
        "labeller_kind": kind,
        "pattern_hash": pattern_hash,
        "cohen_kappa": kappa,
        "n_headlines": n_adjudicated,
        "n_by_language": dict(sorted(by_language.items())),
        "precision": false["precision"]["value"],
        "recall": false["recall"]["value"],
    }
    return result


def _subset_metrics(rows, reference, keep) -> dict[str, Any]:
    indices = [i for i, row in enumerate(rows) if keep(row)]
    subset = [rows[i] for i in indices]
    return {
        "n_rows": len(subset),
        "metrics": _metrics_block(subset, [reference[i] for i in indices]),
    }


def score_sheet(sheet: Path, key_path: Path, out: Path) -> dict[str, Any]:
    """Read the sheet and the key, score them and write the JSON to ``out``.

    Refuses a sheet with an empty or unknown label. The floors and the current pattern
    hash come from `configs/factcheck.yaml`.
    """
    from ingest.title_predicate import pattern_set_hash

    config = load_yaml("factcheck")
    block = config.get("title_predicate") or {}
    key = read_key(key_path)
    result = score_rows(
        join_sheet_and_key(read_sheet(sheet), key),
        key,
        current_pattern_hash=pattern_set_hash(config),
        min_adjudicated_rows=int(block.get("min_adjudicated_rows", 100)),
        min_kappa=float(block.get("min_kappa", 0.6)),
        min_language_rows=int(block.get("min_language_rows", 30)),
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
