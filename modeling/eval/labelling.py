"""The hand-labelling workflow: draw a weighted sample, read the sheet, measure agreement.

Three parts that share one sheet layout.

* ``build_weighted_sample`` draws the rows a person will label. It favours rows that
  are likely misinformation-like, because a uniform draw holds almost none. Every row
  records its stratum and its sampling weight, so a metric can be reweighted back to the
  corpus. The sheet holds no model score, and its stratum is a neutral code.
* ``resolve_sheet`` reads the adjudicated label and refuses labels that no person has
  reviewed, unless the caller opts in. An opt-in stamps every output.
* ``agreement_report`` measures how well the two labellers agree before adjudication.

Every row that is left out of a count has a reason code.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.eval.corpus_transfer import _is_null, parse_label, unscorable_reason
from modeling.eval.metrics import Interval
from modeling.eval.weighted import DEFAULT_SEED

log = logging.getLogger(__name__)


SHEET_COLUMNS = (
    "id",
    "source",
    "lang",
    "title",
    "text",
    "sample_seed",
    "label_a",
    "labeller_a",
    "label_b",
    "labeller_b",
    "reports_on_false_claim_a",
    "reports_on_false_claim_b",
    "adjudicated_label",
    "adjudicated_reports_on_false_claim",
    "labeller_kind",
    "notes",
)
BLANK_COLUMNS = SHEET_COLUMNS[SHEET_COLUMNS.index("label_a") :]
#: The design file holds what the sample design knows about each row. A labeller who sees
#: a stratum or a weight learns how likely the row is to be misinformation-like, so these
#: columns stay out of the sheet and join back on ``id`` when the sheet is read.
DESIGN_COLUMNS = ("id", "stratum", "pool", "drawn", "weight")

HUMAN = "human"
MODEL_KINDS = ("model-unreviewed", "llm-unreviewed")
MODEL_LABEL_NOTE = "labeller: model, unreviewed by a person"

# Reason codes. Each is counted.
FACT_CHECK_PUBLISHER = "fact_check_publisher"
EXCLUDED_SOURCE = "excluded_source"
NO_TITLE = "no_title"
NO_SIMHASH = "no_simhash"
EMBEDDING_UNAVAILABLE = "embedding_unavailable"
EMBEDDING_RECORD_MISSING = "embedding_record_missing"
EMBEDDING_DIMENSION_MISMATCH = "embedding_dimension_mismatch"
NO_FALSE_CLAIMS = "no_false_claims"
UNMARKED_LABELLER_KIND = "labeller_kind_unmarked"
MODEL_LABELS_REFUSED = "model_labels_refused"
INVALID_WEIGHT = "invalid_weight"
EMBEDDING_MODEL_MISMATCH = "embedding_model_mismatch"
EMBEDDING_MODEL_UNVERIFIED = "embedding_model_unverified"
DESIGN_MISSING = "design_missing"
CLAIM_WITHOUT_SIMHASH = "claim_without_simhash"
#: A labeller writes this word in ``notes`` when the label needs a web search to check.
UNCHECKED_NOTE = "unchecked"


# ---------------------------------------------------------------------------
# claim matching
# ---------------------------------------------------------------------------
def _bands(radius: int, bits: int = 64) -> list[tuple[int, int]]:
    """Split ``bits`` into ``radius + 1`` bands of (shift, mask).

    Two hashes within ``radius`` bit flips agree exactly on at least one band, so an
    exact lookup per band finds every candidate. The Hamming check then confirms it.
    """
    count = radius + 1
    size, extra = divmod(bits, count)
    out, shift = [], 0
    for i in range(count):
        width = size + (1 if i < extra else 0)
        out.append((shift, (1 << width) - 1))
        shift += width
    return out


def simhash_matches(
    record_ids: Sequence[str],
    record_hashes: Sequence[int | None],
    claim_hashes: Iterable[int],
    radius: int,
) -> dict[str, int]:
    """Record id to the smallest Hamming distance to any claim, for records within ``radius``.

    A hash of 0 means "no signal" (``ingest.normalize.simhash``) and never matches.
    """
    bands = _bands(radius)
    index: list[dict[int, list[int]]] = [{} for _ in bands]
    for claim in {int(h) for h in claim_hashes if h}:
        for table, (shift, mask) in zip(index, bands, strict=True):
            table.setdefault((claim >> shift) & mask, []).append(claim)
    found: dict[str, int] = {}
    for rid, value in zip(record_ids, record_hashes, strict=True):
        if not value:
            continue
        value = int(value)
        best = None
        for table, (shift, mask) in zip(index, bands, strict=True):
            for claim in table.get((value >> shift) & mask, ()):
                distance = bin(value ^ claim).count("1")
                if distance <= radius and (best is None or distance < best):
                    best = distance
        if best is not None:
            found[str(rid)] = best
    return found


def embedding_matches(
    record_ids: Sequence[str],
    record_matrix: np.ndarray,
    claim_matrix: np.ndarray,
    min_similarity: float,
    block: int = 2048,
) -> set[str]:
    """Ids of records whose best cosine similarity to any claim reaches ``min_similarity``."""
    if not len(record_ids) or not len(claim_matrix):
        return set()
    claims = _unit(claim_matrix)
    records = _unit(record_matrix)
    hit: set[str] = set()
    for start in range(0, len(records), block):
        best = (records[start : start + block] @ claims.T).max(axis=1)
        hit.update(str(record_ids[start + i]) for i in np.where(best >= min_similarity)[0])
    return hit


def _unit(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norm == 0, 1.0, norm)


# ---------------------------------------------------------------------------
# the weighted sample
# ---------------------------------------------------------------------------
@dataclass
class Stratum:
    name: str
    share: float


@dataclass
class SamplingConfig:
    seed: int = DEFAULT_SEED
    likely_sources: tuple[str, ...] = ("reddit", "mastodon", "youtube", "telegram")
    #: Sources kept out of the draw entirely, counted as ``excluded_source``. A source whose
    #: corpus cannot carry the case under study contributes only noise to a transfer measure,
    #: and its stratum share is better spent on sources that can.
    exclude_sources: frozenset[str] = frozenset()
    strata: tuple[Stratum, ...] = ()
    #: Record-to-claim matching. Separate from the claim-to-claim dedup in
    #: ``configs/factcheck.yaml``, because a post is matched to a claim, not a claim to a claim.
    simhash_hamming: int = 3
    min_similarity: float | None = None
    embedding_model: str | None = None
    claim_labels: tuple[str, ...] = ("false",)


def parse_sampling_config(block: dict[str, Any]) -> SamplingConfig:
    strata = tuple(
        Stratum(str(item["name"]), float(item["share"])) for item in block.get("strata") or []
    )
    if not strata:
        raise ValueError("configs/labelling.yaml lists no strata")
    if len({s.name for s in strata}) != len(strata):
        raise ValueError("configs/labelling.yaml repeats a stratum name")
    if any(s.share <= 0 for s in strata):
        raise ValueError("every stratum share in configs/labelling.yaml must be above 0")
    if "other" not in {s.name for s in strata}:
        raise ValueError("configs/labelling.yaml needs a stratum named `other`")
    match = block.get("claim_match") or {}
    similarity = match.get("min_similarity")
    if similarity is not None and not 0.0 < float(similarity) <= 1.0:
        raise ValueError("claim_match.min_similarity must be above 0 and at most 1")
    return SamplingConfig(
        seed=int(block.get("seed", DEFAULT_SEED)),
        likely_sources=tuple(str(s) for s in block.get("likely_positive_sources") or ()),
        exclude_sources=frozenset(str(s).lower() for s in block.get("exclude_sources") or ()),
        strata=strata,
        simhash_hamming=int(match.get("simhash_hamming", 3)),
        min_similarity=None if similarity is None else float(similarity),
        embedding_model=match.get("embedding_model") or None,
        claim_labels=tuple(str(s) for s in block.get("claim_labels") or ("false",)),
    )


def load_sampling_config() -> SamplingConfig:
    from modeling.config import load_yaml

    return parse_sampling_config(load_yaml("labelling"))


class AllocationError(ValueError):
    """The sample is too small to hold one row of every stratum that has rows."""


def allocate(pools: dict[str, int], shares: dict[str, float], n: int) -> dict[str, int]:
    """Rows to draw per stratum: proportional to share, at least 1 where the pool is not empty.

    A stratum cannot give more rows than its pool holds. What it cannot fill goes to the
    strata that still have rows. A stratum with rows but no draw would leave part of the
    corpus unrepresented, so ``n`` below the number of such strata raises.
    """
    live = sorted((k for k, p in pools.items() if p > 0), key=lambda k: -shares[k])
    if not live:
        return {}
    if n < len(live):
        raise AllocationError(
            f"n={n} is below the {len(live)} strata that have rows ({', '.join(live)}). "
            "Every such stratum needs at least one row, or the reweighted figures would "
            "leave part of the corpus out. Raise --n."
        )
    total = sum(shares[k] for k in live)
    alloc = {k: min(pools[k], max(1, int(n * shares[k] / total))) for k in live}
    while sum(alloc.values()) < n:
        room = [k for k in live if alloc[k] < pools[k]]
        if not room:
            break
        alloc[max(room, key=lambda k: shares[k] / (alloc[k] + 1))] += 1
    while sum(alloc.values()) > n:
        biggest = max(alloc, key=lambda k: alloc[k])
        if alloc[biggest] <= 1:
            break
        alloc[biggest] -= 1
    return alloc


@dataclass
class ClaimEvidence:
    """What the sampler knows about fact-check claims. All parts are optional."""

    hashes: list[int] = field(default_factory=list)
    #: Embeddings of the claims and of corpus records, from the same model.
    claim_matrix: np.ndarray | None = None
    record_ids: list[str] = field(default_factory=list)
    record_matrix: np.ndarray | None = None
    min_similarity: float | None = None
    #: Reason code for an embedding path that did not run, or ``None``.
    embedding_reason: str | None = EMBEDDING_UNAVAILABLE
    #: Notes the evidence loader wants printed, such as an unverified model id.
    notes: list[str] = field(default_factory=list)
    #: Counted degradations of the evidence itself, such as a claim with no simhash.
    degraded: dict[str, int] = field(default_factory=dict)


@dataclass
class WeightedSample:
    frame: pd.DataFrame
    pool: int
    excluded: dict[str, int]
    #: Counted degradations that do not drop a row: a path that did not run, a missing field.
    degraded: dict[str, int]
    seed: int
    #: Stratum code to its meaning, pool size and drawn rows.
    strata: dict[str, dict[str, Any]]
    claim_match_counts: dict[str, int]
    notes: list[str]
    #: One row per sampled row: id, stratum, pool, drawn, weight. Written beside the sheet.
    design: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=DESIGN_COLUMNS))


def build_weighted_sample(
    records: pd.DataFrame,
    *,
    n: int,
    language_allowed: Callable[[str | None], bool],
    fact_checker_domains: frozenset[str],
    config: SamplingConfig,
    claims: ClaimEvidence | None = None,
    lang: str | None = None,
    seed: int | None = None,
) -> WeightedSample:
    """Stratified draw of rows the misinfo stage can score, without fact-checker articles.

    ``records`` needs ``id``, ``source``, ``source_detail``, ``lang``, ``text``, and
    optionally ``raw`` and ``simhash``. A Hindi draw also needs a title, because the
    Hindi scorer reads titles. Rows that cannot be sampled are counted by reason.
    """
    from modeling.config import normalise_domain
    from modeling.titles import title_of

    seed = config.seed if seed is None else seed
    claims = claims or ClaimEvidence()
    pool = records
    if lang is not None:
        pool = pool[pool["lang"].map(lambda v: not _is_null(v) and str(v).lower() == lang)]
    pool = pool.copy()
    pool["title"] = [title_of(r) or "" for r in pool["raw"]] if "raw" in pool else ""

    excluded: Counter[str] = Counter()
    degraded: Counter[str] = Counter()
    keep = []
    domains = pool["source_detail"].map(normalise_domain) if "source_detail" in pool else None
    sources = (
        pool["source"].map(lambda v: "" if _is_null(v) else str(v).lower())
        if "source" in pool
        else None
    )
    for i, (row_lang, text, title) in enumerate(
        zip(pool["lang"], pool["text"], pool["title"], strict=True)
    ):
        reason = None
        # A blanket rule, so it is read before the per-row checks: an excluded source never
        # enters the draw, whatever the row says.
        if sources is not None and sources.iloc[i] in config.exclude_sources:
            reason = EXCLUDED_SOURCE
        if reason is None and domains is not None and domains.iloc[i] in fact_checker_domains:
            reason = FACT_CHECK_PUBLISHER
        if reason is None:
            if lang == "hi":
                reason = laya_title_skip_reason(title, row_lang, language_allowed)
            else:
                reason = unscorable_reason(row_lang, text, language_allowed)
        if reason:
            excluded[reason] += 1
        keep.append(reason is None)
    scorable = pool[keep].reset_index(drop=True)
    notes: list[str] = []
    empty = WeightedSample(scorable.iloc[0:0], len(pool), dict(excluded), {}, seed, {}, {}, notes)
    if not len(scorable):
        return empty

    ids = scorable["id"].astype(str).tolist()
    by_simhash: dict[str, int] = {}
    if claims.hashes:
        if "simhash" in scorable:
            hashes = [None if _is_null(h) else int(h) for h in scorable["simhash"]]
            degraded[NO_SIMHASH] += sum(h is None for h in hashes)
            by_simhash = simhash_matches(ids, hashes, claims.hashes, config.simhash_hamming)
        else:
            degraded[NO_SIMHASH] += len(scorable)
            notes.append("no simhash column in the records: simhash matching did not run")
    else:
        degraded[NO_FALSE_CLAIMS] += 1
        notes.append("no fact-check claims on disk: no row can match a claim")
    by_embedding: set[str] = set()
    if claims.embedding_reason is None and claims.record_matrix is not None:
        if claims.claim_matrix is None or claims.min_similarity is None:
            degraded[EMBEDDING_UNAVAILABLE] += 1
        elif claims.record_matrix.shape[1] != claims.claim_matrix.shape[1]:
            degraded[EMBEDDING_DIMENSION_MISMATCH] += 1
            notes.append("embedding widths of records and claims differ: embedding path skipped")
        else:
            row_of = {rid: i for i, rid in enumerate(claims.record_ids)}
            present = [rid for rid in ids if rid in row_of]
            degraded[EMBEDDING_RECORD_MISSING] += len(ids) - len(present)
            matrix = claims.record_matrix[[row_of[rid] for rid in present]]
            by_embedding = embedding_matches(
                present, matrix, claims.claim_matrix, claims.min_similarity
            )
    else:
        reason = claims.embedding_reason or EMBEDDING_UNAVAILABLE
        degraded[reason] += 1
        notes.append(f"embedding match path did not run ({reason}); only simhash matches count")
    notes.extend(claims.notes)
    for reason, count in claims.degraded.items():
        degraded[reason] += count

    codes = {s.name: f"S{i + 1}" for i, s in enumerate(config.strata)}
    names = set(codes)
    stratum = []
    for rid, source in zip(ids, scorable["source"], strict=True):
        if "claim_match" in names and (rid in by_simhash or rid in by_embedding):
            stratum.append("claim_match")
        elif source in config.likely_sources and source in names:
            stratum.append(source)
        else:
            stratum.append("other")
    scorable["stratum_name"] = stratum
    match_counts = {
        "simhash": len(by_simhash),
        "embedding": len(by_embedding),
        "both": len(set(by_simhash) & by_embedding),
        "any": len(set(by_simhash) | by_embedding),
    }
    if not match_counts["any"]:
        notes.append(
            "WARNING: no scorable record matches a fact-check claim, so the claim_match "
            "stratum is empty and the sample holds no row that was chosen for a claim match"
        )

    shares = {s.name: s.share for s in config.strata}
    pools = {name: int((scorable["stratum_name"] == name).sum()) for name in shares}
    alloc = allocate(pools, shares, n)
    parts, meta = [], {}
    for name in shares:
        count = alloc.get(name, 0)
        meta[codes[name]] = {"name": name, "pool": pools[name], "drawn": count}
        if not count:
            continue
        stratum_rows = scorable[scorable["stratum_name"] == name]
        drawn = stratum_rows.sample(n=count, random_state=seed).copy()
        drawn["stratum"] = codes[name]
        drawn["pool"] = pools[name]
        drawn["drawn"] = count
        drawn["weight"] = round(pools[name] / count, 6)
        parts.append(drawn)
    if not parts:
        return empty
    # Shuffle so that row order does not reveal the stratum.
    chosen = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    chosen["sample_seed"] = seed
    for column in BLANK_COLUMNS:
        if column not in chosen:
            chosen[column] = ""
    return WeightedSample(
        chosen[list(SHEET_COLUMNS)],
        len(pool),
        dict(excluded),
        dict(degraded),
        seed,
        meta,
        match_counts,
        notes,
        design=chosen[list(DESIGN_COLUMNS)].copy(),
    )


def laya_title_skip_reason(
    title: str, lang: Any, language_allowed: Callable[[str | None], bool]
) -> str | None:
    """Why the ``laya`` stage declines a record, in the stage's own order and codes.

    The stage scores the title. It declines a record with no title (``no_title``), a title
    shorter than ``MIN_CHARS`` (``text_too_short``) and a language it does not score
    (``unsupported_language``). The fact-checker skip comes first, in the caller.
    """
    from modeling.aux.base import SkipReason
    from modeling.text.laya_clf import MIN_CHARS, LayaSkipReason

    if not title:
        return LayaSkipReason.NO_TITLE
    if len(title.strip()) < MIN_CHARS:
        return SkipReason.TEXT_TOO_SHORT
    if not language_allowed(None if _is_null(lang) else str(lang)):
        return SkipReason.UNSUPPORTED_LANGUAGE
    return None


# ---------------------------------------------------------------------------
# reading a sheet
# ---------------------------------------------------------------------------
def read_sheet(path: Path) -> pd.DataFrame:
    """Every cell as text, so a blank label stays blank."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def design_path(sheet_path: Path) -> Path:
    """The design file of a sheet: ``x.csv`` has ``x.design.csv``."""
    return sheet_path.with_suffix(".design.csv")


def read_design(sheet_path: Path) -> pd.DataFrame | None:
    """The design file beside the sheet, or ``None`` when there is none."""
    path = design_path(sheet_path)
    return read_sheet(path) if path.exists() else None


@dataclass
class ResolvedSheet:
    """A sheet read for evaluation.

    ``frame`` has ``label`` (the adjudicated label), ``reports`` (0, 1 or NaN), ``unchecked``
    (true when ``notes`` says the claim needed a web search) and, when a design file was
    joined, ``stratum``, ``pool`` and ``drawn``.
    """

    frame: pd.DataFrame
    refusal: str | None = None
    #: Set when model labels were used with the opt-in flag. Printed on every output.
    labeller_note: str | None = None
    kinds: dict[str, int] = field(default_factory=dict)
    has_design: bool = False


def resolve_sheet_file(path: Path, *, allow_model_labels: bool = False) -> ResolvedSheet:
    """Read a sheet and its design file from disk, then resolve them."""
    return resolve_sheet(
        read_sheet(path), design=read_design(path), allow_model_labels=allow_model_labels
    )


def resolve_sheet(
    sheet: pd.DataFrame,
    *,
    design: pd.DataFrame | None = None,
    allow_model_labels: bool = False,
) -> ResolvedSheet:
    """Read the adjudicated label, join the design and enforce the labeller rule.

    A label from a model that no person reviewed is not ground truth. The sheet is
    refused when a labelled row has such a kind, unless ``allow_model_labels`` is true,
    and then the result carries ``MODEL_LABEL_NOTE``. A labelled row with no
    ``labeller_kind`` is refused in every case, because its source is unknown.
    A sheet from before the adjudicated columns, with only ``label``, is read as is.
    With a ``design`` frame, each labelled row needs a design row with a positive pool.
    """
    column = "adjudicated_label" if "adjudicated_label" in sheet else "label"
    frame = sheet.copy()
    if column not in frame:
        return ResolvedSheet(frame, refusal="the sheet has no adjudicated_label column")
    frame["label"] = frame[column]
    parsed = [parse_label(v) for v in frame["label"]]
    labelled = np.array([p in (0, 1) and p is not None for p in parsed], dtype=bool)
    kind_cells = (
        frame["labeller_kind"].str.strip().str.lower()
        if "labeller_kind" in frame
        else pd.Series("", index=frame.index)
    )
    kinds = dict(Counter(kind_cells[labelled]))
    result = ResolvedSheet(frame, kinds=kinds)

    unmarked = [k for k in kinds if k not in (HUMAN, *MODEL_KINDS)]
    if unmarked:
        result.refusal = (
            f"{sum(kinds[k] for k in unmarked)} labelled rows have no valid labeller_kind "
            f"({UNMARKED_LABELLER_KIND}). Set it to `{HUMAN}` for labels that a person made "
            f"or `{MODEL_KINDS[0]}` for labels from a model."
        )
        return result
    model_rows = sum(kinds.get(k, 0) for k in MODEL_KINDS)
    if model_rows:
        if not allow_model_labels:
            result.refusal = (
                f"{model_rows} labelled rows come from a model and no person reviewed them "
                f"({MODEL_LABELS_REFUSED}). A model label is not ground truth. Review the "
                "labels and set labeller_kind to `human`, or pass --allow-model-labels."
            )
            return result
        result.labeller_note = MODEL_LABEL_NOTE

    if design is not None:
        missing = [c for c in ("id", "stratum", "pool") if c not in design]
        if missing:
            result.refusal = f"the design file lacks the columns {missing} ({DESIGN_MISSING})"
            return result
        joined = frame.drop(columns=["stratum", "pool", "drawn"], errors="ignore").merge(
            design[[c for c in ("id", "stratum", "pool", "drawn") if c in design]],
            on="id",
            how="left",
        )
        joined["pool"] = pd.to_numeric(joined["pool"], errors="coerce")
        no_stratum = joined["stratum"].isna() | (joined["stratum"].astype(str).str.strip() == "")
        bad = int(((joined["pool"].isna() | (joined["pool"] <= 0) | no_stratum) & labelled).sum())
        if bad:
            result.refusal = (
                f"{bad} labelled rows have no design row with a stratum and a positive pool "
                f"({DESIGN_MISSING})"
            )
            return result
        frame = joined
        result.frame = frame
        result.has_design = True
    reports = (
        [parse_label(v) for v in frame["adjudicated_reports_on_false_claim"]]
        if "adjudicated_reports_on_false_claim" in frame
        else [None] * len(frame)
    )
    frame["reports"] = [r if r in (0, 1) else np.nan for r in reports]
    notes = frame["notes"] if "notes" in frame else pd.Series("", index=frame.index)
    frame["unchecked"] = notes.astype(str).str.contains(
        rf"\b{UNCHECKED_NOTE}\b", case=False, regex=True
    )
    return result


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------
@dataclass
class ColumnAgreement:
    column: str
    n_pairs: int = 0
    raw_agreement: float | None = None
    kappa: Interval | None = None
    #: Rows where one labeller or both left the cell blank or wrote an invalid value.
    incomplete: int = 0
    note: str = ""
    disagreements: list[dict[str, str]] = field(default_factory=list)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's kappa for two binary raters. ``None`` when chance agreement is 1."""
    n = len(a)
    if n == 0:
        return None
    observed = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def kappa_interval(
    a: np.ndarray, b: np.ndarray, *, n_bootstrap: int = 2000, seed: int = DEFAULT_SEED
) -> Interval | None:
    point = cohen_kappa(a, b)
    if point is None:
        return None
    n = len(a)
    if n < 10:
        return Interval(point, float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, size=n)
        value = cohen_kappa(a[index], b[index])
        if value is not None:
            draws.append(value)
    if len(draws) < n_bootstrap // 10:
        return Interval(point, float("nan"), float("nan"), len(draws))
    low, high = np.percentile(draws, [2.5, 97.5])
    return Interval(point, float(low), float(high), len(draws))


def agreement_for(
    sheet: pd.DataFrame, column: str, *, n_bootstrap: int = 2000, seed: int = DEFAULT_SEED
) -> ColumnAgreement:
    a_col, b_col = f"{column}_a", f"{column}_b"
    out = ColumnAgreement(column)
    if a_col not in sheet or b_col not in sheet:
        out.note = f"the sheet has no {a_col} and {b_col} columns"
        return out
    a_all = [parse_label(v) for v in sheet[a_col]]
    b_all = [parse_label(v) for v in sheet[b_col]]
    keep = [
        i for i, (x, y) in enumerate(zip(a_all, b_all, strict=True)) if x in (0, 1) and y in (0, 1)
    ]
    out.incomplete = len(sheet) - len(keep)
    out.n_pairs = len(keep)
    if not keep:
        out.note = "no row has a valid label from both labellers"
        return out
    a = np.array([a_all[i] for i in keep], dtype=int)
    b = np.array([b_all[i] for i in keep], dtype=int)
    out.raw_agreement = float((a == b).mean())
    out.kappa = kappa_interval(a, b, n_bootstrap=n_bootstrap, seed=seed)
    if out.kappa is None:
        out.note = "kappa is undefined: both labellers used one class for every row"
    for i, x, y in zip(keep, a, b, strict=True):
        if x != y:
            out.disagreements.append(
                {"id": str(sheet["id"].iloc[i]), f"{a_col}": str(x), f"{b_col}": str(y)}
            )
    return out


def agreement_report(
    sheet: pd.DataFrame, *, n_bootstrap: int = 2000, seed: int = DEFAULT_SEED
) -> dict[str, ColumnAgreement]:
    return {
        column: agreement_for(sheet, column, n_bootstrap=n_bootstrap, seed=seed)
        for column in ("label", "reports_on_false_claim")
    }


def render_agreement(report: dict[str, ColumnAgreement], seed: int) -> list[str]:
    """Plain lines. Ids only: the text of a row stays in the sheet."""
    lines = [f"agreement before adjudication (bootstrap seed {seed})"]
    for column, item in report.items():
        lines.append(
            f"  {column}: {item.n_pairs} rows labelled by both, {item.incomplete} incomplete"
        )
        if item.raw_agreement is not None:
            lines.append(f"    raw agreement {item.raw_agreement:.3f}")
            lines.append(
                "    Cohen kappa " + (str(item.kappa) if item.kappa is not None else "undefined")
            )
        if item.note:
            lines.append(f"    note: {item.note}")
        lines.append(f"    disagreements: {len(item.disagreements)}")
        for d in item.disagreements:
            lines.append("      " + ", ".join(f"{k}={v}" for k, v in d.items()))
    return lines


def fill_agreed(sheet: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Copy a label into the adjudicated column where both labellers agree.

    A disagreement and a cell that already holds an adjudicated value stay as they are.
    """
    out = sheet.copy()
    counts = {"filled": 0, "left_for_adjudication": 0}
    for column, target in (
        ("label", "adjudicated_label"),
        ("reports_on_false_claim", "adjudicated_reports_on_false_claim"),
    ):
        if target not in out:
            out[target] = ""
        for i in out.index:
            a, b = parse_label(out.at[i, f"{column}_a"]), parse_label(out.at[i, f"{column}_b"])
            if out.at[i, target].strip():
                continue
            if a in (0, 1) and a == b:
                out.at[i, target] = str(a)
                counts["filled"] += 1
            elif a in (0, 1) and b in (0, 1):
                counts["left_for_adjudication"] += 1
    return out, counts


# ---------------------------------------------------------------------------
# claim evidence from disk
# ---------------------------------------------------------------------------
def embeddings_file_model(path: Path) -> str | None:
    """The model id that an embeddings file carries, or ``None`` when it carries none.

    The id is a ``model`` column with one value, or the ``model`` key of the Parquet
    schema metadata.
    """
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    meta = (schema.metadata or {}).get(b"model")
    if meta:
        return meta.decode("utf-8")
    if "model" in schema.names:
        values = pq.read_table(path, columns=["model"]).column("model").unique().to_pylist()
        values = [v for v in values if v]
        if len(values) == 1:
            return str(values[0])
        if len(values) > 1:
            return "; ".join(sorted(map(str, values)))
    return None


def claim_evidence(
    claim_rows: Sequence[dict[str, Any]],
    config: SamplingConfig,
    *,
    embeddings_path: Path | None = None,
    embedder_factory: Callable[[], tuple[Any | None, str | None]] | None = None,
    expected_model: str | None = None,
) -> ClaimEvidence:
    """Simhashes of the false claims, plus embeddings when the data host supplies them.

    The corpus records are not embedded here. The embedding path needs a Parquet file
    with ``id`` and ``embedding`` for corpus records, written by the model
    ``expected_model``. The claims (a few thousand short texts) are embedded with the
    model that ``embedder_factory`` returns. If the file names a model that differs from
    ``expected_model``, the path does not run (``embedding_model_mismatch``). A file with
    no model id is used, and the output says that the model is unverified. With no file,
    the path does not run (``embedding_unavailable``). An unavailable embedder gives its own
    reason code. The threshold is ``config.min_similarity``.
    """
    wanted = {label.lower() for label in config.claim_labels}
    claims = [r for r in claim_rows if str(r.get("label", "")).lower() in wanted]
    evidence = ClaimEvidence(
        hashes=[int(r["claim_simhash"]) for r in claims if r.get("claim_simhash")]
    )
    missing = len(claims) - len(evidence.hashes)
    if missing:
        evidence.degraded[CLAIM_WITHOUT_SIMHASH] = missing
        evidence.notes.append(
            f"{CLAIM_WITHOUT_SIMHASH}: {missing} of {len(claims)} false claims have no simhash "
            "and cannot match by simhash"
        )
    if embeddings_path is None or not claims:
        return evidence
    if embedder_factory is None or config.min_similarity is None:
        evidence.notes.append(
            "embedding path needs claim_match.min_similarity in configs/labelling.yaml"
        )
        return evidence
    if not Path(embeddings_path).exists():
        evidence.embedding_reason = EMBEDDING_UNAVAILABLE
        return evidence
    file_model = embeddings_file_model(Path(embeddings_path))
    if file_model is not None and expected_model and file_model != expected_model:
        evidence.embedding_reason = EMBEDDING_MODEL_MISMATCH
        evidence.notes.append(
            f"the embeddings file names model {file_model!r}, the claims would use "
            f"{expected_model!r}"
        )
        return evidence
    if file_model is not None and not expected_model:
        evidence.notes.append(
            f"no expected embedding model is configured, so the model {file_model!r} named by "
            "the embeddings file is not checked"
        )
    if file_model is None:
        evidence.notes.append(
            f"{EMBEDDING_MODEL_UNVERIFIED}: the embeddings file carries no model id, so "
            "this run cannot check that it matches the claim embedder"
        )
    embedder, reason = embedder_factory()
    if embedder is None:
        evidence.embedding_reason = reason or EMBEDDING_UNAVAILABLE
        return evidence
    from modeling.eval.claim_dedup import _embed_texts

    frame = pd.read_parquet(embeddings_path, columns=["id", "embedding"])
    evidence.record_ids = frame["id"].astype(str).tolist()
    evidence.record_matrix = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
    evidence.claim_matrix = _embed_texts(embedder, [str(r["claim"]) for r in claims])
    evidence.min_similarity = config.min_similarity
    evidence.embedding_reason = None
    return evidence
