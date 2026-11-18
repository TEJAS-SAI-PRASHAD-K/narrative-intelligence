"""A second, read-side claim dedup pass over multilingual embeddings.

``ingest.factcheck.collapse_duplicate_claims`` links claims whose ``claim_simhash``
values are within a small Hamming radius. A claim reworded past that radius, or
the same claim debunked in English by one publisher and in Hindi by another,
sits outside it and is counted more than once in the only evaluation set for
this case. This module links those claims by the cosine similarity of their
multilingual claim embeddings.

**The artifact stays lossless.** ``factchecks.parquet`` is never written. The
pass produces cluster ids, one link per collapsed row, and a count per pass, so
the evaluation harness can read the same claim once without deleting anything.

**Every pass that does not run says why.** A missing
``sentence-transformers``, a model that will not load, and a pass disabled in
config each produce a :class:`PassOutcome` with ``collapsed=None`` and a reason
code. None means not measured, which is not the same as a measured zero. The
evaluation report prints those outcomes, so an embedding pass that never ran is
visible in the artifact rather than looking like two passes that agreed.

**The threshold is a config decision.** It lives in ``configs/factcheck.yaml``
under ``claim_dedup`` and is provisional until a person labels the written
sample. ``embedding_precision`` reads the labelled sample and reports a number
only when there are labelled pairs; an unlabelled sample is
``not_measured``, never 0.

The harness imports this module only when it reads a claim universe, so
``modeling.eval.factcheck_eval`` keeps loading no model of its own. The
``ingest`` dependency is real but narrow: the simhash pass is reused as is, and
this module adds a second relation over it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ingest.factcheck import CLAIM_HAMMING_RADIUS, collapse_duplicate_claims
from ingest.title_predicate_eval import (
    LABELLER_KINDS,
    MEASURED_KINDS,
    cohen_kappa,
    wilson,
)
from modeling.config import ModelingSettings, load_yaml, module_config

log = logging.getLogger(__name__)

#: The read-side links artifact, beside the lossless Parquet file it describes.
CLAIM_LINKS_FILENAME = "claim_links.json"
#: The pairs a person labels to measure the embedding pass's precision.
HAND_CHECK_FILENAME = "claim_dedup_sample.csv"
LINKS_SCHEMA_VERSION = 1

#: Rows scored per similarity block. The comparison is still all pairs; the
#: block only bounds the size of the temporary (block, n) matrix.
_SIMILARITY_BLOCK = 512

#: Columns of the hand-check sample. The label and labeller columns are left blank on
#: write. ``is_same_claim`` is the adjudicated value. ``label_1`` and ``label_2`` hold the
#: two labellers' own answers, and ``labeller_kind`` says who they are.
SAMPLE_COLUMNS = (
    "record_id",
    "kept_record_id",
    "similarity",
    "label",
    "claim",
    "kept_label",
    "kept_claim",
    "labeller_kind",
    "labeller_1",
    "labeller_2",
    "label_1",
    "label_2",
    "is_same_claim",
)

#: The kind reported for a sample with no ``labeller_kind`` column. It is never measured.
KIND_UNKNOWN = "unknown"
#: The kind reported when the labelled rows name more than one kind.
KIND_MIXED = "mixed"

#: The words a report prints for each kind of labeller, after ``labeller: ``.
KIND_DESCRIPTIONS = {
    "human": "human",
    "model-reviewed": "model, reviewed by a person",
    "model-unreviewed": "model, unreviewed by a person",
    KIND_UNKNOWN: "unknown",
}
LABELLER_NOTES = {kind: f"labeller: {text}" for kind, text in KIND_DESCRIPTIONS.items()}

#: Cohen's kappa below this value earns a warning. It does not block a measured status.
KAPPA_WARN_BELOW = 0.6

TRUE_LABELS = frozenset({"y", "yes", "true", "1", "same"})
FALSE_LABELS = frozenset({"n", "no", "false", "0", "different"})


class DedupPass(str, Enum):
    """Which pass removed a row from the read-side view."""

    SIMHASH = "simhash"
    EMBEDDING = "embedding"


class PassStatus(str, Enum):
    """Whether a pass ran and what its result means."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    #: The artifact exists but no links file has been written for this data yet.
    NOT_RUN = "not_run"


class EmbeddingReason(str, Enum):
    """Why the embedding pass produced no link for a row or for the whole run."""

    #: The optional ``modeling`` extra is absent. Install it before running.
    DEPENDENCY_MISSING = "sentence_transformers_not_installed"
    #: The package is installed but the model could not be loaded.
    MODEL_UNAVAILABLE = "embedding_model_unavailable"
    #: The model's width does not fit ``embed.vector_dim``.
    DIMENSION_MISMATCH = "embedding_dimension_mismatch"
    #: The pass is switched off in ``configs/factcheck.yaml``.
    DISABLED = "disabled_in_config"
    #: The model loaded but encoding raised. The reason is logged with the error.
    ENCODE_FAILED = "embedding_encode_failed"
    #: The row's claim text is empty, so it has no embedding.
    NO_CLAIM_TEXT = "no_claim_text"
    #: The row is tier `title_predicate`, a verdict inferred from a headline. It is
    #: not embedded, so it cannot absorb a claim that a publisher stated.
    TIER_TITLE_PREDICATE = "tier_title_predicate"
    #: No links artifact was written for this data, so the eval cannot apply it.
    LINKS_ABSENT = "links_artifact_absent"


class PrecisionReason(str, Enum):
    """Why the embedding pass's precision is not a number."""

    SAMPLE_ABSENT = "sample_absent"
    SAMPLE_NOT_LABELLED = "sample_not_labelled"
    #: Labels exist, but the labeller is not (only) a person, so the number is provisional.
    LABELLER_NOT_HUMAN = "labeller_not_human"
    #: The sample has no ``labeller_kind`` column, so nobody recorded who labelled it.
    LABELLER_KIND_MISSING = "labeller_kind_missing"
    #: A labelled sample was kept and the new blank sample went to a side file.
    SAMPLE_HAS_LABELS = "sample_has_labels"
    #: Labelled pairs that the current embedding links do not contain.
    SAMPLE_PAIRS_STALE = "sample_pairs_stale"
    #: A human row lacks two different labeller names.
    LABELLER_NAMES_MISSING = "labeller_names_missing"
    #: No pair has both ``label_1`` and ``label_2``, so there is no agreement figure.
    NO_DOUBLE_LABELLED_PAIR = "no_double_labelled_pair"
    #: No labelled pair is at or above the threshold, so there is no precision.
    NO_PAIRS_AT_THRESHOLD = "no_pairs_at_threshold"
    #: A labelled pair has no similarity, so it cannot be placed against the threshold.
    SAMPLE_SIMILARITY_MISSING = "sample_similarity_missing"
    #: The sample cannot be read at all.
    SAMPLE_REFUSED = "sample_refused"


class PrecisionStatus(str, Enum):
    """Whether the hand-check sample settled the precision."""

    MEASURED = "measured"
    #: The sample is invalid. There is no number.
    REFUSED = "refused"
    #: A number from labels that a person did not give. It is not a measured precision.
    PROVISIONAL = "provisional"
    NOT_MEASURED = "not_measured"


class ClaimDedupConfigError(ValueError):
    """A claim dedup threshold or model that config does not settle."""


class ClaimLinksError(ValueError):
    """A claim links artifact that cannot be trusted."""


class ClaimLinksStale(ClaimLinksError):
    """The links describe different fact-check rows than the ones on disk."""


class ClaimSampleError(ValueError):
    """A hand-check sample whose labels or columns cannot be read."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClaimDedupConfig:
    """The embedding pass's knobs, all from config, none from code."""

    enabled: bool
    model_name: str | None
    min_similarity: float | None
    sample_size: int


def parse_claim_dedup_config(
    block: dict[str, Any] | None, embed_block: dict[str, Any] | None
) -> ClaimDedupConfig:
    """Validate one ``claim_dedup`` mapping, with the embedder config for defaults.

    An enabled pass without a threshold or a model is an error rather than a
    code default: a threshold invented in code cannot be audited, and a model
    name invented in code is how the wrong embedder gets used silently.
    """
    block = block or {}
    embed_block = embed_block or {}
    enabled = bool(block.get("enabled", True))
    model_name = str(block.get("model_name") or embed_block.get("claim_dedup_model") or "")
    raw_threshold = block.get("min_similarity")
    threshold = None if raw_threshold is None else float(raw_threshold)
    if enabled and threshold is None:
        raise ClaimDedupConfigError(
            "configs/factcheck.yaml has no `claim_dedup.min_similarity`. The threshold is a "
            "config decision; set it there rather than editing code."
        )
    if threshold is not None and not 0.0 < threshold <= 1.0:
        raise ClaimDedupConfigError(
            f"`claim_dedup.min_similarity` must be above 0 and at most 1, is {threshold}"
        )
    if enabled and not model_name:
        raise ClaimDedupConfigError(
            "configs/factcheck.yaml has no `claim_dedup.model_name` and `configs/models.yaml` "
            "has no `embed.claim_dedup_model`."
        )
    alternatives = [str(item) for item in (embed_block.get("alternatives") or [])]
    if model_name and alternatives and model_name not in alternatives:
        raise ClaimDedupConfigError(
            f"claim dedup model {model_name!r} is not listed under `embed.alternatives` in "
            f"configs/models.yaml. Add it there first, so the width and version stay auditable."
        )
    sample_size = int(block.get("sample_size", 200))
    if sample_size < 1:
        raise ClaimDedupConfigError(
            f"`claim_dedup.sample_size` must be at least 1, is {sample_size}"
        )
    return ClaimDedupConfig(
        enabled=enabled,
        model_name=model_name or None,
        min_similarity=threshold,
        sample_size=sample_size,
    )


def claim_dedup_config() -> ClaimDedupConfig:
    """The pass's config from ``configs/factcheck.yaml`` and ``configs/models.yaml``."""
    block = load_yaml("factcheck").get("claim_dedup")
    if block is not None and not isinstance(block, dict):
        raise ClaimDedupConfigError("configs/factcheck.yaml `claim_dedup` must be a mapping")
    return parse_claim_dedup_config(block, module_config("embed"))


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClaimLink:
    """One row that the read-side view collapsed, and the row it collapsed into."""

    record_id: str
    kept_record_id: str
    reason: str
    #: Cosine similarity for ``embedding``; ``None`` for ``simhash``.
    similarity: float | None = None
    #: Hamming distance for ``simhash``; ``None`` for ``embedding``.
    hamming_distance: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "record_id": self.record_id,
            "kept_record_id": self.kept_record_id,
            "reason": self.reason,
        }
        if self.similarity is not None:
            out["similarity"] = round(float(self.similarity), 6)
        if self.hamming_distance is not None:
            out["hamming_distance"] = int(self.hamming_distance)
        return out


@dataclass
class PassOutcome:
    """What one pass did, and why it did nothing when it did nothing."""

    name: str
    status: str
    #: Rows the pass was given. Not a claim about how many it read.
    considered: int
    #: ``None`` means not measured. A measured zero is ``0``.
    collapsed: int | None
    reason: str | None = None
    radius: int | None = None
    threshold: float | None = None
    model_name: str | None = None
    embedded: int | None = None
    skipped: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "considered": int(self.considered),
            "collapsed": None if self.collapsed is None else int(self.collapsed),
            "reason": self.reason,
        }
        for key in ("radius", "threshold", "model_name", "embedded"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.skipped:
            out["skipped"] = dict(sorted(self.skipped.items()))
        return out


@dataclass
class ClaimDedupResult:
    """One read-side view of the claim rows, plus what each pass removed."""

    #: One row per final cluster, sorted by ``record_id``.
    kept: list[dict[str, Any]]
    #: One entry per collapsed row, sorted by ``(reason, record_id)``.
    links: list[ClaimLink]
    #: Every input ``record_id`` mapped to its cluster's representative.
    clusters: dict[str, str]
    #: Keyed by ``DedupPass`` value.
    passes: dict[str, PassOutcome]
    #: Collapsed rows per pass, plus the existing diagnostics. A pass that did
    #: not run has no key here at all, so it cannot read as a measured zero.
    counts: dict[str, int]

    def outcome(self, name: DedupPass | str) -> PassOutcome | None:
        return self.passes.get(str(getattr(name, "value", name)))


def _as_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    as_row = getattr(row, "as_row", None)
    if callable(as_row):
        return dict(as_row())
    return dict(row)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _embed_texts(embedder: Any, texts: list[str]) -> np.ndarray:
    prepare = getattr(embedder, "prepare", None)
    if callable(prepare):
        texts = [prepare(text)[0] for text in texts]
    return np.asarray(embedder.embed_texts(texts), dtype=np.float32)


def default_embedder(
    model_name: str, settings: ModelingSettings | None = None
) -> tuple[Any | None, str | None]:
    """The configured multilingual embedder, or ``(None, reason_code)``.

    Lazy on purpose: the evaluation harness imports this module with no
    ``sentence-transformers`` installed, and a missing optional dependency is a
    reason code on one pass, not an import error for the report.
    """
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is None:
        log.error(
            "claim dedup embedding pass unavailable: sentence-transformers is not installed; "
            "install the 'modeling' extra"
        )
        return None, EmbeddingReason.DEPENDENCY_MISSING.value
    from modeling.text.embed import Embedder, EmbeddingDimensionMismatch

    try:
        embedder = Embedder(settings, model_name=model_name)
        if not embedder.load():
            return None, EmbeddingReason.MODEL_UNAVAILABLE.value
    except EmbeddingDimensionMismatch as exc:
        log.error("claim dedup embedder refused: %s", exc)
        return None, EmbeddingReason.DIMENSION_MISMATCH.value
    return embedder, None


def _link_pairs(matrix: np.ndarray, threshold: float) -> _UnionFind:
    """Union every pair of rows whose cosine similarity reaches ``threshold``."""
    size = len(matrix)
    groups = _UnionFind(size)
    for start in range(0, size, _SIMILARITY_BLOCK):
        block = matrix[start : start + _SIMILARITY_BLOCK] @ matrix.T
        left, right = np.where(block >= threshold)
        for offset, column in zip(left, right, strict=True):
            i, j = start + int(offset), int(column)
            if i < j:
                groups.union(i, j)
    return groups


def run_claim_dedup(
    rows: Iterable[Any],
    *,
    config: ClaimDedupConfig | None = None,
    embedder: Any | None = None,
    merge_tier4: bool = False,
) -> ClaimDedupResult:
    """Link claims by simhash, then by multilingual embedding similarity.

    The simhash pass runs first and unchanged. The embedding pass runs over the
    rows that pass left, so a pair already linked by simhash is never counted
    twice. The representative of a final cluster is the smallest ``record_id``
    across both passes, which keeps the read-side view independent of input
    order and of ``extracted_at``.

    A row whose claim text is empty is left to the simhash pass and counted
    under ``no_claim_text``. A pass that cannot run (dependency absent, model
    unavailable, disabled in config) returns ``collapsed=None`` and a reason
    code; the simhash result is still returned.

    ``merge_tier4`` is true when tier 4 counts in the primary evaluation. It must equal
    the setting that the evaluation reads with, or the links artifact is stale.
    """
    config = config or claim_dedup_config()
    items = [_as_row(row) for row in rows]

    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        record_id = str(item.get("record_id") or "")
        if not record_id:
            raise ValueError("a claim row has no record_id; the link cannot be keyed")
        if record_id in by_id:
            raise ValueError(f"claim record_id {record_id!r} appears more than once")
        by_id[record_id] = item

    simhash = collapse_duplicate_claims(items, merge_tier4=merge_tier4)
    kept_ids = [str(row.get("record_id") or "") for row in simhash.kept]
    kept_by_id = {str(row.get("record_id") or ""): row for row in simhash.kept}
    simhash_rep = {record_id: record_id for record_id in kept_ids}
    hamming_to_rep = {record_id: 0 for record_id in kept_ids}
    for record_id, representative, distance in simhash.collapsed:
        simhash_rep[record_id] = representative
        hamming_to_rep[record_id] = int(distance)

    simhash_outcome = PassOutcome(
        name=DedupPass.SIMHASH.value,
        status=PassStatus.OK.value,
        considered=len(items),
        collapsed=len(simhash.collapsed),
        radius=CLAIM_HAMMING_RADIUS,
    )
    base_counts = {
        DedupPass.SIMHASH.value: len(simhash.collapsed),
        "unhashed_kept": int(simhash.counts.get("unhashed_kept", 0)),
        "label_conflict": int(simhash.counts.get("label_conflict", 0)),
    }

    embedding_outcome: PassOutcome
    candidate_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    matrix: np.ndarray | None = None
    index_of: dict[str, int] = {}

    if not config.enabled:
        embedding_outcome = PassOutcome(
            name=DedupPass.EMBEDDING.value,
            status=PassStatus.DISABLED.value,
            considered=len(kept_ids),
            collapsed=None,
            reason=EmbeddingReason.DISABLED.value,
            model_name=config.model_name,
        )
    else:
        for record_id in kept_ids:
            if str(kept_by_id[record_id].get("tier") or "").strip() == "title_predicate":
                skipped[EmbeddingReason.TIER_TITLE_PREDICATE.value] = (
                    skipped.get(EmbeddingReason.TIER_TITLE_PREDICATE.value, 0) + 1
                )
                continue
            text = str(kept_by_id[record_id].get("claim") or "").strip()
            if not text:
                skipped[EmbeddingReason.NO_CLAIM_TEXT.value] = (
                    skipped.get(EmbeddingReason.NO_CLAIM_TEXT.value, 0) + 1
                )
                continue
            candidate_rows.append(kept_by_id[record_id])
            index_of[record_id] = len(candidate_rows) - 1

        if not candidate_rows:
            embedding_outcome = PassOutcome(
                name=DedupPass.EMBEDDING.value,
                status=PassStatus.OK.value,
                considered=len(kept_ids),
                collapsed=0,
                model_name=config.model_name,
                threshold=config.min_similarity,
                embedded=0,
                skipped=skipped,
            )
            counts = {**base_counts, DedupPass.EMBEDDING.value: 0}
            return ClaimDedupResult(
                kept=simhash.kept,
                links=[
                    ClaimLink(
                        record_id=record_id,
                        kept_record_id=simhash_rep[record_id],
                        reason=DedupPass.SIMHASH.value,
                        hamming_distance=hamming_to_rep[record_id],
                    )
                    for record_id in sorted(by_id)
                    if record_id != simhash_rep[record_id]
                ],
                clusters=dict(sorted(simhash_rep.items())),
                passes={
                    DedupPass.SIMHASH.value: simhash_outcome,
                    DedupPass.EMBEDDING.value: embedding_outcome,
                },
                counts=dict(sorted(counts.items())),
            )

        active_embedder, reason = (embedder, None) if embedder is not None else default_embedder(
            str(config.model_name), None
        )
        if active_embedder is None:
            embedding_outcome = PassOutcome(
                name=DedupPass.EMBEDDING.value,
                status=PassStatus.UNAVAILABLE.value,
                considered=len(kept_ids),
                collapsed=None,
                reason=reason or EmbeddingReason.MODEL_UNAVAILABLE.value,
                model_name=config.model_name,
                threshold=config.min_similarity,
                skipped=skipped,
            )
        else:
            encode_failed = False
            try:
                raw = _embed_texts(
                    active_embedder,
                    [str(kept_by_id[record_id].get("claim") or "") for record_id in index_of],
                )
            except Exception as exc:  # noqa: BLE001 - an encode failure is one pass's outage
                log.error("claim dedup embedding pass failed during encoding: %s", exc)
                encode_failed = True
                raw = np.zeros((0, 0), dtype=np.float32)
            if encode_failed:
                matrix = None
            elif raw.ndim != 2 or len(raw) != len(index_of):
                raise ValueError(
                    "the claim dedup embedder returned a matrix of shape "
                    f"{raw.shape}; expected ({len(index_of)}, dim)"
                )
            else:
                from modeling.text.embed import l2_normalize

                matrix = l2_normalize(raw)
            if matrix is None:
                embedding_outcome = PassOutcome(
                    name=DedupPass.EMBEDDING.value,
                    status=PassStatus.UNAVAILABLE.value,
                    considered=len(kept_ids),
                    collapsed=None,
                    reason=EmbeddingReason.ENCODE_FAILED.value,
                    model_name=config.model_name,
                    threshold=config.min_similarity,
                    skipped=skipped,
                )
            else:
                threshold = float(config.min_similarity or 0.0)
                groups = _link_pairs(matrix, threshold)
                rep_by_root: dict[int, str] = {}
                for record_id, index in index_of.items():
                    root = groups.find(index)
                    held = rep_by_root.get(root)
                    if held is None or record_id < held:
                        rep_by_root[root] = record_id
                final_rep_for_simhash = dict(simhash_rep)
                for record_id in index_of:
                    final_rep_for_simhash[record_id] = rep_by_root[
                        groups.find(index_of[record_id])
                    ]
                clusters = {
                    record_id: final_rep_for_simhash[simhash_rep[record_id]]
                    for record_id in by_id
                }
                final_kept_ids = {
                    record_id for record_id in kept_ids if clusters[record_id] == record_id
                }
                embedding_count = len(items) - len(final_kept_ids) - len(simhash.collapsed)

                def similarity(record_id: str, representative: str) -> float | None:
                    left = index_of.get(record_id)
                    right = index_of.get(representative)
                    if left is None or right is None:
                        return None
                    return float(matrix[left] @ matrix[right])

                links: list[ClaimLink] = []
                for record_id in by_id:
                    representative = clusters[record_id]
                    if record_id == representative:
                        continue
                    if simhash_rep[record_id] == record_id:
                        links.append(
                            ClaimLink(
                                record_id=record_id,
                                kept_record_id=representative,
                                reason=DedupPass.EMBEDDING.value,
                                similarity=similarity(record_id, representative),
                            )
                        )
                    else:
                        links.append(
                            ClaimLink(
                                record_id=record_id,
                                kept_record_id=representative,
                                reason=DedupPass.SIMHASH.value,
                                hamming_distance=hamming_to_rep[record_id],
                            )
                        )
                links.sort(key=lambda link: (link.reason, link.record_id))
                embedding_outcome = PassOutcome(
                    name=DedupPass.EMBEDDING.value,
                    status=PassStatus.OK.value,
                    considered=len(kept_ids),
                    collapsed=embedding_count,
                    model_name=config.model_name,
                    threshold=threshold,
                    embedded=len(index_of),
                    skipped=skipped,
                )
                kept = [kept_by_id[record_id] for record_id in sorted(final_kept_ids)]
                counts = {
                    **base_counts,
                    DedupPass.EMBEDDING.value: embedding_count,
                }
                return ClaimDedupResult(
                    kept=kept,
                    links=links,
                    clusters=dict(sorted(clusters.items())),
                    passes={
                        DedupPass.SIMHASH.value: simhash_outcome,
                        DedupPass.EMBEDDING.value: embedding_outcome,
                    },
                    counts=dict(sorted(counts.items())),
                )

    # Reached only when the embedding pass produced no links at all.
    links = [
        ClaimLink(
            record_id=record_id,
            kept_record_id=simhash_rep[record_id],
            reason=DedupPass.SIMHASH.value,
            hamming_distance=hamming_to_rep[record_id],
        )
        for record_id in sorted(by_id)
        if record_id != simhash_rep[record_id]
    ]
    return ClaimDedupResult(
        kept=simhash.kept,
        links=links,
        clusters=dict(sorted(simhash_rep.items())),
        passes={
            DedupPass.SIMHASH.value: simhash_outcome,
            DedupPass.EMBEDDING.value: embedding_outcome,
        },
        counts=dict(sorted(base_counts.items())),
    )


# ---------------------------------------------------------------------------
# The links artifact
# ---------------------------------------------------------------------------
def claim_links_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "factchecks" / CLAIM_LINKS_FILENAME


def hand_check_sample_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "factchecks" / HAND_CHECK_FILENAME


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, write: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    os.close(fd)
    try:
        write(Path(tmp_name))
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_claim_links(
    result: ClaimDedupResult,
    path: Path,
    *,
    source: Path | None = None,
    sample_path: Path | None = None,
) -> Path:
    """Write the links, cluster ids and per-pass counts as JSON.

    The file is written to a temporary sibling and renamed, so a reader sees the
    old artifact or the new one. ``source`` is the Parquet file the pass read;
    its hash lets the evaluation harness refuse an artifact that no longer
    describes the rows on disk. ``sample_path`` points at the pairs a person
    labels to settle the embedding threshold.
    """
    payload: dict[str, Any] = {
        "schema_version": LINKS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(result.clusters),
        "passes": {name: outcome.as_dict() for name, outcome in result.passes.items()},
        "counts": dict(sorted(result.counts.items())),
        "clusters": dict(sorted(result.clusters.items())),
        "links": [link.as_dict() for link in result.links],
    }
    if sample_path is not None:
        payload["sample"] = str(Path(sample_path).resolve())
    if source is not None:
        payload["source"] = {
            "file": Path(source).name,
            "sha256": sha256_file(Path(source)),
        }

    def write(target: Path) -> None:
        target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    _atomic_write(Path(path), write)
    return Path(path)


def read_claim_links(path: Path) -> dict[str, Any] | None:
    """The links artifact, or ``None`` when none has been written.

    A malformed artifact raises: a pass whose output cannot be checked is not a
    pass whose output may be ignored.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClaimLinksError(f"{path} is not readable JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ClaimLinksError(f"{path} is not a JSON object")
    if payload.get("schema_version") != LINKS_SCHEMA_VERSION:
        raise ClaimLinksError(
            f"{path} has schema_version {payload.get('schema_version')!r}; "
            f"expected {LINKS_SCHEMA_VERSION}"
        )
    for key in ("passes", "clusters", "links", "counts"):
        if key not in payload:
            raise ClaimLinksError(f"{path} has no {key!r}")
    if not isinstance(payload["clusters"], dict) or not isinstance(payload["links"], list):
        raise ClaimLinksError(f"{path}: `clusters` must be an object and `links` a list")
    if not isinstance(payload["passes"], dict) or not isinstance(payload["counts"], dict):
        raise ClaimLinksError(f"{path}: `passes` and `counts` must be objects")
    return payload


def ensure_source_matches(links: dict[str, Any], source: Path) -> None:
    """Refuse a links artifact whose source hash is not the file on disk."""
    recorded = (links.get("source") or {}).get("sha256")
    source = Path(source)
    if not recorded or not source.exists():
        return
    actual = sha256_file(source)
    if actual != recorded:
        raise ClaimLinksStale(
            f"the claim links artifact was built from a different factchecks file "
            f"(sha256 {recorded[:12]}..., disk has {actual[:12]}...). Rerun "
            "`python -m modeling.cli dedup-claims` before evaluating."
        )


def clusters_for_ids(
    clusters: dict[str, str], record_ids: Iterable[str], *, path: Path | None = None
) -> dict[str, str]:
    """The clusters, refused unless they cover exactly ``record_ids`` and are canonical."""
    ids = {str(record_id) for record_id in record_ids}
    missing = sorted(ids - set(clusters))
    extra = sorted(set(clusters) - ids)
    where = f" in {path}" if path else ""
    if missing or extra:
        shown_missing = ", ".join(missing[:3])
        shown_extra = ", ".join(extra[:3])
        raise ClaimLinksStale(
            f"the claim links artifact{where} does not describe the current fact-check rows: "
            f"{len(missing)} row(s) have no cluster (for example {shown_missing}) and "
            f"{len(extra)} cluster row(s) are absent from the data (for example {shown_extra}). "
            "Rerun `python -m modeling.cli dedup-claims`."
        )
    for record_id, representative in clusters.items():
        if representative not in ids:
            raise ClaimLinksStale(
                f"the claim links artifact{where} points {record_id!r} at {representative!r}, "
                "which is not a record in the data"
            )
        if clusters.get(representative) != representative:
            raise ClaimLinksStale(
                f"the claim links artifact{where} is not canonical: {record_id!r} -> "
                f"{representative!r}, but {representative!r} is not its own representative"
            )
    return {record_id: clusters[record_id] for record_id in sorted(ids)}


def final_rows_and_embedding_collapses(
    kept_rows: list[dict[str, Any]],
    collapsed: Iterable[tuple[str, str, int]],
    clusters: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Apply final clusters to a simhash read.

    Returns the rows still kept and the simhash-kept rows that the embedding
    pass removed. Rows already collapsed by simhash stay under that reason, so
    the two reasons never double count one row in the evaluation report.
    """
    kept_ids = {str(row["record_id"]) for row in kept_rows}
    for representative in set(clusters.values()):
        if representative not in kept_ids:
            raise ClaimLinksStale(
                f"the claim links artifact names {representative!r} as a kept row, but it is "
                "not one after the simhash pass"
            )
    final_rows: list[dict[str, Any]] = []
    embedding_collapsed: dict[str, str] = {}
    for row in kept_rows:
        record_id = str(row["record_id"])
        representative = clusters.get(record_id, record_id)
        if representative == record_id:
            final_rows.append(row)
        else:
            embedding_collapsed[record_id] = representative
    collapsed_ids = {str(record_id) for record_id, _, _ in collapsed}
    overlap = sorted(set(embedding_collapsed) & collapsed_ids)
    if overlap:
        raise ClaimLinksStale(
            "the claim links artifact places a simhash-collapsed row in the embedding set: "
            + ", ".join(overlap[:3])
        )
    final_rows.sort(key=lambda row: str(row["record_id"]))
    return final_rows, embedding_collapsed


# ---------------------------------------------------------------------------
# Hand-check sample and precision
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingPrecision:
    """The embedding pass's precision on labelled pairs, or why it has none.

    Only labels from a person give status ``measured``. A number with any open
    reason is ``provisional``. ``reasons`` lists every open reason code and
    ``reason`` is the first one.
    """

    status: str
    #: ``None`` means not measured, never a made-up zero. It counts only labelled pairs at
    #: or above ``threshold``.
    precision: float | None
    #: Labelled pairs that ``precision`` counts, and the same-claim pairs among them.
    n_labelled: int
    n_same: int
    sample: str
    reason: str | None = None
    #: Wilson 95 percent interval on ``precision``. It treats pairs as independent, and the
    #: sample is systematic over the similarity range, so the interval is approximate.
    ci_low: float | None = None
    ci_high: float | None = None
    #: Cohen's kappa and raw agreement of ``label_1`` and ``label_2``, over the rows where
    #: both are filled. ``None`` means not computed.
    kappa: float | None = None
    agreement: float | None = None
    n_double_labelled: int = 0
    labeller_kind: str | None = None
    #: Labelled rows per kind.
    kind_counts: dict[str, int] = field(default_factory=dict)
    #: Labelled pairs that are not in the current embedding links. ``None`` means not checked.
    n_stale: int | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: The similarity threshold of the run that made the links, and where it was read.
    threshold: float | None = None
    threshold_source: str | None = None
    #: Labelled pairs below the threshold. They are in the links through a chain of
    #: pairs, so a person can label them, but they are not counted in ``precision``.
    n_below: int = 0
    n_same_below: int = 0
    #: Same-claim pairs below the threshold over all same-claim labelled pairs. It covers
    #: only the sampled links, so it is not the share of all duplicates that the threshold
    #: leaves out.
    same_below_share: float | None = None
    #: Labelled pairs with no usable similarity when a threshold applies.
    n_no_similarity: int = 0
    #: Precision per similarity band, each with a Wilson interval.
    bands: tuple[dict[str, Any], ...] = ()
    #: The message of a refused sample.
    error: str | None = None
    #: A note about which sample file was read.
    sample_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "precision": self.precision,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_labelled": int(self.n_labelled),
            "n_same": int(self.n_same),
            "kappa": self.kappa,
            "agreement": self.agreement,
            "n_double_labelled": int(self.n_double_labelled),
            "labeller_kind": self.labeller_kind,
            "kind_counts": dict(self.kind_counts),
            "n_stale": self.n_stale,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "n_below": int(self.n_below),
            "n_same_below": int(self.n_same_below),
            "same_below_share": self.same_below_share,
            "n_no_similarity": int(self.n_no_similarity),
            "bands": [dict(band) for band in self.bands],
            "sample": self.sample,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "error": self.error,
            "sample_note": self.sample_note,
        }


def _labeller_text(precision: dict[str, Any]) -> str:
    """The text after ``labeller: `` for a precision block."""
    kind = precision.get("labeller_kind")
    if kind == KIND_MIXED:
        parts = ", ".join(
            f"{KIND_DESCRIPTIONS.get(name, name)} {count}"
            for name, count in sorted((precision.get("kind_counts") or {}).items())
        )
        return f"mixed ({parts})"
    return KIND_DESCRIPTIONS.get(str(kind), str(kind))


def describe_precision(precision: dict[str, Any]) -> str:
    """One line of text for a precision block, for the CLI and the report.

    A number that is not measured always carries its labeller note and its reason
    codes, so a model label cannot read as a measured precision. A stale count shows
    on every line where it is not zero.
    """
    status = precision.get("status")
    n_stale = precision.get("n_stale") or 0
    n_all = (
        (precision.get("n_labelled") or 0)
        + (precision.get("n_below") or 0)
        + (precision.get("n_no_similarity") or 0)
    )
    stale = (
        f" {n_stale} of {n_all} labelled pairs are not in the current "
        f"embedding links (reason code {PrecisionReason.SAMPLE_PAIRS_STALE.value})."
        if n_stale
        else ""
    )
    if status == PrecisionStatus.REFUSED.value:
        return (
            f"embedding precision: refused (reason code "
            f"{precision.get('reason') or PrecisionReason.SAMPLE_REFUSED.value}). "
            f"{precision.get('error')}"
        )
    reasons = list(precision.get("reasons") or [])
    threshold = precision.get("threshold")
    where = f"at or above similarity {threshold:g}" if threshold is not None else "labelled"
    below = ""
    if threshold is not None:
        below = (
            f" {precision.get('n_below') or 0} labelled pairs below {threshold:g} are not "
            f"counted ({precision.get('n_same_below') or 0} of them are the same claim)."
        )
        if precision.get("same_below_share") is not None:
            below += (
                f" {precision['same_below_share']:.3f} of the same-claim pairs in the sample "
                "are below the threshold."
            )
        if precision.get("n_no_similarity"):
            below += f" {precision['n_no_similarity']} labelled pairs have no similarity."
        if precision.get("threshold_source") == "config":
            below += " The threshold is from the config, not from the links file."
    if precision.get("precision") is None:
        reason = precision.get("reason") or "not_measured"
        return f"embedding precision: not measured (reason code {reason}).{below}{stale}"
    text = (
        f"embedding precision: {float(precision['precision']):.3f} "
        f"({precision['n_same']} of {precision['n_labelled']} labelled pairs {where} "
        "are the same claim)"
    )
    if precision.get("ci_low") is not None:
        text += f", Wilson 95% interval [{precision['ci_low']:.3f}, {precision['ci_high']:.3f}]"
    if precision.get("kappa") is not None:
        text += (
            f", kappa {precision['kappa']:.3f}, agreement {precision['agreement']:.3f} "
            f"on {precision['n_double_labelled']} double-labelled pairs"
        )
    elif precision.get("n_double_labelled"):
        text += f", agreement {precision['agreement']:.3f}, kappa undefined"
    note = f"labeller: {_labeller_text(precision)}"
    if status == PrecisionStatus.MEASURED.value:
        text = f"{text}. {note}.{below}{stale}"
    else:
        codes = ", ".join(reasons or [str(precision.get("reason"))])
        text = (
            f"{text}. PROVISIONAL, not a measured precision. {note}. "
            f"Reason code {codes}.{below}{stale}"
        )
    for warning in precision.get("warnings") or []:
        text += f" Warning: {warning}"
    return text


def write_hand_check_sample(
    result: ClaimDedupResult,
    rows: Iterable[Any],
    path: Path,
    *,
    size: int = 200,
) -> Path:
    """Write embedding-linked pairs for a person to label, spread across similarity.

    Only the embedding pass is sampled: the simhash pass already has its radius
    and its distance returned. Pairs are ordered by similarity and taken at
    evenly spaced positions, so the sample covers the range rather than only the
    most obvious duplicates. A blank ``is_same_claim`` is the signal that
    nothing has been labelled yet. The labeller columns are blank too.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = _as_row(row)
        by_id[str(item.get("record_id") or "")] = item
    links = [link for link in result.links if link.reason == DedupPass.EMBEDDING.value]
    ordered = sorted(
        links,
        key=lambda link: (
            float("inf") if link.similarity is None else link.similarity,
            link.record_id,
        ),
    )
    if size > 0 and len(ordered) > size:
        if size == 1:
            positions = [len(ordered) // 2]
        else:
            positions = sorted(
                {round(index * (len(ordered) - 1) / (size - 1)) for index in range(size)}
            )
        ordered = [ordered[position] for position in positions]

    def write(target: Path) -> None:
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_COLUMNS), lineterminator="\n")
            writer.writeheader()
            for link in ordered:
                kept = by_id.get(link.kept_record_id, {})
                row = by_id.get(link.record_id, {})
                writer.writerow(
                    {
                        "record_id": link.record_id,
                        "kept_record_id": link.kept_record_id,
                        "similarity": "" if link.similarity is None else f"{link.similarity:.6f}",
                        "label": row.get("label") or "",
                        "claim": row.get("claim") or "",
                        "kept_label": kept.get("label") or "",
                        "kept_claim": kept.get("claim") or "",
                        "labeller_kind": "",
                        "labeller_1": "",
                        "labeller_2": "",
                        "label_1": "",
                        "label_2": "",
                        "is_same_claim": "",
                    }
                )

    _atomic_write(Path(path), write)
    return Path(path)


#: Columns that make a sample count as labelled. A prefilled ``labeller_kind`` does not.
LABEL_COLUMNS = ("is_same_claim", "label_1", "label_2")
SIDE_SAMPLE_FILENAME = "claim_dedup_sample.new.csv"


def sample_has_labels(path: Path) -> bool:
    """True when any row of the sample at ``path`` has a filled label column.

    A prefilled ``labeller_kind`` is normal and does not count. A file with no
    ``is_same_claim`` column is an error, not a blank sample, so the caller cannot
    overwrite it by mistake.
    """
    if not Path(path).exists():
        return False
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "is_same_claim" not in (reader.fieldnames or ()):
            raise ClaimSampleError(
                f"{path}: no is_same_claim column; the file is not replaced. Fix it or move it"
            )
        for row in reader:
            if any(str(row.get(column) or "").strip() for column in LABEL_COLUMNS):
                return True
    return False


def side_sample_path(sample_path: Path) -> Path:
    """The file that receives a new blank sample when the main sample holds labels."""
    return Path(sample_path).with_name(SIDE_SAMPLE_FILENAME)


def embedding_pairs(links: Iterable[Any]) -> set[tuple[str, str]]:
    """The ``(record_id, kept_record_id)`` pairs of the embedding links.

    Items are :class:`ClaimLink` objects or the dicts of ``claim_links.json``.
    """
    pairs: set[tuple[str, str]] = set()
    for link in links:
        item = link.as_dict() if hasattr(link, "as_dict") else link
        if item.get("reason") == DedupPass.EMBEDDING.value:
            pairs.add((str(item["record_id"]), str(item["kept_record_id"])))
    return pairs


def _yes_no(raw: str) -> bool | None:
    """True for a yes label, False for a no label, ``None`` for anything else."""
    value = raw.strip().lower()
    if value in TRUE_LABELS:
        return True
    if value in FALSE_LABELS:
        return False
    return None


def _empty(path: Path, reason: PrecisionReason) -> EmbeddingPrecision:
    return EmbeddingPrecision(
        status=PrecisionStatus.NOT_MEASURED.value,
        precision=None,
        n_labelled=0,
        n_same=0,
        sample=str(path),
        reason=reason.value,
        reasons=(reason.value,),
    )


#: Band edges above the threshold. The last band is closed at 1.0.
_BAND_EDGES = (0.90, 0.95)


def band_edges(threshold: float | None) -> list[float]:
    """The similarity band edges from ``threshold`` (or 0) up to 1.0."""
    start = 0.0 if threshold is None else round(threshold, 6)
    return [start, *[edge for edge in _BAND_EDGES if edge > start], 1.0]


def _band_index(edges: list[float], similarity: float) -> int | None:
    """The band that holds ``similarity``. A value above 1.0 from rounding joins the last."""
    if similarity < edges[0]:
        return None
    for index in range(len(edges) - 1):
        if similarity < edges[index + 1]:
            return index
    return len(edges) - 2


def _band(low: float, high: float, n: int, n_same: int) -> dict[str, Any]:
    closing = "]" if high >= 1.0 else ")"
    out: dict[str, Any] = {
        "label": f"[{low:.2f}, {high:.2f}{closing}",
        "low": low,
        "high": high,
        "n": n,
        "n_same": n_same,
        "precision": None,
        "ci_low": None,
        "ci_high": None,
    }
    if n:
        out["precision"] = n_same / n
        out["ci_low"], out["ci_high"] = wilson(n_same / n, n)
    return out


def _similarity(raw: Any) -> float | None:
    """The sample's similarity as a float, or ``None`` when it is blank or not a number."""
    try:
        value = float(str(raw or "").strip())
    except ValueError:
        return None
    return value if value == value else None


def embedding_precision(
    sample_path: Path,
    *,
    pairs: set[tuple[str, str]] | None = None,
    threshold: float | None = None,
    threshold_source: str | None = None,
) -> EmbeddingPrecision:
    """Precision on the labelled hand-check pairs, with the labeller kind.

    A blank ``is_same_claim`` is not a "no". The function refuses the whole sample
    with :class:`ClaimSampleError` when a filled row has a label that is not yes or
    no, has no valid ``labeller_kind``, names one labeller twice, has a blank
    ``is_same_claim`` with a filled ``label_1`` or ``label_2``, or has an
    ``is_same_claim`` that differs from two labels that agree. A sample with no
    ``labeller_kind`` column is an older sample. It reads as kind ``unknown`` with the
    reason ``labeller_kind_missing``.

    The status is ``measured`` only when every labelled row is from a person with two
    different labeller names, at least one pair has both labels, and every labelled
    pair is in ``pairs``. ``pairs`` is the set of current embedding links, or ``None``
    when the check does not apply. Any other case is ``provisional`` with reason codes.

    The sample holds every embedding link, and a link joins a row to the representative of
    its cluster. A row can join through a chain, so its similarity to the representative is
    below ``threshold``. Such a pair is in the links and is not stale. Precision counts only
    labelled pairs with similarity at or above ``threshold``, because the threshold is the
    claim that the pass makes. Pairs below it are counted apart, as ``n_below``. The
    similarity in the sample has six decimals, so the threshold is rounded the same way.
    With ``threshold`` set to ``None``, every labelled pair counts.
    """
    path = Path(sample_path)
    if not path.exists():
        return _empty(path, PrecisionReason.SAMPLE_ABSENT)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if "is_same_claim" not in header:
            raise ClaimSampleError(
                f"{path}: no is_same_claim column; the hand-check sample cannot be labelled"
            )
        has_kind_column = "labeller_kind" in header
        n_total = 0
        n_labelled = 0
        n_same = 0
        n_below = 0
        n_same_below = 0
        n_no_similarity = 0
        n_stale = 0
        edges = band_edges(threshold)
        band_n = [0] * (len(edges) - 1)
        band_same = [0] * (len(edges) - 1)
        cut = None if threshold is None else round(threshold, 6)
        kinds: dict[str, int] = {}
        names_missing = False
        pair_1: list[str] = []
        pair_2: list[str] = []
        for line, row in enumerate(reader, start=2):
            raw = str(row.get("is_same_claim") or "").strip().lower()
            answers = []
            for column in ("label_1", "label_2"):
                text = str(row.get(column) or "").strip()
                value = _yes_no(text) if text else None
                if text and value is None:
                    raise ClaimSampleError(
                        f"{path} line {line}: {column} is {text!r}; use yes or no"
                    )
                answers.append(value)
            if not raw:
                if any(value is not None for value in answers):
                    raise ClaimSampleError(
                        f"{path} line {line}: label_1 or label_2 is filled but is_same_claim "
                        "is blank; this is a half-filled row. Fill is_same_claim or clear the row"
                    )
                continue
            verdict = _yes_no(raw)
            if verdict is None:
                raise ClaimSampleError(
                    f"{path} line {line}: is_same_claim is {raw!r}; use yes or no"
                )
            if answers[0] is not None and answers[0] == answers[1] and answers[0] != verdict:
                raise ClaimSampleError(
                    f"{path} line {line}: is_same_claim disagrees with label_1 and label_2, "
                    "which agree with each other. Correct the row"
                )
            n_total += 1
            similarity = _similarity(row.get("similarity"))
            if cut is None:
                counted = True
            elif similarity is None:
                n_no_similarity += 1
                counted = False
            elif similarity >= cut:
                counted = True
            else:
                counted = False
                n_below += 1
                n_same_below += int(verdict)
            if counted:
                n_labelled += 1
                n_same += int(verdict)
                if similarity is not None:
                    index = _band_index(edges, similarity)
                    if index is not None:
                        band_n[index] += 1
                        band_same[index] += int(verdict)
            kind = KIND_UNKNOWN
            if has_kind_column:
                kind = str(row.get("labeller_kind") or "").strip()
                if kind not in LABELLER_KINDS:
                    raise ClaimSampleError(
                        f"{path} line {line}: labeller_kind is {kind!r}; "
                        f"use one of {list(LABELLER_KINDS)}"
                    )
            kinds[kind] = kinds.get(kind, 0) + 1
            first = str(row.get("labeller_1") or "").strip().casefold()
            second = str(row.get("labeller_2") or "").strip().casefold()
            if first and first == second:
                raise ClaimSampleError(
                    f"{path} line {line}: labeller_1 and labeller_2 are the same labeller "
                    f"({first!r}); two labellers must be two different names"
                )
            if kind in MEASURED_KINDS and not (first and second):
                names_missing = True
            if answers[0] is not None and answers[1] is not None:
                pair_1.append("yes" if answers[0] else "no")
                pair_2.append("yes" if answers[1] else "no")
            if pairs is not None:
                key = (
                    str(row.get("record_id") or "").strip(),
                    str(row.get("kept_record_id") or "").strip(),
                )
                if key not in pairs:
                    n_stale += 1
    if n_total == 0:
        return _empty(path, PrecisionReason.SAMPLE_NOT_LABELLED)
    if len(kinds) == 1:
        kind = next(iter(kinds))
    else:
        kind = KIND_MIXED
    precision = n_same / n_labelled if n_labelled else None
    ci_low, ci_high = wilson(precision, n_labelled) if n_labelled else (None, None)
    n_same_all = n_same + n_same_below
    same_below_share = (
        n_same_below / n_same_all if threshold is not None and n_same_all else None
    )
    bands = tuple(
        _band(edges[i], edges[i + 1], band_n[i], band_same[i]) for i in range(len(band_n))
    )
    agreement = None
    kappa = cohen_kappa(pair_1, pair_2) if pair_1 else None
    if pair_1:
        agreement = sum(a == b for a, b in zip(pair_1, pair_2, strict=True)) / len(pair_1)

    reasons: list[str] = []
    if not n_labelled:
        reasons.append(PrecisionReason.NO_PAIRS_AT_THRESHOLD.value)
    if n_no_similarity:
        reasons.append(PrecisionReason.SAMPLE_SIMILARITY_MISSING.value)
    if n_stale:
        reasons.append(PrecisionReason.SAMPLE_PAIRS_STALE.value)
    if kind == KIND_UNKNOWN:
        reasons.append(PrecisionReason.LABELLER_KIND_MISSING.value)
    elif kind not in MEASURED_KINDS:
        reasons.append(PrecisionReason.LABELLER_NOT_HUMAN.value)
    if names_missing:
        reasons.append(PrecisionReason.LABELLER_NAMES_MISSING.value)
    if not pair_1:
        reasons.append(PrecisionReason.NO_DOUBLE_LABELLED_PAIR.value)
    warnings: list[str] = []
    if kappa is not None and kappa < KAPPA_WARN_BELOW:
        warnings.append(
            f"kappa {kappa:.3f} is below {KAPPA_WARN_BELOW}. The two labellers disagree "
            "often, so the labels are not reliable."
        )
    if not n_labelled:
        status = PrecisionStatus.NOT_MEASURED
    else:
        status = PrecisionStatus.PROVISIONAL if reasons else PrecisionStatus.MEASURED
    return EmbeddingPrecision(
        status=status.value,
        precision=precision,
        n_labelled=n_labelled,
        n_same=n_same,
        sample=str(path),
        reason=reasons[0] if reasons else None,
        ci_low=ci_low,
        ci_high=ci_high,
        kappa=kappa,
        agreement=agreement,
        n_double_labelled=len(pair_1),
        labeller_kind=kind,
        kind_counts=kinds,
        n_stale=n_stale if pairs is not None else None,
        threshold=threshold,
        threshold_source=threshold_source,
        n_below=n_below,
        n_same_below=n_same_below,
        same_below_share=same_below_share,
        n_no_similarity=n_no_similarity,
        bands=bands,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def run_threshold(embedding_outcome: dict[str, Any]) -> tuple[float | None, str | None]:
    """The threshold of the run that made the links, and where it was read.

    The links file records it. When it does not, the config value is the fallback. The
    source says which one it was, so a report can show that the config may differ from the
    run.
    """
    recorded = embedding_outcome.get("threshold")
    if recorded is not None:
        return float(recorded), "links"
    try:
        configured = claim_dedup_config().min_similarity
    except (ClaimDedupConfigError, OSError, ValueError):
        return None, None
    return (None, None) if configured is None else (float(configured), "config")


def refused_precision(path: Path, exc: Exception) -> EmbeddingPrecision:
    """A precision block for a sample that was refused. It has no number."""
    return EmbeddingPrecision(
        status=PrecisionStatus.REFUSED.value,
        precision=None,
        n_labelled=0,
        n_same=0,
        sample=str(path),
        reason=PrecisionReason.SAMPLE_REFUSED.value,
        reasons=(PrecisionReason.SAMPLE_REFUSED.value,),
        error=str(exc),
    )


def resolve_sample_path(links: dict[str, Any] | None, links_path: Path) -> tuple[Path, str | None]:
    """The sample file to read, and a note when it is not the one the links name.

    ``claim_links.json`` holds an absolute path. When that file is gone, for example
    after the data directory moved, the sample beside the links file is read instead.
    """
    beside = Path(links_path).parent / HAND_CHECK_FILENAME
    recorded = links.get("sample") if links else None
    if not recorded:
        return beside, None
    if Path(recorded).exists():
        return Path(recorded), None
    return beside, (
        f"the sample named in the links file ({recorded}) does not exist, so the sample "
        f"beside the links file was read"
    )


def embedding_report(
    links: dict[str, Any] | None,
    *,
    simhash_count: int,
    simhash_considered: int,
    reason: str | None,
    sample_path: Path | None,
    sample_note: str | None = None,
) -> dict[str, Any]:
    """The per-pass block the evaluation report carries.

    ``links`` is ``None`` when no artifact has been written for the data on
    disk. The embedding pass is then ``not_run`` with a reason code, and its
    ``collapsed`` is ``None``, not 0.
    """
    if links is None:
        passes = {
            DedupPass.SIMHASH.value: PassOutcome(
                name=DedupPass.SIMHASH.value,
                status=PassStatus.OK.value,
                considered=simhash_considered,
                collapsed=simhash_count,
                radius=CLAIM_HAMMING_RADIUS,
            ).as_dict(),
            DedupPass.EMBEDDING.value: PassOutcome(
                name=DedupPass.EMBEDDING.value,
                status=PassStatus.NOT_RUN.value,
                considered=0,
                collapsed=None,
                reason=reason or EmbeddingReason.LINKS_ABSENT.value,
            ).as_dict(),
        }
    else:
        passes = dict(links.get("passes") or {})
    precision = None
    if sample_path:
        # Stale pairs can be counted only when the embedding pass ran and wrote its links.
        pairs = None
        embedding = passes.get(DedupPass.EMBEDDING.value) or {}
        threshold, source = run_threshold(embedding)
        if links is not None and embedding.get("collapsed") is not None:
            pairs = embedding_pairs(links.get("links") or [])
        try:
            precision = embedding_precision(
                sample_path, pairs=pairs, threshold=threshold, threshold_source=source
            )
        except ClaimSampleError as exc:
            log.warning("hand-check sample refused: %s", exc)
            precision = refused_precision(sample_path, exc)
        if sample_note:
            precision = replace(precision, sample_note=sample_note)
    return {
        "passes": passes,
        "precision": precision.as_dict() if precision else None,
    }


def config_without_embedding(config: ClaimDedupConfig) -> ClaimDedupConfig:
    """The same config with the embedding pass switched off, for a simhash-only run."""
    return replace(config, enabled=False)
