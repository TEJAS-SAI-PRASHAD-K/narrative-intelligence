"""Sentence embeddings, cached and L2-normalized.

Default is ``sentence-transformers/all-MiniLM-L6-v2``: 384-dim, small enough to
embed a corpus on a laptop CPU, good enough for clustering short social text.
``BAAI/bge-base-en-v1.5`` (768-dim) is config-switchable and better, and needs a
GPU for bulk work.

Three properties this module is responsible for.

**The dimension is read from the model, never hardcoded.** Phase 4's pgvector
column width depends on it, and a hardcoded 384 that silently disagrees with a
768-dim checkpoint is a migration failure discovered in production.

**Embeddings are cached by ``(model_name, model_version, sha256(text))``.**
Re-embedding a corpus because a plotting line changed is a wasted afternoon,
and the cache is what makes the clustering stage cheap enough to rerun.

**Vectors are L2-normalized before anything downstream sees them.** Every cosine
assumption in this project — HDBSCAN on Euclidean distance, centroid matching
for narrative-id stability, the coordination graph's near-duplicate threshold —
depends on it. Normalizing at the boundary means no downstream module has to
remember.

**Truncation is recorded, not silent.** MiniLM and bge both cap at 512 tokens
(we use 256 by default). For text over the cap, the first sentence is kept whole
and the remainder is filled from the head — a news article's lede carries the
claim, and a naive head-truncation can cut it mid-way. `was_truncated` is
returned alongside so the clustering stage can weight or exclude, and so
"GDELT's embeddings look odd" is answerable rather than mysterious.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class EmbeddingResult:
    """Vectors plus the provenance needed to interpret them."""

    vectors: np.ndarray  # (n, dim), float32, L2-normalized
    record_ids: list[str]
    model_name: str
    model_version: str
    dim: int
    was_truncated: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    #: Records that could not be embedded at all, with a reason.
    skipped: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.record_ids)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "record_id": self.record_ids,
                "embedding": list(self.vectors),
                "was_truncated": self.was_truncated,
            }
        )

    def index_of(self, record_id: str) -> int | None:
        try:
            return self.record_ids.index(record_id)
        except ValueError:
            return None

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "version": self.model_version,
            "dim": self.dim,
            "embedded": len(self.record_ids),
            "truncated": int(self.was_truncated.sum()) if len(self.was_truncated) else 0,
            "skipped": len(self.skipped),
        }


class EmbeddingCache:
    """On-disk cache: one ``.npy`` matrix plus a key index per model+version.

    Not Parquet: this is a dense float matrix with no schema to speak of, and
    ``np.load`` with ``mmap_mode`` is the right tool. The key index is JSON so it
    stays readable when something goes wrong.
    """

    def __init__(
        self,
        model_name: str,
        version: str,
        dim: int,
        settings: ModelingSettings | None = None,
    ):
        self.settings = settings or get_settings()
        self.model_name = model_name
        self.version = version
        self.dim = dim
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{model_name}-{version}").strip("-").lower()
        self.matrix_path = self.settings.embeddings_dir / f"{slug}.npy"
        self.index_path = self.settings.embeddings_dir / f"{slug}.keys.json"
        self._keys: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._pending: list[tuple[str, np.ndarray]] = []
        self._load()

    def _load(self) -> None:
        if not (self.matrix_path.exists() and self.index_path.exists()):
            return
        try:
            self._keys = json.loads(self.index_path.read_text(encoding="utf-8"))
            matrix = np.load(self.matrix_path)
        except Exception as exc:  # pragma: no cover - corrupt cache
            log.warning(
                "embedding cache at %s unreadable (%s); starting fresh", self.matrix_path, exc
            )
            self._keys, self._matrix = {}, None
            return
        if matrix.shape[1] != self.dim or matrix.shape[0] != len(self._keys):
            # A dimension change means a different model wrote this file. Do not
            # try to reconcile: silently mixing 384- and 768-dim vectors would
            # produce a centroid nobody could debug.
            log.warning(
                "embedding cache %s has shape %s but this model is %d-dim with %d keys; "
                "discarding the cache",
                self.matrix_path,
                matrix.shape,
                self.dim,
                len(self._keys),
            )
            self._keys, self._matrix = {}, None
            return
        self._matrix = matrix
        log.debug("embedding cache: %d vectors", len(self._keys))

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def get(self, text: str) -> np.ndarray | None:
        index = self._keys.get(self.key(text))
        if index is None or self._matrix is None:
            return None
        return self._matrix[index]

    def put(self, text: str, vector: np.ndarray) -> None:
        self._pending.append((self.key(text), vector.astype(np.float32)))

    def flush(self) -> None:
        if not self._pending:
            return
        fresh = [(k, v) for k, v in self._pending if k not in self._keys]
        self._pending = []
        if not fresh:
            return
        block = np.stack([v for _, v in fresh]).astype(np.float32)
        if self._matrix is None:
            self._matrix = block
        else:
            self._matrix = np.concatenate([self._matrix, block], axis=0)
        offset = self._matrix.shape[0] - block.shape[0]
        for i, (key, _) in enumerate(fresh):
            self._keys[key] = offset + i
        self.matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.matrix_path, self._matrix)
        self.index_path.write_text(json.dumps(self._keys), encoding="utf-8")
        log.debug("embedding cache flushed: %d total vectors", len(self._keys))


class Embedder:
    """Batched sentence-transformer wrapper with caching and truncation policy."""

    module = "embed"

    def __init__(self, settings: ModelingSettings | None = None, model_name: str | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.model_name = model_name or str(self.config.get("model_name"))
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.max_seq_length = int(self.config.get("max_seq_length", 256))
        self.normalize = bool(self.config.get("normalize", True))
        self._model = None
        self._cache: EmbeddingCache | None = None
        self._dim: int | None = None
        self._load_attempted = False

    # --- model -----------------------------------------------------------
    def load(self) -> bool:
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.error("sentence-transformers is not installed; install the 'modeling' extra")
            return False

        device = self.settings.resolve_device()
        for local_only in (True, False):
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device="cpu" if device == "mps" else device,
                    local_files_only=local_only,
                )
                break
            except Exception as exc:
                if local_only:
                    continue
                log.error("could not load %s: %s", self.model_name, exc)
                return False
        self._model.max_seq_length = self.max_seq_length
        # Read the dimension from the model. Never hardcode it: Phase 4's
        # pgvector column width is derived from this number.
        self._dim = int(self._model.get_sentence_embedding_dimension())
        log.info(
            "embedder: %s (%d-dim, max_seq_length=%d, device=%s)",
            self.model_name,
            self._dim,
            self.max_seq_length,
            self._model.device,
        )
        return True

    @property
    def dim(self) -> int:
        if self._dim is None:
            if not self.load():
                raise RuntimeError(f"embedding model {self.model_name} unavailable")
        return int(self._dim)

    @property
    def cache(self) -> EmbeddingCache:
        if self._cache is None:
            self._cache = EmbeddingCache(self.model_name, self.version, self.dim, self.settings)
        return self._cache

    # --- truncation ------------------------------------------------------
    def prepare(self, text: str) -> tuple[str, bool]:
        """Apply the truncation policy. Returns ``(text, was_truncated)``.

        Character-budgeted rather than token-budgeted, because running the
        tokenizer twice over the corpus to save a few characters is not worth
        it. The budget is deliberately generous (~4 chars/token) so that
        anything the model would truncate is caught, and the sentence-level cut
        keeps the lede intact.
        """
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        budget = self.max_seq_length * 4
        if len(cleaned) <= budget:
            return cleaned, False

        sentences = _SENTENCE_END.split(cleaned)
        lede = sentences[0][:budget]
        remaining = budget - len(lede)
        if remaining <= 0:
            return lede, True
        # Keep the lede whole, then fill from the head of the rest. A news
        # article's claim lives in the first sentence; a naive head-truncation
        # can cut it in half.
        tail = " ".join(sentences[1:])[:remaining]
        return (lede + " " + tail).strip(), True

    # --- embedding -------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed raw strings (already prepared). Cache-aware."""
        if not self.load():
            raise RuntimeError(f"embedding model {self.model_name} unavailable")

        vectors: list[np.ndarray | None] = [self.cache.get(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            fresh = self._model.encode(
                [texts[i] for i in missing],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ).astype(np.float32)
            for slot, vector in zip(missing, fresh, strict=True):
                vectors[slot] = vector
                self.cache.put(texts[slot], vector)
            self.cache.flush()
        matrix = np.stack(vectors).astype(np.float32)
        if self.normalize:
            # Belt and braces: cached vectors were normalized when written, but
            # a cache written under a different setting would otherwise poison
            # every cosine downstream.
            matrix = l2_normalize(matrix)
        return matrix

    def embed_records(
        self, records: pd.DataFrame, *, min_chars: int = 0
    ) -> EmbeddingResult:
        """Embed a frame of Phase 1 records.

        ``min_chars`` skips text too short to embed meaningfully. GDELT records
        carry article metadata rather than body text (median ~83 characters in
        this corpus), and a 30-character headline produces an embedding that
        clusters on stopwords. Skipping is honest; embedding it and pretending
        is not.
        """
        skipped: dict[str, str] = {}
        keep_ids: list[str] = []
        prepared: list[str] = []
        truncated: list[bool] = []

        for row in records.itertuples(index=False):
            record_id = str(row.id)
            text = getattr(row, "text", None)
            if text is None or pd.isna(text) or not str(text).strip():
                skipped[record_id] = "empty_text"
                continue
            cleaned, was_truncated = self.prepare(text)
            if len(cleaned) < min_chars:
                skipped[record_id] = "text_too_short"
                continue
            keep_ids.append(record_id)
            prepared.append(cleaned)
            truncated.append(was_truncated)

        if not prepared:
            log.warning("nothing to embed: %d records all skipped", len(records))
            return EmbeddingResult(
                vectors=np.zeros((0, self.dim), dtype=np.float32),
                record_ids=[],
                model_name=self.model_name,
                model_version=self.version,
                dim=self.dim,
                skipped=skipped,
            )

        vectors = self.embed_texts(prepared)
        result = EmbeddingResult(
            vectors=vectors,
            record_ids=keep_ids,
            model_name=self.model_name,
            model_version=self.version,
            dim=self.dim,
            was_truncated=np.array(truncated, dtype=bool),
            skipped=skipped,
        )
        log.info("embedded %s", result.summary())
        if result.was_truncated.any():
            log.info(
                "%d/%d records exceeded %d tokens and were truncated (lede kept whole)",
                int(result.was_truncated.sum()),
                len(result),
                self.max_seq_length,
            )
        return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, safe against zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two sets of (assumed L2-normalized) vectors.

    Normalizes defensively rather than trusting the caller: a centroid computed
    as a mean of unit vectors is *not* itself a unit vector, and that is exactly
    where this function gets used.
    """
    a = l2_normalize(np.atleast_2d(a))
    b = l2_normalize(np.atleast_2d(b))
    return a @ b.T


def embedding_dim(settings: ModelingSettings | None = None) -> int:
    """The configured model's dimension. Phase 4 sizes its vector column from this."""
    return Embedder(settings).dim


def cache_paths(settings: ModelingSettings | None = None) -> list[Path]:
    root = (settings or get_settings()).embeddings_dir
    return sorted(root.glob("*.npy")) if root.exists() else []
