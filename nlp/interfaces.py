"""The narrow boundary between the backend and the Phase 2/3 models.

Everything the API and the workers know about machine learning is in this file.
Nothing below imports torch, transformers, xgboost or opencv at module scope --
these are Protocols and a loader, and the heavy import happens inside
``app/nlp/adapters.py`` only when a task actually needs to run inference.

Three reasons this boundary exists and is worth the indirection:

1. **The API image does not carry torch.** It imports this module for the
   capability check in /readyz and never loads a model.
2. **Tests stub here.** The acceptance criterion is "pytest passes with no GPU
   and no network"; a fake that satisfies these Protocols is the whole
   mechanism. See tests/conftest.py.
3. **A missing checkpoint degrades one capability, not the service.** Every
   loader returns ``None`` rather than raising, and the caller decides whether
   that is a 503 on one route or a null column.

Phase 2/3's implementation lives in ``modeling/``. This module deliberately does
not re-export it: the Protocol is the contract, ``modeling`` is one
implementation of it, and the Parquet tables under ``data/scored/`` are another.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# capability descriptors
# ---------------------------------------------------------------------------

#: The capability names /readyz reports and the routes gate on. These are the
#: strings the UI sees, so they are stable.
CAPABILITIES: tuple[str, ...] = (
    "embed",
    "cluster",
    "misinfo",
    "stance",
    "aux",
    "bot",
    "deepfake",
    "summarize",
    "compass",
)


@dataclass(frozen=True)
class ModelInfo:
    """What a scored row needs in order to be citable.

    ``version`` is what lands in ``scoring_version`` and in the
    ``model_versions`` map Phase 2 already writes into its Parquet tables. Two
    rows with different versions are not comparable, and the API says so rather
    than averaging them.
    """

    name: str
    version: str
    checkpoint_uri: str | None = None
    dim: int | None = None
    loaded_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "checkpoint_uri": self.checkpoint_uri,
            "dim": self.dim,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
        }


@dataclass(frozen=True)
class Capability:
    """One scorer's availability, as reported by /readyz."""

    name: str
    available: bool
    #: "ready" | "unavailable" | "degraded". `degraded` means a fallback is in
    #: use -- e.g. centroid narrative labels because no LLM key is configured.
    status: str
    detail: str
    model: ModelInfo | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "status": self.status,
            "detail": self.detail,
            "model": self.model.as_dict() if self.model else None,
        }


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class Embedder(Protocol):
    """Sentence embeddings. Vectors are L2-normalized on the way out.

    Normalization is part of the contract, not an implementation detail:
    pgvector stores these and the whole codebase compares them with cosine
    distance. An un-normalized vector sneaking in produces neighbours that are
    subtly wrong in a way no test on the index itself would catch.
    """

    info: ModelInfo

    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        """Return ``(len(texts), dim)`` float32, L2-normalized, row-aligned."""
        ...


@runtime_checkable
class TextClassifier(Protocol):
    """Misinformation likelihood, stance, toxicity, sentiment, emotion.

    ``predict`` returns one dict per input, aligned by index. A row the model
    declines to score (wrong language, too short) gets ``{"skip_reason": ...}``
    rather than a fabricated number -- the same null-versus-zero discipline the
    corpus enforces on engagement metrics.
    """

    info: ModelInfo

    def predict(self, texts: list[str], **kwargs: Any) -> list[dict[str, Any]]: ...


@runtime_checkable
class BotScorer(Protocol):
    """Account-level bot probability with per-feature attribution.

    The attribution is not optional. A bot score with no explanation is exactly
    the black-box number this product exists to argue against, and
    /authors/{id}/score is required to return the features that drove it.
    """

    info: ModelInfo

    def score(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


@runtime_checkable
class Clusterer(Protocol):
    """HDBSCAN over embeddings -> narrative cluster assignments."""

    info: ModelInfo

    def fit_predict(self, vectors: np.ndarray, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class DeepfakeDetector(Protocol):
    """Xception over sampled frames.

    ``explanation`` is a required field of the return, not a nicety: the UI spec
    is explicit that a bare probability is not an acceptable answer to "is this
    video fake".
    """

    info: ModelInfo

    def analyze(self, path: str, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class Summarizer(Protocol):
    """One LLM call per cluster, never per post."""

    info: ModelInfo

    def summarize(self, texts: list[str], **kwargs: Any) -> dict[str, Any]: ...
