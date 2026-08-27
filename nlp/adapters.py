"""Concrete implementations of the ``nlp/interfaces.py`` protocols.

Everything heavy is imported **inside** a function, never at module scope. The
API image does not carry torch, and importing this module has to stay free so
/readyz and the router imports keep working there.

Every loader returns ``None`` rather than raising when a checkpoint is absent.
The caller decides whether that is a 503 on one route, a null column, or a
skipped task -- which is the whole reason the boundary exists.

These delegate to Phase 2/3's ``modeling`` package. They deliberately do not
reimplement any of it: the Protocol is the contract, ``modeling`` is one
implementation, and Phase 2/3's committed Parquet under ``data/scored/`` is
another. A deployment with no checkpoints serves the second one and says so.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from nlp.interfaces import ModelInfo

log = logging.getLogger(__name__)


def _modeling_available() -> bool:
    """Whether the Phase 2/3 tree is importable in this process.

    False in the API image by design. A worker built from Dockerfile.worker has
    it; a worker running on a laptop with a light install may not.
    """
    import importlib.util

    return importlib.util.find_spec("modeling") is not None


@lru_cache(maxsize=1)
def get_embedder():
    """Sentence embedder, or None if no model can be loaded.

    Cached: loading a sentence-transformer costs seconds and several hundred MB,
    and a task that reloaded it per batch would spend more time loading than
    embedding.
    """
    if not _modeling_available():
        log.info("modeling package not installed; no live embedder")
        return None

    try:
        from app.config import get_api_settings
        from modeling.text.embed import Embedder

        settings = get_api_settings()
        inner = Embedder(model_name=settings.embedding_model)
        # `load()` returns False rather than raising when the weights are not
        # reachable, which is Phase 2's own graceful-degradation contract.
        if not inner.load():
            log.info("embedder weights are not available; no live embedder")
            return None
        return _EmbedderAdapter(inner, settings.embedding_model, inner.dim)
    except Exception as exc:
        # A missing checkpoint, a missing torch, an out-of-memory on load: all
        # of them mean "this capability is unavailable", and none of them should
        # take down the caller.
        log.warning("could not load the embedder: %s: %s", type(exc).__name__, exc)
        return None


class _EmbedderAdapter:
    """Wraps Phase 2's embedder to satisfy the ``Embedder`` protocol."""

    def __init__(self, inner: Any, model_name: str, dim: int) -> None:
        self._inner = inner
        self.info = ModelInfo(name=model_name, version="v0.1.0", dim=dim)

    def encode(self, texts: list[str], *, batch_size: int = 64):
        import numpy as np

        vectors = self._inner.embed_texts(texts)
        matrix = getattr(vectors, "vectors", vectors)
        matrix = np.asarray(matrix, dtype="float32")

        # Normalize here as well as in Phase 2. It costs one pass and it is the
        # only way to be certain: pgvector stores these and the whole codebase
        # compares them with cosine, so one un-normalized batch produces
        # neighbours that are subtly wrong and that no index test would catch.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


@lru_cache(maxsize=1)
def get_post_scorer():
    """Per-post scorer: toxicity, sentiment, emotion, anomaly, misinfo, stance."""
    if not _modeling_available():
        return None
    try:
        from modeling.aux import aux_versions

        return _PostScorerAdapter(aux_versions())
    except Exception as exc:
        log.warning("could not load the post scorer: %s: %s", type(exc).__name__, exc)
        return None


class _PostScorerAdapter:
    """Wraps Phase 2's auxiliary scorers.

    Phase 2 exposes these as a DataFrame pass over the whole corpus, because
    the anomaly score is a *within-corpus percentile rank* and is only
    meaningful relative to a full record set. This adapter runs the same pass
    over a batch and passes the anomaly through as Phase 2 computed it -- with
    a skip reason on batches too small for the rank to mean anything, rather
    than a number that changes depending on how the batch happened to be cut.
    """

    #: Below this a percentile rank within the batch is not a percentile of
    #: anything. Phase 2's own scoring run computes these corpus-wide.
    MIN_BATCH_FOR_ANOMALY = 200

    def __init__(self, versions: dict[str, str]) -> None:
        import hashlib
        import json

        canonical = json.dumps(dict(versions), sort_keys=True, separators=(",", ":"))
        self.info = ModelInfo(
            name="phase2-aux",
            version="phase2:" + hashlib.sha256(canonical.encode()).hexdigest()[:12],
        )
        self._versions = dict(versions)

    def predict(self, texts: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        import pandas as pd

        from modeling.aux import run_aux_pass

        frame = pd.DataFrame(
            {
                "id": [f"batch:{i}" for i in range(len(texts))],
                "text": texts,
                "lang": [None] * len(texts),
                "author_id": [f"batch:author{i}" for i in range(len(texts))],
                "timestamp": pd.Timestamp.utcnow(),
            }
        )
        scored = run_aux_pass(frame, pd.DataFrame(), settings=None)

        drop_anomaly = len(texts) < self.MIN_BATCH_FOR_ANOMALY
        out: list[dict[str, Any]] = []
        for _, row in scored.iterrows():
            emotion = row.get("emotion") or {}
            emotion_scores = dict(emotion) if isinstance(emotion, dict) else {}
            usable = {k: v for k, v in emotion_scores.items() if v is not None}
            reasons = list(row.get("skip_reasons") or ())
            if drop_anomaly:
                reasons.append("anomaly:batch_too_small_for_percentile_rank")
            out.append(
                {
                    "toxicity": _clean(row.get("toxicity")),
                    "anomaly": None if drop_anomaly else _clean(row.get("anomaly_score")),
                    "misinfo_likelihood": _clean(row.get("misinfo_prob")),
                    "stance": row.get("stance"),
                    "stance_confidence": _clean(row.get("stance_conf")),
                    "sentiment": row.get("sentiment"),
                    "sentiment_score": _clean(row.get("sentiment_score")),
                    "emotion": max(usable, key=usable.get) if usable else None,
                    "emotion_scores": emotion_scores,
                    "skip_reason": reasons[0] if reasons else None,
                }
            )
        return out


@lru_cache(maxsize=1)
def get_bot_scorer():
    """XGBoost account classifier with SHAP attribution."""
    if not _modeling_available():
        return None
    try:
        from app.config import get_api_settings, load_yaml
        from modeling.accounts.bot_clf import BotModel

        version = str((load_yaml("models").get("bot") or {}).get("version", "v0.1.0"))
        directory = get_api_settings().models_dir / "bot" / version
        model = BotModel.load(directory)
        if model is None:
            log.info("no bot checkpoint at %s", directory)
            return None
        return _BotScorerAdapter(model, version)
    except Exception as exc:
        log.warning("could not load the bot classifier: %s: %s", type(exc).__name__, exc)
        return None


class _BotScorerAdapter:
    """Wraps Phase 2's ``BotModel`` plus its SHAP attribution.

    ``BotModel.predict_proba`` takes a numpy matrix whose columns are its own
    ``feature_names``, in that order. Building it from a dict of whatever the
    caller happened to have is the part that goes wrong silently: a column in
    the wrong position produces a confident, meaningless probability. The
    matrix is therefore built by name, and a feature the caller did not supply
    is NaN rather than 0 -- XGBoost handles NaN as missing natively, while a 0
    would be read as a real measurement of zero followers.
    """

    def __init__(self, inner: Any, version: str) -> None:
        self._inner = inner
        self.info = ModelInfo(name="xgboost-bot-clf", version=version)

    def score(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import numpy as np

        from modeling.accounts.bot_clf import shap_contributions

        names = list(self._inner.feature_names)
        matrix = np.array(
            [[_as_float(row.get(name)) for name in names] for row in features],
            dtype="float64",
        )
        probabilities = self._inner.predict_proba(matrix)
        attributions = shap_contributions(self._inner.estimator, matrix, names, top_k=8)

        return [
            {
                "bot_score": float(probability),
                # The attribution is not optional. A bot score with no
                # explanation is the black-box number this product argues
                # against, and /authors/{id}/score is required to return it.
                "top_features": list(attribution or []),
            }
            for probability, attribution in zip(probabilities, attributions, strict=False)
        ]


def _as_float(value: Any) -> float:
    """Missing stays missing. NaN, not 0.

    XGBoost treats NaN as "no value for this feature" natively. A 0 would be
    read as a measurement -- zero followers, zero posts -- and would move the
    prediction in a direction nobody intended.
    """
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@lru_cache(maxsize=1)
def get_deepfake_detector():
    """Xception frame classifier."""
    if not _modeling_available():
        return None
    try:
        from app.config import get_api_settings, load_yaml
        from modeling.media.deepfake_clf import DeepfakeScorer

        version = str((load_yaml("models").get("deepfake") or {}).get("version", "v0.1.0"))
        directory = get_api_settings().models_dir / "deepfake" / version
        scorer = DeepfakeScorer()
        if not scorer.load(directory if directory.exists() else None):
            log.info("no deepfake checkpoint at %s", directory)
            return None
        return _DeepfakeAdapter(scorer, version)
    except Exception as exc:
        log.warning("could not load the deepfake detector: %s: %s", type(exc).__name__, exc)
        return None


class _DeepfakeAdapter:
    def __init__(self, inner: Any, version: str) -> None:
        self._inner = inner
        self.info = ModelInfo(name="xception-deepfake", version=version)

    def analyze(self, path: str, **kwargs: Any) -> dict[str, Any]:
        from pathlib import Path

        return self._inner.score_media(Path(path))


@lru_cache(maxsize=1)
def get_summarizer():
    """LLM narrative labeller. Needs an API key, not a checkpoint."""
    from app.config import get_api_settings

    if not get_api_settings().anthropic_api_key:
        return None
    if not _modeling_available():
        return None
    try:
        from modeling.text.summarize import NarrativeSummarizer

        return _SummarizerAdapter(NarrativeSummarizer())
    except Exception as exc:
        log.warning("could not load the summarizer: %s: %s", type(exc).__name__, exc)
        return None


class _SummarizerAdapter:
    def __init__(self, inner: Any) -> None:
        from app.config import get_api_settings

        self._inner = inner
        self.info = ModelInfo(name=get_api_settings().llm_model, version="v0.1.0")

    def summarize(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        return self._inner.summarize(texts, **kwargs)


def _clean(value: Any) -> float | None:
    """Pandas NaN is not a score. It is a missing value, and it must not reach
    the database as a float that compares false against everything."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN != NaN


def reset() -> None:
    """Drop every cached model. Used by tests and after a checkpoint remount."""
    for loader in (
        get_embedder,
        get_post_scorer,
        get_bot_scorer,
        get_deepfake_detector,
        get_summarizer,
    ):
        loader.cache_clear()
