"""Auxiliary scoring passes: toxicity, sentiment, emotion, anomaly.

These fill four of the product's six scorecard fields. All four are
off-the-shelf or unsupervised on purpose -- fine-tuning any of them would create
a metric this project would then have to defend, and no measured problem
justifies one.

:func:`run_aux_pass` is the orchestrator: it reads Phase 1 records, runs each
scorer, and assembles one ``record_scores`` frame. It is the fastest path from
a clean checkout to a real artifact, which is why it is built before any
training happens -- it validates the whole IO contract first.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from modeling.aux.anomaly import AnomalyScorer
from modeling.aux.base import AuxScorer, ScoringOutcome, SkipReason, warm_cache
from modeling.aux.emotion import EmotionScorer
from modeling.aux.sentiment import SentimentScorer
from modeling.aux.toxicity import ToxicityScorer
from modeling.config import ModelingSettings, get_settings, module_config
from modeling.io import utcnow

log = logging.getLogger(__name__)

TEXT_SCORERS: tuple[type[AuxScorer], ...] = (ToxicityScorer, SentimentScorer, EmotionScorer)

__all__ = [
    "AnomalyScorer",
    "AuxScorer",
    "EmotionScorer",
    "ScoringOutcome",
    "SentimentScorer",
    "SkipReason",
    "ToxicityScorer",
    "aux_model_names",
    "aux_versions",
    "run_aux_pass",
    "warm_cache",
]


def aux_model_names() -> list[str]:
    """The HF model ids the aux pass needs. Backs ``cli warm-cache``."""
    return [str(module_config(m).get("model_name")) for m in ("toxicity", "sentiment", "emotion")]


def aux_versions() -> dict[str, str]:
    """The ``model_versions`` block every aux-scored row carries."""
    return {
        module: str(module_config(module).get("version", "v0.0.0-unset"))
        for module in ("toxicity", "sentiment", "emotion", "anomaly")
    }


def run_aux_pass(
    records: pd.DataFrame,
    authors: pd.DataFrame | None = None,
    *,
    settings: ModelingSettings | None = None,
    scorers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run every aux scorer and assemble a ``record_scores`` frame.

    One row per input record, always. A record no scorer could handle still
    gets a row -- with nulls and a populated ``skip_reasons`` -- because a
    missing row and a null row mean different things downstream: "not in the
    corpus" versus "in the corpus, not assessed".
    """
    settings = settings or get_settings()
    wanted = set(scorers) if scorers else {"toxicity", "sentiment", "emotion", "anomaly"}
    if not len(records):
        log.warning("aux pass received zero records")
        return pd.DataFrame()

    outcomes: dict[str, ScoringOutcome] = {}
    for scorer_cls in TEXT_SCORERS:
        if scorer_cls.module not in wanted:
            continue
        outcomes[scorer_cls.module] = scorer_cls(settings).score(records)

    if "anomaly" in wanted:
        outcomes["anomaly"] = AnomalyScorer(settings).score(records, authors)

    versions = {module: aux_versions()[module] for module in outcomes}
    now = utcnow()
    rows: list[dict[str, Any]] = []

    for row in records.itertuples(index=False):
        record_id = str(row.id)
        entry: dict[str, Any] = {
            "record_id": record_id,
            "source": str(getattr(row, "source", "unknown")),
            "scored_at": now,
        }
        reasons: list[str] = []

        toxicity = outcomes.get("toxicity")
        if toxicity is not None:
            entry["toxicity"] = toxicity.values.get(record_id)
            _note(reasons, "toxicity", toxicity.skipped.get(record_id))

        sentiment = outcomes.get("sentiment")
        if sentiment is not None:
            payload = sentiment.values.get(record_id) or {}
            entry["sentiment"] = payload.get("sentiment")
            entry["sentiment_score"] = payload.get("sentiment_score")
            _note(reasons, "sentiment", sentiment.skipped.get(record_id))

        emotion = outcomes.get("emotion")
        if emotion is not None:
            entry["emotion"] = emotion.values.get(record_id)
            _note(reasons, "emotion", emotion.skipped.get(record_id))

        anomaly = outcomes.get("anomaly")
        if anomaly is not None:
            entry["anomaly_score"] = anomaly.values.get(record_id)
            _note(reasons, "anomaly", anomaly.skipped.get(record_id))

        entry["skip_reasons"] = reasons
        # Per-row provenance: only the scorers that actually produced a value
        # for this row are claimed. A row skipped by toxicity must not claim a
        # toxicity version, or a retrain would think it is current.
        entry["model_versions"] = {
            module: version
            for module, version in versions.items()
            if outcomes[module].values.get(record_id) is not None
        }
        rows.append(entry)

    frame = pd.DataFrame(rows)
    _log_coverage(frame)
    return frame


def _note(reasons: list[str], module: str, reason: str | None) -> None:
    if reason:
        reasons.append(f"{module}:{reason}")


def _log_coverage(frame: pd.DataFrame) -> None:
    """Say plainly how much of the corpus each scorer actually covered.

    A column that is 98% null is a finding, not a detail, and it is much
    cheaper to notice here than in a dashboard three weeks later.
    """
    total = len(frame)
    for column in ("toxicity", "sentiment_score", "emotion", "anomaly_score"):
        if column not in frame.columns:
            continue
        filled = int(frame[column].notna().sum())
        pct = 100 * filled / total if total else 0.0
        level = log.warning if pct < 50 else log.info
        level("aux coverage: %-16s %5d/%d rows (%.1f%%)", column, filled, total, pct)
