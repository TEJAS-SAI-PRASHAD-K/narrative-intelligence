"""A fitted TF-IDF baseline that can score new text.

``baselines.tfidf_logreg`` fits and throws the model away, which is right for a
report and useless for scoring a second corpus. This module keeps the fit.

Three things are enforced because each fails without an error:

* The model is fitted on exactly the training rows of a shared
  ``SplitResult``. Scoring the split's own test rows later is then honest.
* The model was trained on English only. A Hindi or unlabelled-language text
  gets no score and a counted reason code. English is never assumed.
* Which column of ``predict_proba`` means "misinformation" is read from the
  saved metadata and the classifier's ``classes_``, and a mismatch refuses
  to load. A flipped column gives a confident wrong score for every record.

The result type is ``modeling.aux.base.ScoringOutcome``: ``values`` and
``skipped`` together cover every input id.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from modeling.aux.base import ScoringOutcome, SkipReason
from modeling.config import get_settings
from modeling.datasets.splits import SplitResult
from modeling.eval.baselines import build_tfidf_pipeline

log = logging.getLogger(__name__)

ARTIFACT_FORMAT = 1
#: The only language this model was trained on and may score.
SCORED_LANGUAGE = "en"
POSITIVE_LABEL = 1
POSITIVE_CLASS = "misinformation"
DEFAULT_ARTIFACT_NAME = "tfidf_misinfo.joblib"


class TfidfSkipReason(SkipReason):
    #: ``lang`` is null or unrecognised, so English cannot be established.
    UNKNOWN_LANGUAGE = "unknown_language"


class OrientationError(RuntimeError):
    """The saved label orientation disagrees with the classifier."""


def default_artifact_path() -> Path:
    return get_settings().artifacts_dir / "baselines" / DEFAULT_ARTIFACT_NAME


@dataclass(frozen=True)
class TfidfModel:
    """A loaded pipeline with its verified metadata."""

    pipeline: Pipeline
    metadata: dict[str, Any]
    #: Column of ``predict_proba`` that holds P(misinformation).
    positive_index: int


def _check_orientation(pipeline: Pipeline, metadata: dict[str, Any]) -> int:
    if (
        metadata.get("positive_label") != POSITIVE_LABEL
        or metadata.get("positive_class") != POSITIVE_CLASS
    ):
        raise OrientationError(
            f"saved orientation is positive_label={metadata.get('positive_label')!r}, "
            f"positive_class={metadata.get('positive_class')!r}; this scorer requires "
            f"{POSITIVE_LABEL!r} / {POSITIVE_CLASS!r}. Refusing to score."
        )
    classes = [int(c) for c in pipeline.classes_]
    if classes != [0, 1]:
        raise OrientationError(
            f"classifier classes_ are {classes}, expected [0, 1]. "
            "Column order cannot be trusted. Refusing to score."
        )
    if metadata.get("classes") != classes:
        raise OrientationError(
            f"classes_ at fit time were {metadata.get('classes')!r} but the loaded "
            f"classifier has {classes}. Refusing to score."
        )
    return classes.index(POSITIVE_LABEL)


def fit_tfidf_on_split(
    frame: pd.DataFrame,
    split: SplitResult,
    *,
    text_col: str = "text",
    label_col: str = "label",
    path: str | os.PathLike[str] | None = None,
    save: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit on ``frame.iloc[split.train]`` only and optionally save.

    ``frame`` must be the frame the split indexed (the deduped one returned by
    ``group_train_val_test``). Labels are 0/1 with 1 = misinformation. The seed
    comes from settings, not from the caller.
    """
    train = frame.iloc[split.train]
    texts = train[text_col].fillna("").astype(str).tolist()
    labels = train[label_col].to_numpy()
    if not set(np.unique(labels).tolist()) <= {0, 1} or len(np.unique(labels)) != 2:
        raise ValueError(
            f"training labels must contain both 0 and 1 and nothing else; got "
            f"{sorted(set(np.unique(labels).tolist()))!r}"
        )
    seed = get_settings().seed
    pipeline = build_tfidf_pipeline(seed)
    pipeline.fit(texts, labels.astype(int))
    metadata = {
        "format": ARTIFACT_FORMAT,
        "n_train": len(texts),
        "split_description": split.describe(),
        "seed": seed,
        "positive_label": POSITIVE_LABEL,
        "positive_class": POSITIVE_CLASS,
        "classes": [int(c) for c in pipeline.classes_],
        "language": SCORED_LANGUAGE,
        "sklearn_version": sklearn.__version__,
    }
    _check_orientation(pipeline, metadata)
    if save:
        save_tfidf(pipeline, metadata, path)
    return pipeline, metadata


def save_tfidf(
    pipeline: Pipeline, metadata: dict[str, Any], path: str | os.PathLike[str] | None = None
) -> Path:
    """Write ``{"pipeline", "metadata"}`` atomically; returns the path."""
    _check_orientation(pipeline, metadata)
    target = Path(path) if path is not None else default_artifact_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, tmp)
    os.replace(tmp, target)
    return target


def load_tfidf(path: str | os.PathLike[str] | None = None) -> TfidfModel:
    """Load and verify. Raises on a missing file or inconsistent orientation."""
    source = Path(path) if path is not None else default_artifact_path()
    if not source.exists():
        raise FileNotFoundError(f"no TF-IDF artifact at {source}; fit one first")
    payload = joblib.load(source)
    if not isinstance(payload, dict) or "pipeline" not in payload or "metadata" not in payload:
        raise ValueError(f"{source} is not a TF-IDF artifact (missing pipeline or metadata)")
    pipeline, metadata = payload["pipeline"], payload["metadata"]
    if metadata.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"unsupported artifact format {metadata.get('format')!r}")
    positive_index = _check_orientation(pipeline, metadata)
    if metadata.get("sklearn_version") != sklearn.__version__:
        log.warning(
            "TF-IDF artifact was fitted with scikit-learn %s but %s is installed; "
            "scores may differ or fail",
            metadata.get("sklearn_version"),
            sklearn.__version__,
        )
    return TfidfModel(pipeline=pipeline, metadata=metadata, positive_index=positive_index)


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value) or value is pd.NA


def score_tfidf(
    model: TfidfModel,
    ids: Sequence[Any],
    texts: Sequence[Any],
    langs: Sequence[Any],
) -> ScoringOutcome:
    """P(misinformation) per id. English only; everything else is skipped.

    ``values`` maps id to probability. ``skipped`` maps id to a reason code.
    Together they cover every input id exactly once.
    """
    if not (len(ids) == len(texts) == len(langs)):
        raise ValueError(
            f"ids, texts and langs must be the same length; got "
            f"{len(ids)}, {len(texts)}, {len(langs)}"
        )
    keys = [str(i) for i in ids]
    if len(set(keys)) != len(keys):
        raise ValueError("ids must be unique; duplicates would hide records")

    outcome = ScoringOutcome()
    todo: list[tuple[str, str]] = []
    for key, text, lang in zip(keys, texts, langs, strict=True):
        if _is_null(lang) or not str(lang).strip():
            outcome.skipped[key] = TfidfSkipReason.UNKNOWN_LANGUAGE
        elif str(lang).strip().lower() != SCORED_LANGUAGE:
            outcome.skipped[key] = TfidfSkipReason.UNSUPPORTED_LANGUAGE
        elif _is_null(text) or not str(text).strip():
            outcome.skipped[key] = TfidfSkipReason.EMPTY_TEXT
        else:
            todo.append((key, str(text)))

    if todo:
        try:
            proba = model.pipeline.predict_proba([t for _, t in todo])[:, model.positive_index]
        except Exception as exc:
            log.warning("TF-IDF inference failed for %d records: %s", len(todo), exc)
            for key, _ in todo:
                outcome.skipped[key] = TfidfSkipReason.INFERENCE_ERROR
        else:
            for (key, _), p in zip(todo, proba, strict=True):
                outcome.values[key] = float(p)

    if outcome.skipped:
        log.info(
            "TF-IDF skipped %d of %d records: %s",
            len(outcome.skipped),
            len(keys),
            outcome.tally(),
        )
    return outcome
