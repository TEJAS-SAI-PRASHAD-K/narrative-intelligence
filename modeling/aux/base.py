"""Shared plumbing for the auxiliary scoring passes.

The four aux scorers (toxicity, sentiment, emotion, anomaly) share the same
obligations, so they share the same base class rather than four near-copies:

* **Batched and CPU-capable.** These run in a Phase 4 background worker with no
  GPU, so nothing here may assume CUDA.
* **Cached by text hash.** Keyed by ``(model_name, model_version, sha256(text))``.
  Re-scoring a corpus because a plotting line changed is a wasted afternoon, and
  the cache is what makes ``score`` genuinely resumable.
* **Language-gated.** Every one of these models is English-only. Non-English text
  gets ``null`` and a reason code, never a score. An English toxicity model run
  on German text produces a number, and that number is noise.
* **Reason-coded skips.** Same discipline as Phase 1: every null has a logged
  reason. Silent nulls become unexplainable metrics three weeks later.

The models are pretrained and used off the shelf. That is a deliberate choice,
not laziness: fine-tuning any of them would create a metric this project would
then have to defend, and no measured problem justifies one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config

log = logging.getLogger(__name__)


class SkipReason:
    """Reason codes for a null score. Mirrors ``ingest.schema.DropReason``."""

    EMPTY_TEXT = "empty_text"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    MODEL_UNAVAILABLE = "model_unavailable"
    TEXT_TOO_SHORT = "text_too_short"
    INFERENCE_ERROR = "inference_error"
    NOT_ENOUGH_HISTORY = "not_enough_history"


@dataclass
class ScoringOutcome:
    """What a scorer produced for a batch of records.

    ``values`` is keyed by record id; a record absent from it was skipped, and
    ``skipped`` says why. The two together are exhaustive over the input.
    """

    values: dict[str, Any] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def tally(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reason in self.skipped.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts


class TextCache:
    """Parquet-backed cache keyed by ``(model, version, sha256(text))``.

    A flat file rather than a database: the corpus is Parquet everywhere else,
    the access pattern is "load all, look up, append", and a sqlite dependency
    for one dictionary would be the wrong trade.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        version: str,
        settings: ModelingSettings | None = None,
    ):
        self.settings = settings or get_settings()
        self.name = name
        self.model_name = model_name
        self.version = version
        self.path = self.settings.cache_dir / "aux" / f"{name}.parquet"
        self._entries: dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            frame = pd.read_parquet(self.path)
        except Exception as exc:  # pragma: no cover - corrupt cache
            log.warning("aux cache %s unreadable (%s); starting fresh", self.path, exc)
            return
        self._entries = dict(zip(frame["key"].tolist(), frame["payload"].tolist(), strict=True))
        log.debug("aux cache %s: %d entries", self.name, len(self._entries))

    def key(self, text: str) -> str:
        digest = hashlib.sha256(
            f"{self.model_name}|{self.version}|{text}".encode()
        ).hexdigest()
        return digest[:32]

    def get(self, text: str) -> Any | None:
        payload = self._entries.get(self.key(text))
        if payload is None:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:  # pragma: no cover
            return None

    def put(self, text: str, value: Any) -> None:
        self._entries[self.key(text)] = json.dumps(value)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {"key": list(self._entries), "payload": list(self._entries.values())}
        ).sort_values("key", kind="stable")
        frame.to_parquet(self.path, index=False, compression="zstd")
        self._dirty = False
        log.debug("aux cache %s flushed: %d entries", self.name, len(self._entries))


class AuxScorer(ABC):
    """One auxiliary scoring pass over records."""

    #: Config section in configs/models.yaml, and the key in ``model_versions``.
    module: str
    #: The column(s) this scorer fills in ``record_scores``.
    output_columns: tuple[str, ...] = ()
    #: Minimum characters to attempt. Below this the models are guessing.
    min_chars: int = 3

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.model_name = str(self.config.get("model_name", self.module))
        self.batch_size = int(self.config.get("batch_size", 32))
        self.max_length = int(self.config.get("max_length", 256))
        self._cache: TextCache | None = None
        self._unavailable_reason: str | None = None

    @property
    def cache(self) -> TextCache:
        if self._cache is None:
            self._cache = TextCache(self.module, self.model_name, self.version, self.settings)
        return self._cache

    # --- gating ----------------------------------------------------------
    def gate(self, text: str | None, lang: str | None) -> str | None:
        """Return a skip reason, or ``None`` to proceed.

        The one place the English-only policy is applied to text scoring. A
        ``lang`` of ``None`` (common for short text, where Phase 1 declined to
        guess) is admitted only because ``score_unknown_language`` says so, and
        that assumption is recorded rather than hidden.
        """
        # pd.isna, not `is None`: a null text arrives from Parquet as NaN or
        # pd.NA depending on the column's dtype, and `str(NaN)` is the
        # three-character string "nan", which sails through a length check and
        # gets scored as if it were content.
        if text is None or pd.isna(text) or not str(text).strip():
            return SkipReason.EMPTY_TEXT
        if len(str(text).strip()) < self.min_chars:
            return SkipReason.TEXT_TOO_SHORT
        if not self.settings.language_allowed(lang):
            return SkipReason.UNSUPPORTED_LANGUAGE
        return None

    # --- lifecycle -------------------------------------------------------
    @abstractmethod
    def load(self) -> bool:
        """Load the model. Return False if unavailable (offline, no cache).

        Returning False rather than raising is what lets batch scoring proceed
        with honest nulls for one scorer while the others still run.
        """

    @abstractmethod
    def score_texts(self, texts: Sequence[str]) -> list[Any]:
        """Score a batch of already-gated texts. Same length as the input."""

    def score(self, records: pd.DataFrame) -> ScoringOutcome:
        """Score a frame of Phase 1 records. Cache-aware, batched, gated."""
        outcome = ScoringOutcome()
        pending: list[tuple[str, str]] = []  # (record_id, text)

        for row in records.itertuples(index=False):
            record_id = str(row.id)
            text = getattr(row, "text", None)
            lang = getattr(row, "lang", None)
            reason = self.gate(text, lang)
            if reason:
                outcome.skipped[record_id] = reason
                continue
            cached = self.cache.get(str(text))
            if cached is not None:
                outcome.values[record_id] = cached
                continue
            pending.append((record_id, str(text)))

        if pending and not self.load():
            for record_id, _ in pending:
                outcome.skipped[record_id] = (
                    self._unavailable_reason or SkipReason.MODEL_UNAVAILABLE
                )
            self._log_tally(outcome, len(records))
            return outcome

        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            texts = [text for _, text in chunk]
            try:
                results = self.score_texts(texts)
            except Exception as exc:  # pragma: no cover - runtime model failure
                log.warning(
                    "%s inference failed on a batch of %d: %s", self.module, len(chunk), exc
                )
                for record_id, _ in chunk:
                    outcome.skipped[record_id] = SkipReason.INFERENCE_ERROR
                continue
            for (record_id, text), value in zip(chunk, results, strict=True):
                outcome.values[record_id] = value
                self.cache.put(text, value)

        self.cache.flush()
        self._log_tally(outcome, len(records))
        return outcome

    def _log_tally(self, outcome: ScoringOutcome, total: int) -> None:
        tally = outcome.tally()
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
        log.info(
            "%s: scored %d/%d%s",
            self.module,
            len(outcome.values),
            total,
            f" (skipped: {reasons})" if tally else "",
        )


# ---------------------------------------------------------------------------
# transformers helper
# ---------------------------------------------------------------------------
def load_pipeline(model_name: str, task: str, *, top_k: int | None = None) -> Any | None:
    """Build a transformers pipeline on CPU, or return None if unavailable.

    ``local_files_only`` is tried first so a cached model works with the network
    down, then a normal load. Returning None (rather than raising) is what lets
    the pipeline degrade to honest nulls on a machine that has never downloaded
    the weights.
    """
    try:
        from transformers import pipeline
    except ImportError:
        log.warning("transformers is not installed; %s will write nulls", model_name)
        return None

    settings = get_settings()
    device = settings.resolve_device()
    # mps is flaky for these small classifiers and gains nothing at this corpus
    # size; the deployment target is CPU anyway, so measure what we ship.
    torch_device = 0 if device == "cuda" else -1

    for local_only in (True, False):
        try:
            return pipeline(
                task,
                model=model_name,
                device=torch_device,
                top_k=top_k,
                truncation=True,
                local_files_only=local_only,
            )
        except Exception as exc:
            if local_only:
                log.debug("%s not in the local HF cache; trying a download", model_name)
                continue
            log.warning(
                "could not load %s (%s). This scorer will write nulls with reason "
                "'%s'; download it once with `python -m modeling.cli warm-cache`.",
                model_name,
                type(exc).__name__,
                SkipReason.MODEL_UNAVAILABLE,
            )
            return None
    return None


def label_scores(prediction: Any) -> dict[str, float]:
    """Normalize a transformers classification output to ``{label: score}``.

    The shape varies by pipeline and version: a dict, a list of dicts, or a
    list-of-lists when ``top_k`` is set. Normalizing once here keeps three
    scorers from each getting it subtly wrong.
    """
    if isinstance(prediction, dict):
        return {str(prediction["label"]).lower(): float(prediction["score"])}
    if isinstance(prediction, list):
        if prediction and isinstance(prediction[0], list):
            prediction = prediction[0]
        return {
            str(item["label"]).lower(): float(item["score"])
            for item in prediction
            if isinstance(item, dict)
        }
    return {}


def iter_batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def warm_cache(model_names: Iterable[str]) -> dict[str, bool]:
    """Pre-download the aux models so later runs work offline.

    Backs ``modeling.cli warm-cache``. Separated from scoring on purpose: a
    scoring run should not be the thing that discovers it needs 1.5 GB of
    weights.
    """
    results: dict[str, bool] = {}
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        log.error("transformers is not installed; install the 'modeling' extra")
        return {name: False for name in model_names}
    for name in model_names:
        try:
            AutoTokenizer.from_pretrained(name)
            AutoModelForSequenceClassification.from_pretrained(name)
            results[name] = True
            log.info("cached %s", name)
        except Exception as exc:
            results[name] = False
            log.error("could not cache %s: %s", name, exc)
    return results


def cache_root(settings: ModelingSettings | None = None) -> Path:
    return (settings or get_settings()).cache_dir / "aux"
