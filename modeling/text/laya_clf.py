"""Laya: a System One decision model, scored against `misinfo` rather than instead of it.

Laya is non-autoregressive and generates no text. You hand it a *state* and
*typed questions*, and it answers them in one forward pass with calibrated
probabilities. For this project the relevant primitive is ``noul``: one
question, one calibrated ``P(true)``.

**Why it is here at all.** The corpus is English and Hindi since the India
retarget, and every other text scorer lists only ``en`` under ``languages`` in
``configs/models.yaml`` -- Hindi records are collected, normalized, stored,
and then skipped by every other text scorer. Laya ships a multilingual
checkpoint (mmBERT-base, 100+ languages), which is the only thing in this
project capable of closing that gap. The English checkpoint exists here as the
control: if Laya cannot match ``misinfo`` on English, its Hindi numbers are not worth much either.

**It does not replace anything.** `modeling/eval/` compares it against the
roberta-base fine-tune and the TF-IDF baseline on identical splits. A second
model nobody compared to the first is not a result.

Three things inverted from every other scorer in this tree, each on purpose:

* **Not English-gated.** `modeling/aux/base.py` gates on language because an
  English toxicity model run on German text "produces a number, and that
  number is noise". Laya multilingual is the exception that the gate exists to
  make possible, so its own list is `en` and `hi`, and it routes by language.
* **Fine-tuning is mandatory.** Vendor-reported, the base checkpoint scores
  0.362 on typed-decisions against a 0.461 majority-class baseline -- *worse
  than guessing*. A zero-shot number is a baseline to report, never a model to
  ship, and :meth:`LayaScorer.score` says so in its outcome metadata.
* **Calibration is two things.** ``laya.train.finetune`` fits one temperature per
  question inside the checkpoint, and :meth:`LayaScorer.score` returns that value.
  It is not the contract of ``record_scores.misinfo_prob``. The batch stage in
  ``modeling/scoring.py`` writes only the output of a ``Calibrator`` fitted on
  validation scores and stored beside the checkpoint. See ``laya_calibration``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modeling.aux.base import ScoringOutcome, SkipReason
from modeling.config import ModelingSettings, get_settings, module_config

log = logging.getLogger(__name__)

#: Minimum characters before a claim is worth asking about. Below this the
#: state is a fragment and the answer is about the fragment, not the claim.
MIN_CHARS = 20


class LayaSkipReason(SkipReason):
    """Adds the reasons specific to a routed, multi-checkpoint scorer."""

    #: No checkpoint is configured for this record's language.
    UNROUTED_LANGUAGE = "unrouted_language"
    #: `lang` is null and `score_unknown_language` is off, so there is no
    #: defensible checkpoint to pick.
    UNKNOWN_LANGUAGE = "unknown_language"
    #: A fine-tuned checkpoint directory is configured but does not exist.
    #: The base checkpoint is never substituted for it.
    CHECKPOINT_MISSING = "checkpoint_missing"
    #: The scoring stage found no calibrator beside the checkpoint, or the file is
    #: unreadable, or it was rejected. See ``laya_calibration``. The raw score is
    #: never written as ``misinfo_prob``.
    CALIBRATOR_MISSING = "calibrator_missing"
    CALIBRATOR_UNREADABLE = "calibrator_unreadable"
    CALIBRATOR_REJECTED = "calibrator_rejected"
    #: The record has no usable title. The stage scores titles only.
    NO_TITLE = "no_title"
    #: The calibrator file is marked demo, or was fitted for other weights.
    CALIBRATOR_DEMO = "calibrator_demo"
    CALIBRATOR_CHECKPOINT_MISMATCH = "calibrator_checkpoint_mismatch"
    #: The raw score is not a finite number in [0, 1].
    INVALID_SCORE = "invalid_score"


class LayaCheckpointMissing(FileNotFoundError):
    """A configured fine-tuned checkpoint directory does not exist."""


@dataclass
class LayaAvailability:
    """Whether this module can run, and precisely why not."""

    ok: bool
    reason: str = ""
    laya_version: str | None = None

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.ok


@dataclass
class LayaScorer:
    """Scores records with Laya, routing by language to a checkpoint.

    Lazy-imports ``laya`` and degrades with a clear message, matching how
    `modeling/media/*` treats the deepfake tree: a missing optional dependency
    marks the module unavailable rather than breaking an import chain.
    """

    module: str = "laya"
    settings: ModelingSettings | None = None
    _config: dict[str, Any] = field(default_factory=dict, repr=False)
    _agents: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        self._config = self._config or module_config(self.module)

    # --- availability ----------------------------------------------------
    def available(self) -> LayaAvailability:
        """Whether ``laya`` is importable. Never raises."""
        try:
            import laya
        except ImportError as exc:
            return LayaAvailability(
                ok=False,
                reason=(
                    f"laya is not installed ({exc}). Install with "
                    '`pip install -e ".[laya]"`. The module is marked '
                    "unavailable and no record is scored; nothing is guessed."
                ),
            )
        return LayaAvailability(ok=True, laya_version=getattr(laya, "__version__", None))

    # --- the question ----------------------------------------------------
    def question(self) -> dict[str, dict[str, Any]]:
        """The typed question, in the shape ``predict`` accepts.

        ``noul`` returns ``P(true)`` **whatever** ``labels`` renders, so
        ``criteria["true"]`` has to describe the *false* claim for this column
        to mean the same thing as ``misinfo_prob``. See the config, and
        :func:`polarity_note`.
        """
        spec = dict(self._config.get("question") or {})
        question_id = spec.pop("id", "misinfo")
        criteria = spec.get("criteria") or {}
        # Collapse the YAML folded-block whitespace; the model sees this text.
        if isinstance(criteria, dict):
            spec["criteria"] = {k: " ".join(str(v).split()) for k, v in criteria.items()}
        if "instructions" in spec:
            spec["instructions"] = " ".join(str(spec["instructions"]).split())
        return {question_id: spec}

    @property
    def question_id(self) -> str:
        return str((self._config.get("question") or {}).get("id") or "misinfo")

    # --- routing ---------------------------------------------------------
    def _route_key(self, lang: str | None) -> str | None:
        route = self._config.get("route") or {}
        return route.get((lang or "").lower()) or None

    def finetuned_path_for(self, lang: str | None) -> str | None:
        """The configured fine-tuned directory for a language, or None if unset."""
        tuned = (self._config.get("finetuned_checkpoints") or {}).get((lang or "").lower())
        return str(tuned) if tuned else None

    def checkpoint_for(self, lang: str | None) -> tuple[str | None, str | None]:
        """``(checkpoint, subfolder)`` for a language, or ``(None, None)``.

        Routing is config, not code: Hindi has no business going through the
        English encoder, and that belongs in a file someone can read rather
        than in a branch someone has to find.

        A language with a configured fine-tuned directory gets that directory
        and no subfolder. A configured directory that is not there raises
        :class:`LayaCheckpointMissing`. Falling back to the base checkpoint
        would publish zero-shot scores under the name of a fine-tune.
        """
        key = self._route_key(lang)
        if not key:
            return None, None
        tuned = self.finetuned_path_for(lang)
        if tuned:
            path = Path(tuned).expanduser()
            if not path.is_dir():
                raise LayaCheckpointMissing(
                    f"finetuned_checkpoints[{(lang or '').lower()}] is {tuned!r}, "
                    "which is not a directory. Fix the path or set it to null to "
                    "use the base checkpoint."
                )
            return str(path), None
        checkpoint = self._config.get(key)
        if not checkpoint:
            return None, None
        subfolder = None
        if key == "multilingual_checkpoint":
            subfolder = self._config.get("multilingual_subfolder")
        return str(checkpoint), (str(subfolder) if subfolder else None)

    def model_version_for(self, lang: str | None) -> str:
        """Identifies the model behind a language's scores, for ``model_versions``."""
        version = self._config.get("version", "v0.0.0-unset")
        tuned = self.finetuned_path_for(lang)
        if tuned:
            return f"{version}+finetuned:{Path(tuned).name}"
        revision = self._config.get("revision")
        base = f"{version}+base"
        return f"{base}@{str(revision)[:12]}" if revision else base

    def agent_for(self, lang: str | None) -> Any:
        """Load (and cache) the agent for a language. Raises if unroutable."""
        checkpoint, subfolder = self.checkpoint_for(lang)
        if not checkpoint:
            raise KeyError(f"no checkpoint routed for lang={lang!r}")
        # The pin applies to Hub snapshots only. A local fine-tune has no
        # revision, and laya ignores one for a directory that exists.
        revision = None if self.finetuned_path_for(lang) else self._config.get("revision")
        cache_key = f"{checkpoint}#{subfolder or ''}@{revision or ''}"
        if cache_key not in self._agents:
            import laya

            kwargs: dict[str, Any] = {"device": self._config.get("device", "auto")}
            if subfolder:
                kwargs["subfolder"] = subfolder
            if revision:
                kwargs["revision"] = str(revision)
            label = f"{checkpoint} ({subfolder})" if subfolder else checkpoint
            log.info("loading laya checkpoint %s revision=%s", label, revision)
            self._agents[cache_key] = laya.load(checkpoint, **kwargs)
        return self._agents[cache_key]

    def max_len_for(self, lang: str | None) -> int:
        if self._route_key(lang) == "multilingual_checkpoint":
            return int(self._config.get("multilingual_max_len", 1024))
        return int(self._config.get("max_len", 512))

    def batch_size(self) -> int:
        """Configured states per forward pass. Below 1 is refused, not clamped."""
        size = self._config.get("batch_size", 16)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"laya batch_size must be a positive integer, got {size!r}")
        return size

    # --- scoring ---------------------------------------------------------
    def _is_long(self, text: str, lang: str | None) -> bool:
        strategy = self._config.get("long_document_strategy", "predict_long")
        # Rough token estimate; Devanagari is denser per character than Latin,
        # so this is deliberately pessimistic rather than tight.
        return strategy == "predict_long" and len(text) > self.max_len_for(lang) * 2

    def _extract(self, result: Any) -> float:
        answer = ((result or {}).get("answers") or {}).get(self.question_id) or {}
        if "noul" not in answer:
            raise ValueError(
                f"laya returned no `noul` for question {self.question_id!r}; "
                f"got keys {sorted(answer)}. Is the question still type=noul?"
            )
        return float(answer["noul"])

    def score_one(self, agent: Any, text: str, lang: str | None) -> float:
        """One calibrated ``P(claim is false)``.

        Long states go through ``predict_long`` rather than ``predict``:
        ``predict`` truncates anything over ``max_len`` to a single window and
        **silently drops the rest**, and fact-check articles in this corpus run
        2,000-7,000 characters. ``predict_long`` scans overlapping windows and,
        for a ``noul``, takes the max over them -- the claim holds if any window
        supports it.
        """
        questions = self.question()
        if self._is_long(text, lang) and hasattr(agent, "predict_long"):
            result = agent.predict_long(text, questions)
        else:
            result = agent.predict(text, questions, max_len=self.max_len_for(lang))
        return self._extract(result)

    def _score_group(
        self, agent: Any, items: list[tuple[str, str]], lang: str | None, outcome: ScoringOutcome
    ) -> None:
        """Score ``(record_id, text)`` pairs that share a checkpoint.

        ``predict_batch`` truncates each state to ``max_len`` and does not scan
        windows, so long states stay on the per-record ``predict_long`` path.
        A failed batch call is retried one record at a time, so one bad state
        costs one record and carries its own reason.
        """
        short: list[tuple[str, str]] = []
        for record_id, text in items:
            if hasattr(agent, "predict_batch") and not self._is_long(text, lang):
                short.append((record_id, text))
            else:
                self._score_single(agent, record_id, text, lang, outcome)

        size = self.batch_size()
        questions = self.question()
        max_len = self.max_len_for(lang)
        for start in range(0, len(short), size):
            chunk = short[start : start + size]
            try:
                results = agent.predict_batch(
                    [text for _, text in chunk], questions, batch_size=size, max_len=max_len
                )
                if len(results) != len(chunk):
                    raise ValueError(
                        f"predict_batch returned {len(results)} results for {len(chunk)} states"
                    )
            except Exception as exc:
                log.warning(
                    "laya batch of %d failed (%s); scoring its records one at a time",
                    len(chunk),
                    exc,
                )
                for record_id, text in chunk:
                    self._score_single(agent, record_id, text, lang, outcome)
                continue
            for (record_id, _), result in zip(chunk, results, strict=True):
                try:
                    outcome.values[record_id] = self._extract(result)
                except Exception as exc:
                    log.warning("laya batch result unusable for %s: %s", record_id, exc)
                    outcome.skipped[record_id] = LayaSkipReason.INFERENCE_ERROR

    def _score_single(
        self, agent: Any, record_id: str, text: str, lang: str | None, outcome: ScoringOutcome
    ) -> None:
        try:
            outcome.values[record_id] = self.score_one(agent, text, lang)
        except Exception as exc:
            log.warning("laya inference failed for %s: %s", record_id, exc)
            outcome.skipped[record_id] = LayaSkipReason.INFERENCE_ERROR

    def score(self, records: Any) -> ScoringOutcome:
        """Score a DataFrame of records. ``values`` is ``record_id -> P(false)``.

        Every absent record appears in ``skipped`` with a reason. The two
        together are exhaustive over the input, matching the aux-scorer
        contract.
        """
        outcome = ScoringOutcome()
        availability = self.available()
        if not availability.ok:
            log.warning("%s", availability.reason)
            for record_id in records["id"]:
                outcome.skipped[str(record_id)] = LayaSkipReason.MODEL_UNAVAILABLE
            return outcome

        score_unknown = bool(getattr(self.settings, "score_unknown_language", False))
        groups: dict[str, list[tuple[str, str]]] = {}
        for row in records.itertuples():
            record_id = str(getattr(row, "id", ""))
            text = (getattr(row, "text", "") or "").strip()
            lang = getattr(row, "lang", None)

            if not text:
                outcome.skipped[record_id] = LayaSkipReason.EMPTY_TEXT
                continue
            if len(text) < MIN_CHARS:
                outcome.skipped[record_id] = LayaSkipReason.TEXT_TOO_SHORT
                continue
            # NaN as well as None: a null string column read back from Parquet
            # arrives as NaN, and str(NaN) would be routed as the language "nan".
            if lang is None or lang != lang:
                if not score_unknown:
                    outcome.skipped[record_id] = LayaSkipReason.UNKNOWN_LANGUAGE
                    continue
                # Assuming English is recorded as an assumption, exactly as
                # modeling/config.py's `score_unknown_language` requires.
                lang = "en"
            lang = str(lang).lower()
            if not self.settings.language_allowed("laya", lang):
                outcome.skipped[record_id] = LayaSkipReason.UNROUTED_LANGUAGE
                continue
            try:
                checkpoint, _ = self.checkpoint_for(lang)
            except LayaCheckpointMissing as exc:
                log.warning("%s", exc)
                outcome.skipped[record_id] = LayaSkipReason.CHECKPOINT_MISSING
                continue
            if not checkpoint:
                outcome.skipped[record_id] = LayaSkipReason.UNROUTED_LANGUAGE
                continue
            groups.setdefault(lang, []).append((record_id, text))

        for lang, items in groups.items():
            try:
                agent = self.agent_for(lang)
            except Exception as exc:
                log.warning("laya checkpoint for lang=%s failed to load: %s", lang, exc)
                for record_id, _ in items:
                    outcome.skipped[record_id] = LayaSkipReason.INFERENCE_ERROR
                continue
            self._score_group(agent, items, lang, outcome)
        return outcome


class LayaTrainingExporter:
    """Writes the fine-tuning JSONL, using the *same* question the scorer asks.

    **Do not use laya's CSV path for this.** ``laya.train.read_data`` sends a
    ``.csv`` to ``rows_from_csv``, which "discovers distinct labels in
    `label_column` ... and builds a **choice** question with those labels" --
    so a CSV of ``text,label`` with labels ``false``/``true`` trains a two-option
    *choice* head. This project asks a ``noul`` at inference. Training one head
    and querying another degrades quietly rather than erroring, and the
    resulting numbers would look like a weak model instead of a wiring mistake.

    So the JSONL path is used, and the question comes from
    :meth:`LayaScorer.question` rather than being written out again here. One
    definition, read by both sides, is the only way train/inference drift
    cannot happen.

    The row schema, from ``laya.train.items_from_rows``::

        {"state": ..., "questions": {...}, "expected": {"<qid>": <bool>}}

    ``expected`` for a ``noul`` is a **boolean**, and ``True`` means the answer
    to the question is yes. The question asks whether the claim is false, so
    ``True`` is the misinformation class -- matching ``misinfo``'s
    ``label = (binary == "false")`` in modeling/datasets/liar.py.

    Soft targets are available instead, via
    ``{"gold": {"<qid>": {"probabilities": {"false": p, "true": q}}}}``, which
    is the form to reach for if a label source ever carries confidence. The
    benchmarks do not, so this writes hard booleans.
    """

    def __init__(self, scorer: LayaScorer | None = None):
        self.scorer = scorer or LayaScorer()
        #: Rows the last :meth:`export` call skipped, by reason code.
        self.skipped: dict[str, int] = {}

    def row(self, text: str, is_misinformation: bool) -> dict[str, Any]:
        return {
            "state": text,
            "questions": self.scorer.question(),
            "expected": {self.scorer.question_id: bool(is_misinformation)},
        }

    def export(self, frame: Any, path: Any, label_column: str = "label") -> int:
        """Write one JSONL row per labelled record. Returns the row count.

        ``label_column`` is read with the benchmark convention this project
        already uses: truthy / ``1`` means misinformation. See liar.py.

        A row with blank text (``empty_text``) or a missing label
        (``null_label``) is skipped. Skips are counted per reason in
        ``self.skipped``, which each call resets, and logged at warning level.
        """
        import json as _json
        from pathlib import Path as _Path

        out = _Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        skipped = {"empty_text": 0, "null_label": 0}
        with out.open("w", encoding="utf-8") as handle:
            for record in frame.itertuples():
                text = (getattr(record, "text", "") or "").strip()
                if not text:
                    skipped["empty_text"] += 1
                    continue
                label = getattr(record, label_column, None)
                # `label != label` is the NaN test. pandas stores a missing
                # value in a numeric column as NaN, not None, so an `is None`
                # check passes it through and `int(nan)` then raises. An
                # unlabelled row must be skipped, never coerced -- guessing a
                # class here would put fabricated labels in the training set.
                if label is None or label != label:
                    skipped["null_label"] += 1
                    continue
                handle.write(
                    _json.dumps(self.row(text, bool(int(label))), ensure_ascii=False) + "\n"
                )
                written += 1
        self.skipped = {reason: n for reason, n in skipped.items() if n}
        if self.skipped:
            log.warning("skipped rows exporting laya training data to %s: %s", out, self.skipped)
        log.info("wrote %d laya training rows to %s", written, out)
        return written


def polarity_note() -> str:
    """The one sentence that must stay true, quoted in the model card.

    Kept as code rather than a comment so a test can assert the contract it
    describes, in the spirit of tests/test_label_orientation.py: "resolve it,
    or refuse. Never guess."
    """
    return (
        "laya's `noul` returns P(true), and `criteria.true` describes the FALSE "
        "claim, so this column is P(claim is false) -- the same orientation as "
        "misinfo_prob, which is P(label 1) where LABEL_NAMES = "
        '["not-misinformation", "misinformation-like"].'
    )
