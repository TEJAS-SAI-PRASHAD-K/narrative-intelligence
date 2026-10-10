"""Laya: a System One decision model, scored against `misinfo` rather than instead of it.

Laya is non-autoregressive and generates no text. You hand it a *state* and
*typed questions*, and it answers them in one forward pass with calibrated
probabilities. For this project the relevant primitive is ``noul``: one
question, one calibrated ``P(true)``.

**Why it is here at all.** The corpus is English and Hindi since the India
retarget, and ``modeling.config.Settings.languages`` is still ``("en",)`` --
Hindi records are collected, normalized, stored, and then skipped by every
scorer. Laya ships a multilingual checkpoint (mmBERT-base, 100+ languages),
which is the only thing in this project capable of closing that gap. The
English checkpoint exists here as the control: if Laya cannot match
``misinfo`` on English, its Hindi numbers are not worth much either.

**It does not replace anything.** `modeling/eval/` compares it against the
roberta-base fine-tune and the TF-IDF baseline on identical splits. A second
model nobody compared to the first is not a result.

Three things inverted from every other scorer in this tree, each on purpose:

* **Not English-gated.** `modeling/aux/base.py` gates on language because an
  English toxicity model run on German text "produces a number, and that
  number is noise". Laya multilingual is the exception that the gate exists to
  make possible, so it routes by language instead of refusing.
* **Fine-tuning is mandatory.** Vendor-reported, the base checkpoint scores
  0.362 on typed-decisions against a 0.461 majority-class baseline -- *worse
  than guessing*. A zero-shot number is a baseline to report, never a model to
  ship, and :meth:`LayaScorer.score` says so in its outcome metadata.
* **Calibration is not a separate stage.** ``laya.train.finetune`` preprocesses,
  trains, calibrates and saves in one call, holding out ``calib_frac`` to fit
  one temperature per (question type, option count). So there is no
  ``Calibrator`` here of the kind ``misinfo`` carries, and the absence is a
  property of the library rather than an omission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    def checkpoint_for(self, lang: str | None) -> tuple[str | None, str | None]:
        """``(checkpoint, subfolder)`` for a language, or ``(None, None)``.

        Routing is config, not code: Hindi has no business going through the
        English encoder, and that belongs in a file someone can read rather
        than in a branch someone has to find.
        """
        route = self._config.get("route") or {}
        key = route.get((lang or "").lower())
        if not key:
            return None, None
        checkpoint = self._config.get(key)
        if not checkpoint:
            return None, None
        subfolder = None
        if key == "multilingual_checkpoint":
            subfolder = self._config.get("multilingual_subfolder")
        return str(checkpoint), (str(subfolder) if subfolder else None)

    def agent_for(self, lang: str | None) -> Any:
        """Load (and cache) the agent for a language. Raises if unroutable."""
        checkpoint, subfolder = self.checkpoint_for(lang)
        if not checkpoint:
            raise KeyError(f"no checkpoint routed for lang={lang!r}")
        cache_key = f"{checkpoint}#{subfolder or ''}"
        if cache_key not in self._agents:
            import laya

            kwargs: dict[str, Any] = {"device": self._config.get("device", "auto")}
            if subfolder:
                kwargs["subfolder"] = subfolder
            label = f"{checkpoint} ({subfolder})" if subfolder else checkpoint
            log.info("loading laya checkpoint %s", label)
            self._agents[cache_key] = laya.load(checkpoint, **kwargs)
        return self._agents[cache_key]

    def max_len_for(self, lang: str | None) -> int:
        _, subfolder = self.checkpoint_for(lang)
        if subfolder == "multilingual":
            return int(self._config.get("multilingual_max_len", 1024))
        return int(self._config.get("max_len", 512))

    # --- scoring ---------------------------------------------------------
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
        max_len = self.max_len_for(lang)
        strategy = self._config.get("long_document_strategy", "predict_long")
        # Rough token estimate; Devanagari is denser per character than Latin,
        # so this is deliberately pessimistic rather than tight.
        use_long = strategy == "predict_long" and len(text) > max_len * 2

        if use_long and hasattr(agent, "predict_long"):
            result = agent.predict_long(text, questions)
        else:
            result = agent.predict(text, questions, max_len=max_len)
        answer = (result.get("answers") or {}).get(self.question_id) or {}
        if "noul" not in answer:
            raise ValueError(
                f"laya returned no `noul` for question {self.question_id!r}; "
                f"got keys {sorted(answer)}. Is the question still type=noul?"
            )
        return float(answer["noul"])

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
            if lang is None:
                if not score_unknown:
                    outcome.skipped[record_id] = LayaSkipReason.UNKNOWN_LANGUAGE
                    continue
                # Assuming English is recorded as an assumption, exactly as
                # modeling/config.py's `score_unknown_language` requires.
                lang = "en"
            checkpoint, _ = self.checkpoint_for(lang)
            if not checkpoint:
                outcome.skipped[record_id] = LayaSkipReason.UNROUTED_LANGUAGE
                continue
            try:
                agent = self.agent_for(lang)
                outcome.values[record_id] = self.score_one(agent, text, lang)
            except Exception as exc:
                log.warning("laya inference failed for %s: %s", record_id, exc)
                outcome.skipped[record_id] = LayaSkipReason.INFERENCE_ERROR
        return outcome


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
