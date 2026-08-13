"""Narrative labels and summaries. The only LLM call in the product path.

**The policy, restated because it is easy to erode.** An LLM is permitted here
for exactly two bounded tasks: labelling a narrative cluster, and extracting a
claim from a noisy representative post. It is forbidden as a classifier for
misinformation, bot-likeness, toxicity, sentiment, stance or deepfake. The
reasons are not stylistic:

* it is not reproducible run to run, so a reported metric cannot be defended;
* there is no train/test methodology to report;
* cost scales with corpus size, and the corpus is the thing that grows;
* a grader cannot audit it.

Every score that appears in the product comes from a trainable model with
reportable metrics. This module produces *text*, not scores.

**Bounded by construction.** One call per cluster, not per post. Cached by
centroid hash, so a rerun over unchanged clusters costs nothing. Hard ceiling on
calls per run. Token usage logged so the cost is reportable.

**The pipeline completes with no API key.** Without one, ``label_source`` becomes
``"centroid"`` and the label is the representative post's first sentence,
truncated. That is a worse label and an honest one, and it is the path the test
suite exercises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from modeling.config import ModelingSettings, get_settings, module_config
from modeling.io import as_list

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You label clusters of social media posts that share a claim.

You will receive 3-5 posts from one cluster. Return STRICT JSON with exactly two \
keys:
  "label": a one-line claim-style headline, at most 12 words
  "summary": 2-4 sentences describing what the posts claim and how they frame it

Hard rules:
- NEVER assert the claim as true. You are describing what posts say, not \
adjudicating it.
- Use reported speech: "Posts claim that...", "Users allege...", "Several \
accounts say...".
- NEVER invent details, names, numbers, dates or sources that are not in the \
posts you were given.
- If the posts do not share a single coherent claim, say so in the summary and \
make the label descriptive rather than assertive.
- Return only the JSON object. No preamble, no code fence, no commentary.
"""

CLAIM_PROMPT = """\
Extract the single central claim from this post as one short declarative \
sentence, in reported speech ("The post claims that ..."). Invent nothing. \
Return STRICT JSON with one key: "claim".
"""


@dataclass
class SummaryResult:
    narrative_id: str
    label: str
    summary: str | None
    label_source: str  # "llm" | "centroid" | "manual"
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class SummarizerRun:
    """What one summarization pass cost and produced."""

    results: dict[str, SummaryResult] = field(default_factory=dict)
    llm_calls: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    invalid_json: int = 0

    def cost_note(self, model: str) -> str:
        return (
            f"{self.llm_calls} call(s) to {model}, {self.cache_hits} cache hit(s), "
            f"{self.fallbacks} centroid fallback(s); "
            f"{self.tokens_in} in / {self.tokens_out} out tokens"
        )


class LLMClient:
    """Thin Anthropic wrapper. Absent key -> ``available`` is False, not a crash."""

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        config = module_config("summarize")
        self.model = str(config.get("llm_model", self.settings.llm_model))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 512))
        self.max_retries = int(config.get("max_retries", 2))
        self._client = None
        self.calls = 0

        if not self.settings.anthropic_api_key:
            self.available = False
            log.info(
                "no ANTHROPIC_API_KEY; narrative labels will fall back to centroid text. "
                "This is a supported path, not a failure."
            )
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            self.available = True
        except ImportError:
            self.available = False
            log.warning("anthropic is not installed; install the 'llm' extra or use the fallback")

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any] | None, int, int]:
        """One call, retried only on invalid JSON. Returns (payload, in, out)."""
        if not self.available:
            return None, 0, 0
        if self.calls >= self.settings.llm_max_calls_per_run:
            log.warning(
                "hit the %d-call ceiling for this run; remaining clusters use the fallback",
                self.settings.llm_max_calls_per_run,
            )
            return None, 0, 0

        tokens_in = tokens_out = 0
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except Exception as exc:
                log.warning("LLM call failed (%s); falling back", type(exc).__name__)
                return None, tokens_in, tokens_out

            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
                tokens_out += int(getattr(usage, "output_tokens", 0) or 0)

            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            payload = _parse_json(text)
            if payload is not None:
                return payload, tokens_in, tokens_out
            log.warning(
                "LLM returned invalid JSON (attempt %d/%d)", attempt + 1, self.max_retries + 1
            )

        return None, tokens_in, tokens_out

    def classify(self, prompt: str, text: str) -> bool | None:
        """Zero-shot yes/no, for the *baseline comparison only*.

        Not a product path. See the module docstring and
        ``modeling/eval/baselines.zero_shot_llm``.
        """
        payload, _, _ = self.complete_json(
            prompt, json.dumps({"text": text[:2000]})
        )
        if not payload:
            return None
        answer = payload.get("answer")
        if isinstance(answer, bool):
            return answer
        return str(answer).strip().lower() in {"yes", "true", "1"}


class NarrativeSummarizer:
    module = "summarize"

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.client = LLMClient(settings)
        self._cache_path = self.settings.cache_dir / "summaries.json"
        self._cache: dict[str, dict[str, Any]] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("summary cache corrupt; starting fresh")

    def summarize(
        self, narratives: list[dict[str, Any]], texts_by_record: dict[str, str]
    ) -> SummarizerRun:
        """One call per narrative. Cached by centroid hash."""
        run = SummarizerRun()
        for narrative in narratives:
            narrative_id = str(narrative["narrative_id"])
            representatives = [
                texts_by_record.get(r, "") for r in as_list(narrative.get("representative_ids"))
            ]
            representatives = [t for t in representatives if t and t.strip()][:5]
            if not representatives:
                run.results[narrative_id] = SummaryResult(
                    narrative_id, "(no representative text)", None, "centroid"
                )
                run.fallbacks += 1
                continue

            key = _centroid_key(narrative.get("centroid"), representatives)
            cached = self._cache.get(key)
            if cached:
                run.cache_hits += 1
                run.results[narrative_id] = SummaryResult(
                    narrative_id, cached["label"], cached.get("summary"), cached["label_source"]
                )
                continue

            payload, tokens_in, tokens_out = self.client.complete_json(
                SYSTEM_PROMPT, _render_posts(representatives)
            )
            run.tokens_in += tokens_in
            run.tokens_out += tokens_out

            if payload and payload.get("label"):
                run.llm_calls += 1
                result = SummaryResult(
                    narrative_id=narrative_id,
                    label=str(payload["label"]).strip()[:200],
                    summary=str(payload.get("summary", "")).strip()[:1200] or None,
                    label_source="llm",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
            else:
                if self.client.available:
                    run.invalid_json += 1
                run.fallbacks += 1
                result = SummaryResult(
                    narrative_id=narrative_id,
                    label=centroid_label(representatives[0]),
                    summary=None,
                    label_source="centroid",
                )

            run.results[narrative_id] = result
            self._cache[key] = {
                "label": result.label,
                "summary": result.summary,
                "label_source": result.label_source,
            }

        self._flush()
        log.info("summarize: %s", run.cost_note(self.client.model))
        if run.invalid_json:
            log.warning(
                "%d cluster(s) fell back after %d invalid-JSON attempts each",
                run.invalid_json,
                self.client.max_retries + 1,
            )
        return run

    def _flush(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8"
        )


def centroid_label(text: str, limit: int = 90) -> str:
    """The no-API-key label: the representative post's first sentence.

    Deliberately plain. It is visibly a quotation rather than a generated
    headline, which is the right signal to a reader that no model wrote it.
    """
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    cleaned = re.sub(r"https?://\S+", "", cleaned).strip()
    first = re.split(r"(?<=[.!?])\s+", cleaned)[0] if cleaned else ""
    if len(first) <= limit:
        return first or "(untitled narrative)"
    return first[: limit - 1].rsplit(" ", 1)[0] + "…"


def _render_posts(posts: list[str]) -> str:
    lines = ["Posts from one cluster:", ""]
    for i, post in enumerate(posts, 1):
        collapsed = re.sub(r"\s+", " ", post).strip()[:800]
        lines.append(f"[{i}] {collapsed}")
    return "\n".join(lines)


def _centroid_key(centroid: Any, representatives: list[str]) -> str:
    """Cache key: the centroid plus the representative text.

    Both, not just the centroid: two runs can produce the same centroid from
    different representatives after a re-cluster, and the label is generated
    from the representatives.
    """
    payload = json.dumps(
        {
            "centroid": [round(float(v), 4) for v in as_list(centroid)],
            "posts": [r[:400] for r in representatives],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _parse_json(text: str) -> dict[str, Any] | None:
    """Parse strict JSON, tolerating a code fence the model may add anyway."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: the first {...} block. Not a general fix, just enough to
        # survive a stray sentence before the object.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None
