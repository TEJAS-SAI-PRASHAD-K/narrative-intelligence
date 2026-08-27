"""Citation and hedging validation for generated Compass Context.

**This module is the reason Compass is safe to ship.** Everything else in the
pipeline is retrieval and prompting; this is what decides whether the output is
allowed to be persisted. The rules it enforces are in code, not in the prompt,
because a prompt is a request and a validator is a guarantee.

Three rules, in the order they matter:

1. **Every sentence-level claim must be covered by a citation span.** A context
   whose sentences are not each backed by a retrieved source is an ungrounded
   generation wearing a bibliography.
2. **The narrative's claim is never stated as fact.** The output is a hedged
   analyst note. A fact-check that asserts the thing it is checking has
   laundered a claim into an authoritative voice.
3. **Failure means persisting nothing.** Two failed attempts and the row is
   written with `insufficient_evidence` and an *empty* context. An uncited AI
   fact-check inside a misinformation product is the worst failure this system
   has, and an empty field is honest where a paragraph is not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Sentence splitter. Deliberately simple: abbreviations will occasionally
#: over-split, which costs a false validation failure and a retry. The opposite
#: error -- merging two sentences so one citation appears to cover both -- would
#: let an uncited claim through, so the bias is chosen on purpose.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")

#: Phrasings that assert the claim rather than reporting on it. These are
#: matched against the *context*, and any hit fails the lint: the note may say
#: "reporting indicates X", never "X".
_UNHEDGED = (
    r"\bthis (?:is|was) (?:true|false|a hoax|fabricated|confirmed)\b",
    r"\bin fact,?\s",
    r"\bthe truth is\b",
    r"\bproves? that\b",
    r"\bdefinitively\b",
    r"\bthere is no doubt\b",
    r"\bclearly (?:shows?|demonstrates?)\b",
    r"\bhas been debunked\b(?!\s+by)",
    r"\bwe (?:can )?confirm\b",
    r"\bit is certain\b",
)

#: At least one of these must appear. A note with no hedging language at all is
#: not reporting on evidence, whatever else it does.
_HEDGES = (
    "according to",
    "reporting indicates",
    "reporting suggests",
    "sources indicate",
    "has not confirmed",
    "could not be verified",
    "appears to",
    "analysts should",
    "no evidence",
    "unverified",
    "remains unclear",
    "did not respond",
)

MIN_CITATIONS = 1
#: A context shorter than this is not a note, it is a fragment.
MIN_CONTEXT_CHARS = 80


@dataclass
class ValidationResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    uncovered_sentences: list[str] = field(default_factory=list)
    unhedged_matches: list[str] = field(default_factory=list)
    sentence_count: int = 0
    covered_count: int = 0

    def as_report(self) -> dict[str, Any]:
        """Persisted on the row for the writeup's error analysis.

        "Why did this fail" is otherwise unanswerable after the fact, and the
        failure modes are exactly what a reader of the writeup wants to see.
        """
        return {
            "ok": self.ok,
            "failures": self.failures,
            "sentence_count": self.sentence_count,
            "covered_count": self.covered_count,
            "uncovered_sentences": self.uncovered_sentences[:10],
            "unhedged_matches": self.unhedged_matches[:10],
        }


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Sentences with their character offsets into ``text``."""
    if not text.strip():
        return []
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for part in _SENTENCE.split(text.strip()):
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        out.append((start, end, part))
        cursor = end
    return out


def validate(context: str, citations: list[dict[str, Any]], *, claim: str = "") -> ValidationResult:
    """Decide whether a generated context may be persisted."""
    result = ValidationResult(ok=True)

    if not context or not context.strip():
        result.ok = False
        result.failures.append("empty_context")
        return result

    if len(context.strip()) < MIN_CONTEXT_CHARS:
        result.ok = False
        result.failures.append("context_too_short")

    if len(citations) < MIN_CITATIONS:
        result.ok = False
        result.failures.append("no_citations")
        return result

    # --- rule 1: every sentence covered by a citation span ---------------
    sentences = split_sentences(context)
    result.sentence_count = len(sentences)
    spans = [
        (int(c["char_start"]), int(c["char_end"]))
        for c in citations
        if c.get("char_start") is not None and c.get("char_end") is not None
    ]
    if not spans:
        result.ok = False
        result.failures.append("citations_have_no_spans")
        return result

    for start, end, sentence in sentences:
        # Overlap, not containment: a citation that covers most of a sentence
        # is a real citation, and requiring exact boundaries would fail on
        # trailing whitespace.
        covered = any(span_start < end and span_end > start for span_start, span_end in spans)
        if covered:
            result.covered_count += 1
        else:
            result.uncovered_sentences.append(sentence.strip()[:200])

    if result.uncovered_sentences:
        result.ok = False
        result.failures.append("uncited_sentences")

    # --- rule 2: hedged, and never asserting the claim -------------------
    lowered = context.lower()
    for pattern in _UNHEDGED:
        for match in re.finditer(pattern, lowered):
            result.unhedged_matches.append(match.group(0))
    if result.unhedged_matches:
        result.ok = False
        result.failures.append("unhedged_assertion")

    if not any(hedge in lowered for hedge in _HEDGES):
        result.ok = False
        result.failures.append("no_hedging_language")

    # --- rule 3: the claim must not be restated as fact ------------------
    # The ORIGINAL context, not the lowercased copy: the sentence splitter keys
    # on a following capital letter, so splitting lowercased text yields one
    # giant "sentence" and a hedge anywhere in it masks an unattributed claim
    # everywhere else.
    if claim and _restates_claim(context, claim.lower()):
        result.ok = False
        result.failures.append("restates_claim_as_fact")

    return result


def _restates_claim(context: str, claim: str) -> bool:  # context is NOT lowercased
    """Whether the context asserts the claim in its own voice.

    Approximate by construction: it looks for the claim's distinctive words
    appearing in a sentence with no attribution. A false positive costs a retry;
    a false negative publishes a laundered claim, so the check errs toward
    failing.
    """
    words = [w for w in re.findall(r"[a-z]{5,}", claim) if w not in _STOPWORDS]
    if len(words) < 3:
        return False

    for _, _, sentence in split_sentences(context):
        lowered = sentence.lower()
        hits = sum(1 for word in words if word in lowered)
        if hits < max(3, len(words) // 2):
            continue
        # The claim's substance is here. That is fine if it is attributed and
        # a problem if it is stated flat.
        if not any(hedge in lowered for hedge in _HEDGES):
            return True
    return False


_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "their",
    "there",
    "these",
    "those",
    "which",
    "while",
    "would",
    "could",
    "should",
    "being",
    "other",
    "where",
}


def insufficient_evidence_row(claim: str, report: dict[str, Any], attempts: int) -> dict[str, Any]:
    """The row written when validation fails and the pipeline gives up.

    The row exists so the attempt is auditable; the paragraph does not, because
    the paragraph could not be sourced. `context` is an empty string rather than
    NULL: NULL would be indistinguishable from "not generated yet", and those
    are different states an analyst must be able to tell apart.
    """
    return {
        "claim": claim,
        "context": "",
        "verification_status": "insufficient_evidence",
        "risk": "medium",
        "caution_note": (
            "Retrieval could not source this claim well enough to support a cited "
            "note, so none was written. The absence of sources is not evidence that "
            "the claim is false, and it is not evidence that it is true."
        ),
        "validation_report": report,
        "attempts": attempts,
    }
