"""Compass Context generation: retrieve, generate, validate, persist or refuse.

The loop is deliberately short and the refusal path is deliberately cheap:

    retrieve -> (too few docs?) -> insufficient_evidence, done
             -> generate -> validate -> ok?     -> persist
                                     -> failed? -> regenerate once
                                                -> failed again? ->
                                                   insufficient_evidence, done

Two attempts, then stop. A third would be the model talking itself into
something, and the whole point of the validator is that we do not negotiate with
an ungrounded generation.

Regeneration **inserts** and marks the previous row `superseded_by`. Nothing is
ever mutated: an analyst has to be able to say what the system claimed and when.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.compass.retrieve import MIN_DOCUMENTS, RetrievedDocument, retrieve
from app.compass.validate import insufficient_evidence_row, validate

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


def _prompt_template() -> str:
    from pathlib import Path

    return (Path(__file__).parent / "prompts" / "context_v1.md").read_text(encoding="utf-8")


def generate_for_narrative(
    session: Session, project_id, narrative_id: uuid.UUID, *, max_attempts: int = 2
) -> dict[str, Any]:
    """Generate and persist one Compass Context. Always writes exactly one row."""
    from app.config import get_api_settings

    settings = get_api_settings()

    narrative = session.execute(
        text("SELECT id, title, claim, summary FROM narratives WHERE id = :n"),
        {"n": narrative_id},
    ).first()
    if narrative is None:
        raise ValueError(f"no narrative {narrative_id}")

    # The claim is what gets fact-checked. Falling back to the title is right --
    # it is the shortest statement of what the cluster asserts -- but the
    # summary is a description of the cluster's behaviour, not a claim, so it is
    # not used.
    claim = (narrative.claim or narrative.title or "").strip()
    if not claim:
        return _persist(
            session,
            narrative_id,
            insufficient_evidence_row(
                claim="",
                report={"failures": ["narrative_has_no_claim"]},
                attempts=0,
            ),
            [],
            model=settings.llm_model,
            retrieved=0,
        )

    documents = retrieve(session, project_id, narrative_id)
    if len(documents) < MIN_DOCUMENTS:
        # Refusing here rather than generating over one passage. The validator
        # would reject it anyway, and this saves an LLM call to be told so.
        return _persist(
            session,
            narrative_id,
            insufficient_evidence_row(
                claim=claim,
                report={
                    "failures": ["insufficient_retrieval"],
                    "documents_found": len(documents),
                    "minimum_required": MIN_DOCUMENTS,
                },
                attempts=0,
            ),
            [],
            model=settings.llm_model,
            retrieved=len(documents),
        )

    client = _llm_client()
    if client is None:
        return _persist(
            session,
            narrative_id,
            insufficient_evidence_row(
                claim=claim,
                report={
                    "failures": ["no_llm_configured"],
                    "detail": (
                        "ANTHROPIC_API_KEY is unset. Compass generates over retrieved "
                        "sources or not at all; there is no heuristic fallback, because "
                        "a keyword-assembled fact-check would be worse than none."
                    ),
                },
                attempts=0,
            ),
            [],
            model=settings.llm_model,
            retrieved=len(documents),
        )

    last_report: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        raw = _call(client, claim, documents, settings.llm_model)
        if raw is None:
            last_report = {"failures": ["llm_call_failed"]}
            continue

        context = (raw.get("context") or "").strip()
        if raw.get("verification_status") == "insufficient_evidence" and not context:
            # The model refused, which is a correct answer and one the prompt
            # explicitly asks for. Take it at its word rather than retrying.
            return _persist(
                session,
                narrative_id,
                insufficient_evidence_row(
                    claim=claim,
                    report={"failures": ["model_declined"], "attempts": attempt},
                    attempts=attempt,
                ),
                [],
                model=settings.llm_model,
                retrieved=len(documents),
            )

        citations = _resolve_citations(raw.get("citations") or [], documents, context)
        result = validate(context, citations, claim=claim)
        last_report = result.as_report()

        if result.ok:
            log.info("compass: narrative %s validated on attempt %d", narrative_id, attempt)
            return _persist(
                session,
                narrative_id,
                {
                    "claim": claim,
                    "context": context,
                    "verification_status": raw.get("verification_status") or "unverified",
                    "risk": raw.get("risk") or "medium",
                    "caution_note": raw.get("caution_note"),
                    "validation_report": last_report,
                    "attempts": attempt,
                },
                citations,
                model=settings.llm_model,
                retrieved=len(documents),
            )

        log.warning(
            "compass: narrative %s failed validation on attempt %d: %s",
            narrative_id,
            attempt,
            result.failures,
        )

    # Both attempts failed validation. Persist the refusal, not the paragraph.
    return _persist(
        session,
        narrative_id,
        insufficient_evidence_row(claim=claim, report=last_report, attempts=max_attempts),
        [],
        model=settings.llm_model,
        retrieved=len(documents),
    )


def _llm_client():
    from app.config import get_api_settings

    settings = get_api_settings()
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic

        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    except ImportError:
        log.warning("anthropic is not installed; compass cannot generate")
        return None


def _call(client, claim: str, documents: list[RetrievedDocument], model: str):
    """One generation. Returns the parsed JSON, or None on any failure."""
    passages = "\n\n".join(doc.as_passage(i + 1) for i, doc in enumerate(documents))
    prompt = _prompt_template().format(claim=claim, passages=passages)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            # Zero temperature: this is an evidence-summarising task with a
            # right answer shape, not a creative one, and a reproducible output
            # is worth more than variety when the next step is validation.
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        body = "".join(block.text for block in response.content if hasattr(block, "text"))
    except Exception as exc:
        log.error("compass LLM call failed: %s: %s", type(exc).__name__, exc)
        return None

    return _parse_json(body)


def _parse_json(body: str) -> dict[str, Any] | None:
    """Extract the JSON object, tolerating a fenced code block around it."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.DOTALL)
    candidate = fenced.group(1) if fenced else body
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        log.warning("compass: could not parse the model's JSON: %s", exc)
        return None


def _resolve_citations(
    raw_citations: list[dict[str, Any]], documents: list[RetrievedDocument], context: str
) -> list[dict[str, Any]]:
    """Turn the model's `{source, supports}` pairs into spans into ``context``.

    The span is found by locating the quoted sentence in the context. A citation
    whose `supports` text is not actually in the context is dropped, not
    guessed at: a fabricated quote is exactly the failure the span requirement
    exists to catch, and inventing an offset for it would defeat the check.
    """
    resolved: list[dict[str, Any]] = []
    for item in raw_citations:
        try:
            index = int(item.get("source", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= index < len(documents)):
            continue

        supports = (item.get("supports") or "").strip()
        start = context.find(supports) if supports else -1
        if start < 0 and supports:
            # Try a prefix: models frequently paraphrase the tail of their own
            # sentence when quoting it back.
            probe = supports[:60]
            start = context.find(probe) if probe else -1
            if start >= 0:
                supports = context[start : start + len(supports)]
        if start < 0:
            log.debug("compass: citation quote not found in context; dropping it")
            continue

        document = documents[index]
        resolved.append(
            {
                "url": document.url or f"corpus://{document.post_id}",
                "title": document.title,
                "publisher": document.publisher,
                "domain": document.domain,
                "retrieved_at": None,
                "snippet": document.snippet[:2000],
                "char_start": start,
                "char_end": start + len(supports),
            }
        )
    return resolved


def _persist(
    session: Session,
    narrative_id: uuid.UUID,
    row: dict[str, Any],
    citations: list[dict[str, Any]],
    *,
    model: str,
    retrieved: int,
) -> dict[str, Any]:
    """Insert a new context and supersede the previous one.

    Never an UPDATE. The audit trail of what the system claimed, and when, is
    the deliverable -- an analyst who acted on last week's context has to be
    able to see it even after a regeneration disagreed with it.
    """
    from app.db import utcnow

    context_id = uuid.uuid4()
    session.execute(
        text(
            """
            INSERT INTO compass_contexts (
                id, narrative_id, claim, context, verification_status, risk,
                caution_note, model, prompt_version, generated_at, attempts,
                retrieved_document_count, validation_report
            ) VALUES (
                :id, :narrative_id, :claim, :context, :verification_status, :risk,
                :caution_note, :model, :prompt_version, :generated_at, :attempts,
                :retrieved, cast(:validation_report AS jsonb)
            )
            """
        ),
        {
            "id": context_id,
            "narrative_id": narrative_id,
            "claim": row["claim"],
            "context": row["context"],
            "verification_status": row["verification_status"],
            "risk": row["risk"],
            "caution_note": row.get("caution_note"),
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "generated_at": utcnow(),
            "attempts": row.get("attempts", 1),
            "retrieved": retrieved,
            "validation_report": json.dumps(row.get("validation_report") or {}, default=str),
        },
    )

    for citation in citations:
        session.execute(
            text(
                """
                INSERT INTO compass_citations (
                    context_id, url, title, publisher, domain, retrieved_at,
                    snippet, char_start, char_end
                ) VALUES (
                    :context_id, :url, :title, :publisher, :domain, :retrieved_at,
                    :snippet, :char_start, :char_end
                )
                """
            ),
            {**citation, "context_id": context_id, "retrieved_at": utcnow()},
        )

    # Supersede every earlier live context for this narrative. Done after the
    # insert so there is never a moment with no live row.
    session.execute(
        text(
            """
            UPDATE compass_contexts SET superseded_by = :new
            WHERE narrative_id = :n AND id <> :new AND superseded_by IS NULL
            """
        ),
        {"new": context_id, "n": narrative_id},
    )

    return {
        "context_id": str(context_id),
        "verification_status": row["verification_status"],
        "citations": len(citations),
        "attempts": row.get("attempts", 1),
        "retrieved_documents": retrieved,
    }
