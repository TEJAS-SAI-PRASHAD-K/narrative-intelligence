"""The Compass citation validator.

This is the module that decides whether a generated fact-check may be
persisted, so it gets the most adversarial tests in the suite. The failure it
exists to prevent -- an uncited AI fact-check inside a misinformation product --
is the worst thing this system could ship.

No database and no network: the validator is pure.
"""

from __future__ import annotations

import pytest

from app.compass.validate import (
    insufficient_evidence_row,
    split_sentences,
    validate,
)

CLAIM = "Mail-in ballots in three counties were counted twice."

GOOD_CONTEXT = (
    "According to Reuters, state election officials have not confirmed the "
    "duplicate tabulation described online. Reporting indicates the log excerpt "
    "circulating with the claim does not identify which county produced it."
)


def _spans(context: str) -> list[dict]:
    """One citation per sentence, covering it exactly."""
    return [
        {"url": f"https://example.test/{i}", "char_start": start, "char_end": end}
        for i, (start, end, _) in enumerate(split_sentences(context))
    ]


def test_a_fully_cited_hedged_note_passes():
    result = validate(GOOD_CONTEXT, _spans(GOOD_CONTEXT), claim=CLAIM)
    assert result.ok, result.failures
    assert result.covered_count == result.sentence_count == 2


def test_an_uncited_sentence_fails():
    """The rule the whole module exists for."""
    citations = _spans(GOOD_CONTEXT)[:1]  # drop the second sentence's citation
    result = validate(GOOD_CONTEXT, citations, claim=CLAIM)

    assert not result.ok
    assert "uncited_sentences" in result.failures
    assert len(result.uncovered_sentences) == 1
    assert result.covered_count == 1


def test_a_context_with_no_citations_at_all_fails():
    result = validate(GOOD_CONTEXT, [], claim=CLAIM)
    assert not result.ok
    assert "no_citations" in result.failures


def test_citations_without_spans_fail():
    """A citation that does not point at the text it supports is a bibliography."""
    result = validate(
        GOOD_CONTEXT,
        [{"url": "https://example.test/1", "char_start": None, "char_end": None}],
        claim=CLAIM,
    )
    assert not result.ok
    assert "citations_have_no_spans" in result.failures


@pytest.mark.parametrize(
    "context",
    [
        "In fact, the ballots were counted once. According to Reuters, this is settled.",
        "This proves that the claim is false. Reporting indicates officials agree.",
        "The truth is the log is fabricated. According to AP, no county matches it.",
        "Officials clearly show the claim is wrong. According to Reuters, it is.",
    ],
)
def test_unhedged_assertions_fail(context):
    """The note reports on evidence; it never rules on the claim itself."""
    result = validate(context, _spans(context), claim=CLAIM)
    assert not result.ok
    assert "unhedged_assertion" in result.failures
    assert result.unhedged_matches


def test_a_note_with_no_attribution_at_all_fails():
    """Flat statements with citations attached are still the model's own voice."""
    context = (
        "The tabulator log does not name a county. Three counties reported no "
        "duplicate scans during the certification window."
    )
    result = validate(context, _spans(context), claim=CLAIM)
    assert not result.ok
    assert "no_hedging_language" in result.failures


def test_restating_the_claim_as_fact_fails():
    """Laundering the claim into an authoritative voice is the subtle failure."""
    context = (
        "Mail-in ballots in three counties were counted twice during the "
        "tabulation window. According to Reuters, officials are reviewing."
    )
    result = validate(context, _spans(context), claim=CLAIM)
    assert not result.ok
    assert "restates_claim_as_fact" in result.failures


def test_attributing_the_claim_is_fine():
    """The same substance, attributed, is exactly what a note should say."""
    context = (
        "According to posts circulating online, mail-in ballots in three counties "
        "were counted twice. Reporting indicates officials have not confirmed this."
    )
    result = validate(context, _spans(context), claim=CLAIM)
    assert result.ok, result.failures


def test_an_empty_context_fails_rather_than_passing_vacuously():
    """No sentences means no uncited sentences, which must not read as success."""
    for empty in ("", "   ", "\n"):
        result = validate(empty, _spans(GOOD_CONTEXT), claim=CLAIM)
        assert not result.ok
        assert "empty_context" in result.failures


def test_a_fragment_is_too_short_to_be_a_note():
    context = "According to Reuters, no."
    result = validate(context, _spans(context), claim=CLAIM)
    assert not result.ok
    assert "context_too_short" in result.failures


def test_the_refusal_row_persists_nothing_but_the_refusal():
    """What lands when validation fails twice.

    The row exists so the attempt is auditable. The paragraph does not, because
    it could not be sourced. `context` is an empty string rather than NULL:
    NULL is indistinguishable from "not generated yet", and an analyst must be
    able to tell those apart.
    """
    row = insufficient_evidence_row(
        claim=CLAIM, report={"failures": ["uncited_sentences"]}, attempts=2
    )
    assert row["verification_status"] == "insufficient_evidence"
    assert row["context"] == ""
    assert row["context"] is not None
    assert row["attempts"] == 2
    assert "not evidence that the claim is false" in row["caution_note"]
    assert row["validation_report"]["failures"] == ["uncited_sentences"]


def test_the_validation_report_records_why_it_failed():
    """ "Why did this fail" must be answerable after the fact."""
    result = validate(GOOD_CONTEXT, _spans(GOOD_CONTEXT)[:1], claim=CLAIM)
    report = result.as_report()
    assert report["ok"] is False
    assert report["sentence_count"] == 2
    assert report["covered_count"] == 1
    assert report["uncovered_sentences"]


def test_sentence_offsets_point_into_the_original_text():
    """The spans are how the UI underlines a sourced claim, so they must land."""
    for start, end, sentence in split_sentences(GOOD_CONTEXT):
        assert GOOD_CONTEXT[start:end] == sentence
