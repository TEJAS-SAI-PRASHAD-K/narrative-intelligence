"""Laya scoring: routing, polarity, long states, and graceful absence.

`laya` is an optional dependency and is not installed in CI, so everything here
runs against a fake agent built to the documented `predict` contract. That is
the point rather than a limitation: the behaviour worth pinning is this
project's -- which checkpoint a language goes to, which end of the probability
means "misinformation", what happens to a 7,000-character article -- and none
of it should change when the library updates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from modeling.aux.base import SkipReason
from modeling.text.laya_clf import LayaScorer, LayaSkipReason, polarity_note


class FakeAgent:
    """Implements the documented `predict` / `predict_long` contract.

    Returns ``{"answers": {qid: {"noul": p}}}``, which is the shape
    `laya.agent.system_one` documents: a ``noul`` question answers with
    ``P(true)`` in 0.0-1.0.
    """

    def __init__(self, noul: float = 0.9, name: str = "fake"):
        self.noul = noul
        self.name = name
        self.predict_calls: list[dict] = []
        self.long_calls: list[dict] = []

    def predict(self, state, questions, **kwargs):
        self.predict_calls.append({"state": state, "questions": questions, **kwargs})
        qid = next(iter(questions))
        return {"answers": {qid: {"noul": self.noul, "confidence": 0.8}}, "usage": {}}

    def predict_long(self, state, questions, **kwargs):
        self.long_calls.append({"state": state, "questions": questions, **kwargs})
        qid = next(iter(questions))
        return {"answers": {qid: {"noul": self.noul}}, "usage": {}}


@pytest.fixture
def scorer():
    return LayaScorer()


def _frame(rows):
    return pd.DataFrame(rows)


def _force_available(scorer, monkeypatch):
    """Pretend laya imported, so routing and shape logic can be tested."""
    stub = type("Availability", (), {"ok": True, "reason": ""})()
    monkeypatch.setattr(scorer, "available", lambda: stub)


class TestQuestionSpec:
    def test_it_is_a_noul_so_the_answer_is_one_probability(self, scorer):
        questions = scorer.question()
        assert list(questions) == ["misinfo"]
        assert questions["misinfo"]["type"] == "noul"

    def test_criteria_true_describes_the_FALSE_claim(self, scorer):
        """The polarity contract. Getting this backwards inverts the column.

        `noul` returns P(true) whatever `labels` renders, and misinfo_prob is
        P(misinformation-like). So criteria["true"] must describe the false
        claim, or ROC-AUC lands near 0 and reads as a bad model rather than a
        flipped sign -- the failure tests/test_label_orientation.py exists for.
        """
        criteria = scorer.question()["misinfo"]["criteria"]
        true_side = criteria["true"].lower()
        false_side = criteria["false"].lower()
        assert "false" in true_side or "fabricated" in true_side
        assert "accurate" in false_side
        # And the two must not both describe the same thing.
        assert true_side != false_side

    def test_the_polarity_note_states_the_contract(self):
        note = polarity_note().lower()
        assert "p(claim is false)" in note
        assert "misinfo_prob" in note

    def test_folded_yaml_whitespace_is_collapsed(self, scorer):
        """The model sees this text; YAML's newlines are not part of the prompt."""
        spec = scorer.question()["misinfo"]
        assert "\n" not in spec["instructions"]
        assert not spec["instructions"].endswith(" ")
        for value in spec["criteria"].values():
            assert "\n" not in value


class TestRouting:
    def test_hindi_goes_to_the_multilingual_checkpoint(self, scorer):
        """The entire reason this module exists."""
        checkpoint, subfolder = scorer.checkpoint_for("hi")
        assert subfolder == "multilingual"
        assert checkpoint

    def test_english_goes_to_the_english_checkpoint(self, scorer):
        checkpoint, subfolder = scorer.checkpoint_for("en")
        assert subfolder is None
        assert checkpoint

    def test_an_unrouted_language_is_refused_not_guessed(self, scorer):
        """Tamil is in the corpus incidentally. Scoring it with either
        checkpoint would be a guess, so routing returns nothing."""
        assert scorer.checkpoint_for("ta") == (None, None)
        assert scorer.checkpoint_for(None) == (None, None)
        assert scorer.checkpoint_for("") == (None, None)

    def test_case_is_not_significant(self, scorer):
        assert scorer.checkpoint_for("HI") == scorer.checkpoint_for("hi")

    def test_context_window_follows_the_checkpoint(self, scorer):
        """mmBERT-base carries 1024; ModernBERT-large 512. Using the wrong one
        truncates or wastes budget."""
        assert scorer.max_len_for("hi") > scorer.max_len_for("en")


class TestScoring:
    def _patched(self, scorer, monkeypatch, agent):
        _force_available(scorer, monkeypatch)
        monkeypatch.setattr(scorer, "agent_for", lambda lang: agent)
        return scorer

    def test_a_short_state_uses_predict_not_predict_long(self, scorer, monkeypatch):
        agent = FakeAgent(noul=0.87)
        self._patched(scorer, monkeypatch, agent)
        frame = _frame([{"id": "news:1", "text": "a claim " * 5, "lang": "en"}])
        outcome = scorer.score(frame)
        assert outcome.values == {"news:1": 0.87}
        assert len(agent.predict_calls) == 1
        assert not agent.long_calls

    def test_a_long_article_scans_windows_instead_of_being_truncated(self, scorer, monkeypatch):
        """`predict` truncates past max_len and SILENTLY DROPS THE REST.

        Fact-check articles in this corpus run 2,000-7,000 characters, so the
        default path would score a headline and call it an article.
        """
        agent = FakeAgent(noul=0.4)
        self._patched(scorer, monkeypatch, agent)
        long_text = "x" * 5000
        frame = _frame([{"id": "news:2", "text": long_text, "lang": "en"}])
        outcome = scorer.score(frame)
        assert outcome.values == {"news:2": 0.4}
        assert len(agent.long_calls) == 1
        assert not agent.predict_calls

    def test_hindi_and_english_get_different_agents(self, scorer, monkeypatch):
        seen: list[str | None] = []
        agents = {"en": FakeAgent(0.1, "en"), "hi": FakeAgent(0.9, "hi")}

        def agent_for(lang):
            seen.append(lang)
            return agents[lang]

        _force_available(scorer, monkeypatch)
        monkeypatch.setattr(scorer, "agent_for", agent_for)
        frame = _frame(
            [
                {"id": "news:en", "text": "an english claim here", "lang": "en"},
                {"id": "news:hi", "text": "यह एक गलत दावा है जो वायरल हो रहा है", "lang": "hi"},
            ]
        )
        outcome = scorer.score(frame)
        assert seen == ["en", "hi"]
        assert outcome.values == {"news:en": 0.1, "news:hi": 0.9}

    def test_values_and_skipped_are_exhaustive_over_the_input(self, scorer, monkeypatch):
        """The aux-scorer contract: no record vanishes without a reason."""
        self._patched(scorer, monkeypatch, FakeAgent())
        frame = _frame(
            [
                {"id": "a", "text": "a perfectly fine long claim here", "lang": "en"},
                {"id": "b", "text": "", "lang": "en"},
                {"id": "c", "text": "tiny", "lang": "en"},
                {"id": "d", "text": "a claim in an unrouted language", "lang": "ta"},
            ]
        )
        outcome = scorer.score(frame)
        assert set(outcome.values) | set(outcome.skipped) == {"a", "b", "c", "d"}
        assert not set(outcome.values) & set(outcome.skipped)
        assert outcome.skipped["b"] == SkipReason.EMPTY_TEXT
        assert outcome.skipped["c"] == SkipReason.TEXT_TOO_SHORT
        assert outcome.skipped["d"] == LayaSkipReason.UNROUTED_LANGUAGE

    def test_an_inference_error_is_a_reason_code_not_a_crash(self, scorer, monkeypatch):
        class Exploding(FakeAgent):
            def predict(self, *a, **k):
                raise RuntimeError("cuda is sad")

        self._patched(scorer, monkeypatch, Exploding())
        frame = _frame([{"id": "x", "text": "a claim long enough to score", "lang": "en"}])
        outcome = scorer.score(frame)
        assert outcome.values == {}
        assert outcome.skipped["x"] == SkipReason.INFERENCE_ERROR

    def test_a_missing_noul_is_an_error_rather_than_a_zero(self, scorer, monkeypatch):
        """If the question stops being a noul, that must be loud.

        Returning 0.0 would read as "confidently not misinformation" for every
        record in the corpus.
        """
        class WrongShape(FakeAgent):
            def predict(self, state, questions, **kwargs):
                qid = next(iter(questions))
                return {"answers": {qid: {"choice": "false"}}}

        agent = WrongShape()
        with pytest.raises(ValueError, match="no `noul`"):
            scorer.score_one(agent, "a claim long enough", "en")


class TestUnknownLanguage:
    def test_null_lang_is_skipped_when_the_setting_says_so(self, scorer, monkeypatch):
        monkeypatch.setattr(scorer.settings, "score_unknown_language", False, raising=False)
        _force_available(scorer, monkeypatch)
        frame = _frame([{"id": "u", "text": "no language tag on this claim", "lang": None}])
        outcome = scorer.score(frame)
        assert outcome.skipped["u"] == LayaSkipReason.UNKNOWN_LANGUAGE

    def test_null_lang_is_assumed_english_when_the_setting_allows(self, scorer, monkeypatch):
        monkeypatch.setattr(scorer.settings, "score_unknown_language", True, raising=False)
        _force_available(scorer, monkeypatch)
        monkeypatch.setattr(scorer, "agent_for", lambda lang: FakeAgent(0.55))
        frame = _frame([{"id": "u", "text": "no language tag on this claim", "lang": None}])
        outcome = scorer.score(frame)
        assert outcome.values == {"u": 0.55}


class TestGracefulAbsence:
    def test_without_laya_every_record_is_skipped_and_none_is_guessed(self, scorer):
        """laya is optional and absent in CI; this is the real code path."""
        if scorer.available().ok:
            pytest.skip("laya is installed in this environment")
        frame = _frame(
            [
                {"id": "a", "text": "a claim long enough to score", "lang": "en"},
                {"id": "b", "text": "दूसरा दावा जो काफी लंबा है", "lang": "hi"},
            ]
        )
        outcome = scorer.score(frame)
        assert outcome.values == {}
        assert set(outcome.skipped) == {"a", "b"}
        assert all(r == SkipReason.MODEL_UNAVAILABLE for r in outcome.skipped.values())

    def test_the_unavailable_message_says_how_to_fix_it(self, scorer):
        if scorer.available().ok:
            pytest.skip("laya is installed in this environment")
        reason = scorer.available().reason
        assert "pip install" in reason
        assert "[laya]" in reason


class TestTrainingExport:
    """The JSONL the fine-tune consumes.

    The whole point of this class is that laya's CSV shortcut builds a CHOICE
    question from the distinct labels, while this project asks a NOUL at
    inference. Training one head and querying another degrades silently.
    """

    def test_the_exported_question_is_the_one_the_scorer_asks(self, scorer):
        """One definition, read by both sides. Drift is otherwise invisible."""
        from modeling.text.laya_clf import LayaTrainingExporter

        exporter = LayaTrainingExporter(scorer)
        row = exporter.row("a claim", True)
        assert row["questions"] == scorer.question()
        assert row["questions"][scorer.question_id]["type"] == "noul"

    def test_expected_is_a_boolean_and_true_means_misinformation(self, scorer):
        """`expected` for a noul is a bool; True = yes to "is this false?".

        Matches liar.py's `label = (binary == "false")`, so a 1 in the
        benchmark and a True here mean the same class.
        """
        from modeling.text.laya_clf import LayaTrainingExporter

        exporter = LayaTrainingExporter(scorer)
        assert exporter.row("x", True)["expected"] == {"misinfo": True}
        assert exporter.row("x", False)["expected"] == {"misinfo": False}
        assert isinstance(exporter.row("x", 1)["expected"]["misinfo"], bool)

    def test_export_writes_one_json_object_per_line(self, scorer, tmp_path):
        import json

        from modeling.text.laya_clf import LayaTrainingExporter

        frame = _frame(
            [
                {"id": "a", "text": "a false claim", "label": 1},
                {"id": "b", "text": "a true claim", "label": 0},
            ]
        )
        path = tmp_path / "train.jsonl"
        assert LayaTrainingExporter(scorer).export(frame, path) == 2
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["expected"]["misinfo"] for r in rows] == [True, False]
        assert all(r["state"] for r in rows)

    def test_unlabelled_and_empty_rows_are_not_exported(self, scorer, tmp_path):
        from modeling.text.laya_clf import LayaTrainingExporter

        frame = _frame(
            [
                {"id": "a", "text": "a real claim", "label": 1},
                {"id": "b", "text": "", "label": 1},
                {"id": "c", "text": "unlabelled", "label": None},
            ]
        )
        path = tmp_path / "train.jsonl"
        assert LayaTrainingExporter(scorer).export(frame, path) == 1

    def test_devanagari_survives_the_json_round_trip(self, scorer, tmp_path):
        """ensure_ascii=False, so the file stays readable and the tokenizer
        sees the same bytes the corpus holds."""
        import json

        from modeling.text.laya_clf import LayaTrainingExporter

        claim = "यह दावा पूरी तरह से गलत है"
        frame = _frame([{"id": "a", "text": claim, "label": 1}])
        path = tmp_path / "train.jsonl"
        LayaTrainingExporter(scorer).export(frame, path)
        assert claim in path.read_text(encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8").strip())["state"] == claim
