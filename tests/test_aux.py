"""Auxiliary scorers: gating, null discipline, caching, anomaly features.

None of these tests download a model. What they check is the machinery that
surrounds the models -- which is where the mistakes that corrupt a dashboard
live. A wrong toxicity score is one bad number; a scorer that silently writes
0.0 where it meant "not assessed" is a systematically wrong column.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from modeling.aux.anomaly import MIN_POSTS_FOR_BASELINE, AnomalyScorer
from modeling.aux.base import AuxScorer, SkipReason, TextCache
from modeling.aux.emotion import EmotionScorer
from modeling.aux.sentiment import SentimentScorer
from modeling.aux.toxicity import ToxicityScorer
from modeling.io import EMOTIONS

BASE = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    from modeling import config as C

    C.get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    settings = C.get_settings()
    settings.ensure_dirs()
    yield settings
    C.get_settings.cache_clear()


class StubScorer(AuxScorer):
    """A scorer whose model always loads and returns the text length.

    Lets the gating, batching and caching paths be tested without weights.
    """

    module = "toxicity"
    output_columns = ("toxicity",)

    def __init__(self, settings=None, available=True):
        super().__init__(settings)
        self.available = available
        self.calls: list[list[str]] = []
        self._last: dict[str, float] = {}

    def load(self) -> bool:
        return self.available

    def score_texts(self, texts):
        self.calls.append(list(texts))
        values = [round(min(len(t) / 100, 1.0), 6) for t in texts]
        self._last.update(dict(zip(texts, values, strict=True)))
        return values

    def values_for(self, record_id: str) -> float:
        """The value this scorer produced, for comparing against a cache hit."""
        return next(iter(self._last.values()))


def make_records(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# gating: the English-only policy, applied in exactly one place
# ---------------------------------------------------------------------------
def test_non_english_is_skipped_not_scored(isolated_settings):
    """An English toxicity model run on German text returns a number, and that
    number is noise. Null with a reason code is the honest output."""
    records = make_records([
        {"id": "m:1", "text": "this is a perfectly ordinary english sentence", "lang": "en"},
        {"id": "m:2", "text": "der vertrag wurde ohne ausschreibung vergeben", "lang": "de"},
    ])
    outcome = StubScorer(isolated_settings).score(records)
    assert "m:1" in outcome.values
    assert "m:2" not in outcome.values
    assert outcome.skipped["m:2"] == SkipReason.UNSUPPORTED_LANGUAGE


def test_unknown_language_is_admitted_only_by_explicit_policy(isolated_settings, monkeypatch):
    """Phase 1 leaves `lang` None for short text rather than guessing. Whether
    we then score it is a stated policy, not an accident."""
    records = make_records([{"id": "m:1", "text": "short but scoreable text here", "lang": None}])

    assert "m:1" in StubScorer(isolated_settings).score(records).values

    from modeling import config as C

    C.get_settings.cache_clear()
    monkeypatch.setenv("SCORE_UNKNOWN_LANGUAGE", "false")
    strict = C.get_settings()
    outcome = StubScorer(strict).score(records)
    assert outcome.skipped["m:1"] == SkipReason.UNSUPPORTED_LANGUAGE
    C.get_settings.cache_clear()


def test_null_lang_from_parquet_does_not_crash_the_gate(isolated_settings):
    """Phase 1 leaves `lang` unset for short text rather than guessing, and a
    null string column read back from Parquet arrives as NaN, not None. Treating
    NaN as "some language" crashes; treating it as a language name would be
    worse."""
    records = make_records([
        {"id": "m:1", "text": "an english sentence with enough characters", "lang": np.nan},
        {"id": "m:2", "text": "another english sentence with enough length", "lang": None},
    ])
    outcome = StubScorer(isolated_settings).score(records)
    assert set(outcome.values) == {"m:1", "m:2"}


def test_null_text_from_parquet_is_empty_not_the_string_nan(isolated_settings):
    """`str(NaN)` is the three-character string "nan", which sails through a
    length check and gets scored as if it were content."""
    records = make_records([{"id": "m:1", "text": np.nan, "lang": "en"}])
    outcome = StubScorer(isolated_settings).score(records)
    assert outcome.skipped["m:1"] == SkipReason.EMPTY_TEXT
    assert not outcome.values


def test_empty_and_too_short_text_are_distinct_reasons(isolated_settings):
    records = make_records([
        {"id": "m:1", "text": "", "lang": "en"},
        {"id": "m:2", "text": "ok", "lang": "en"},
        {"id": "m:3", "text": None, "lang": "en"},
    ])
    outcome = StubScorer(isolated_settings).score(records)
    assert outcome.skipped["m:1"] == SkipReason.EMPTY_TEXT
    assert outcome.skipped["m:2"] == SkipReason.TEXT_TOO_SHORT
    assert outcome.skipped["m:3"] == SkipReason.EMPTY_TEXT
    assert not outcome.values


def test_unavailable_model_skips_rather_than_fabricating(isolated_settings):
    """No weights on disk must mean nulls, not zeros. A 0.0 is indistinguishable
    from a confident negative once it reaches the dashboard."""
    records = make_records([{"id": "m:1", "text": "a sentence long enough to score", "lang": "en"}])
    outcome = StubScorer(isolated_settings, available=False).score(records)
    assert not outcome.values
    assert outcome.skipped["m:1"] == SkipReason.MODEL_UNAVAILABLE


def test_every_record_is_either_scored_or_skipped(isolated_settings):
    """The two dicts must be exhaustive: a record in neither is a silent loss."""
    records = make_records([
        {"id": "m:1", "text": "a sentence long enough to score properly", "lang": "en"},
        {"id": "m:2", "text": "", "lang": "en"},
        {"id": "m:3", "text": "another perfectly fine english sentence", "lang": "fr"},
    ])
    outcome = StubScorer(isolated_settings).score(records)
    covered = set(outcome.values) | set(outcome.skipped)
    assert covered == {"m:1", "m:2", "m:3"}


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
def test_second_run_hits_the_cache_and_calls_no_model(isolated_settings):
    records = make_records([{"id": "m:1", "text": "a sentence long enough to score", "lang": "en"}])
    first = StubScorer(isolated_settings)
    first.score(records)
    assert first.calls  # the model ran

    expected = first.values_for("m:1")

    second = StubScorer(isolated_settings)
    outcome = second.score(records)
    assert outcome.values["m:1"] == pytest.approx(expected)
    assert second.calls == [], "cached text must not reach the model again"


def test_cache_key_includes_the_model_version(isolated_settings):
    """A version bump must invalidate the cache, or a retrain silently reuses
    the old model's scores."""
    cache_a = TextCache("toxicity", "unitary/toxic-bert", "v0.1.0", isolated_settings)
    cache_b = TextCache("toxicity", "unitary/toxic-bert", "v0.2.0", isolated_settings)
    assert cache_a.key("same text") != cache_b.key("same text")

    cache_c = TextCache("toxicity", "other/model", "v0.1.0", isolated_settings)
    assert cache_a.key("same text") != cache_c.key("same text")


def test_cache_survives_a_flush_and_reload(isolated_settings):
    cache = TextCache("toxicity", "m", "v1", isolated_settings)
    cache.put("hello", 0.25)
    cache.flush()
    reloaded = TextCache("toxicity", "m", "v1", isolated_settings)
    assert reloaded.get("hello") == 0.25
    assert reloaded.get("never seen") is None


# ---------------------------------------------------------------------------
# emotion mapping: no bucket may be synthesized
# ---------------------------------------------------------------------------
def test_emotion_buckets_match_the_contract(isolated_settings):
    scorer = EmotionScorer(isolated_settings)
    assert set(scorer.label_map.values()) <= set(EMOTIONS)
    # The chosen model covers all seven, so nothing is left flat-zero.
    assert scorer.uncovered == []


def test_sentiment_score_is_signed_and_symmetric():
    """An ambivalent post must read as ~0, not as "positive, confidence 0.45"."""
    from modeling.aux.sentiment import _first

    scores = {"positive": 0.45, "neutral": 0.10, "negative": 0.45}
    assert _first(scores, ("positive", "label_2")) == 0.45
    signed = scores["positive"] - scores["negative"]
    assert signed == pytest.approx(0.0)


def test_scorers_declare_the_columns_they_fill():
    for cls in (ToxicityScorer, SentimentScorer, EmotionScorer):
        assert cls.output_columns
        assert cls.module


# ---------------------------------------------------------------------------
# anomaly: features, null discipline, and what the score actually means
# ---------------------------------------------------------------------------
def corpus_for_anomaly(n_authors=8, posts=6, with_engagement=True):
    rows = []
    for a in range(n_authors):
        for p in range(posts):
            rows.append({
                "id": f"m:{a}-{p}",
                "author_id": f"mastodon:user{a}",
                "source": "mastodon",
                "text": f"post {p} from author {a} about the local contract dispute",
                "lang": "en",
                "timestamp": BASE + timedelta(days=a, hours=p * 3),
                "engagement": (
                    {"likes": p * 2, "shares": None, "replies": None, "views": None}
                    if with_engagement
                    else {"likes": None, "shares": None, "replies": None, "views": None}
                ),
                "urls": [],
                "hashtags": ["contract"],
                "mentions": [],
                "simhash": (a << 50) | p,
            })
    return pd.DataFrame(rows)


def test_anomaly_scores_are_ranks_in_the_unit_interval(isolated_settings):
    outcome = AnomalyScorer(isolated_settings).score(corpus_for_anomaly())
    assert outcome.values
    values = list(outcome.values.values())
    assert all(0.0 <= v <= 1.0 for v in values)


def test_anomaly_skips_authors_with_no_baseline(isolated_settings):
    """"Unusual for this author" is undefined against a sample of one."""
    frame = corpus_for_anomaly(n_authors=3, posts=6)
    thin = pd.DataFrame([{
        **frame.iloc[0].to_dict(),
        "id": "m:lonely",
        "author_id": "mastodon:lonely",
    }])
    outcome = AnomalyScorer(isolated_settings).score(pd.concat([frame, thin], ignore_index=True))
    assert outcome.skipped["m:lonely"] == SkipReason.NOT_ENOUGH_HISTORY
    assert "m:lonely" not in outcome.values


def test_anomaly_is_deterministic_under_the_seed(isolated_settings):
    frame = corpus_for_anomaly()
    a = AnomalyScorer(isolated_settings).score(frame).values
    b = AnomalyScorer(isolated_settings).score(frame).values
    assert a == b


def test_engagement_nulls_get_an_indicator_not_a_zero(isolated_settings):
    """Phase 1's rule carries through: null means "not measurable on this
    platform". Filling it with 0 would make it look like a measured absence."""
    scorer = AnomalyScorer(isolated_settings)
    features, _ = scorer.build_features(corpus_for_anomaly(with_engagement=False))
    frame = features.as_frame()
    assert "engagement_is_missing" in frame.columns
    assert (frame["engagement_is_missing"] == 1.0).all()


def test_engagement_zero_and_null_are_distinguished(isolated_settings):
    """likes=0 is measured; all-null is not. The indicator must tell them apart."""
    rows = corpus_for_anomaly(n_authors=4, posts=5).to_dict(orient="records")
    rows[0]["engagement"] = {"likes": 0, "shares": None, "replies": None, "views": None}
    rows[1]["engagement"] = {"likes": None, "shares": None, "replies": None, "views": None}
    scorer = AnomalyScorer(isolated_settings)
    features, _ = scorer.build_features(pd.DataFrame(rows))
    frame = features.as_frame()
    assert frame.loc[rows[0]["id"], "engagement_is_missing"] == 0.0
    assert frame.loc[rows[1]["id"], "engagement_is_missing"] == 1.0


def test_anomaly_features_contain_no_nan_or_inf(isolated_settings):
    """IsolationForest silently misbehaves on inf; a ratio with a zero
    denominator is the obvious way to produce one."""
    frame = corpus_for_anomaly()
    features, _ = AnomalyScorer(isolated_settings).build_features(frame)
    assert np.isfinite(features.matrix).all()


def test_every_anomaly_feature_is_named(isolated_settings):
    """A feature nobody can name is a feature nobody can defend when the
    dashboard flags an account because of it."""
    features, _ = AnomalyScorer(isolated_settings).build_features(corpus_for_anomaly())
    assert len(features.names) == features.matrix.shape[1]
    assert all(name and not name.startswith("Unnamed") for name in features.names)


def test_min_posts_threshold_is_stated_not_magic():
    assert MIN_POSTS_FOR_BASELINE >= 2


# ---------------------------------------------------------------------------
# the assembled record_scores frame
# ---------------------------------------------------------------------------
def test_aux_pass_emits_one_row_per_record_even_when_nothing_scored(isolated_settings):
    """"Not in the corpus" and "in the corpus, not assessed" are different
    facts, and the scored table must be able to express both."""
    from modeling.aux import run_aux_pass

    records = corpus_for_anomaly(n_authors=3, posts=4)
    frame = run_aux_pass(records, None, settings=isolated_settings, scorers=["anomaly"])
    assert len(frame) == len(records)
    assert set(frame["record_id"]) == set(records["id"])
    assert "skip_reasons" in frame.columns


def test_model_versions_claim_only_scorers_that_produced_a_value(isolated_settings):
    """A row skipped by toxicity must not claim a toxicity version, or a
    retrain will think it is already current."""
    from modeling.aux import run_aux_pass

    records = corpus_for_anomaly(n_authors=3, posts=4)
    frame = run_aux_pass(records, None, settings=isolated_settings, scorers=["anomaly"])
    for row in frame.to_dict(orient="records"):
        versions = row["model_versions"]
        if row["anomaly_score"] is None:
            assert "anomaly" not in versions
        else:
            assert versions["anomaly"]
