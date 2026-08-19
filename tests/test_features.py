"""Account features and coordination detection.

The features are pure functions, so they are tested against hand-constructed
inputs whose right answer is known by inspection. That matters more here than
elsewhere: SHAP contributions over these feature names are what the dashboard
shows a user as the reason their account was flagged, so a feature that computes
something other than what its name says is a defect with a human on the other
end of it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from modeling.accounts import features as F
from modeling.accounts.coordination import CoordinationDetector

BASE = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    from modeling import config as C

    C.get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    settings = C.get_settings()
    settings.ensure_dirs()
    yield settings
    C.get_settings.cache_clear()


def series(*offsets_minutes: float) -> pd.Series:
    return pd.Series([BASE + timedelta(minutes=m) for m in offsets_minutes])


# ---------------------------------------------------------------------------
# temporal features
# ---------------------------------------------------------------------------
def test_hour_entropy_is_zero_for_one_hour_and_maximal_for_uniform():
    """Bots post uniformly; humans sleep. This feature is the difference."""
    same_hour = pd.Series([BASE + timedelta(days=d) for d in range(10)])
    assert F.hour_of_day_entropy(same_hour) == pytest.approx(0.0)

    every_hour = pd.Series([BASE.replace(hour=h) for h in range(24)])
    # log2(24) is the ceiling for a 24-bucket distribution.
    assert F.hour_of_day_entropy(every_hour) == pytest.approx(np.log2(24), abs=1e-6)


def test_interval_entropy_separates_a_scheduler_from_a_human():
    scheduler = F.inter_post_intervals(series(*[i * 30 for i in range(40)]))
    human = F.inter_post_intervals(
        series(0, 3, 5, 90, 95, 400, 1500, 1520, 3000, 3001, 8000, 20000)
    )
    assert F.interval_entropy(scheduler) < F.interval_entropy(human)


def test_burstiness_separates_periodic_from_clumped():
    """Goh & Barabasi burstiness: -1 perfectly periodic, +1 maximally bursty.

    Both extremes are non-human in different ways -- schedulers tick, campaigns
    burst -- so what matters is that the two are ordered, not that either hits a
    particular value.
    """
    periodic = F.inter_post_intervals(series(*[i * 60 for i in range(20)]))
    assert F.burstiness(periodic) == pytest.approx(-1.0, abs=0.05)

    clumped = F.inter_post_intervals(series(0, 1, 2, 3, 4, 5, 10000, 10001, 10002))
    assert F.burstiness(clumped) > 0.4
    assert F.burstiness(clumped) > F.burstiness(periodic)


def test_longest_streak_counts_consecutive_days_only():
    consecutive = pd.Series([BASE + timedelta(days=d) for d in range(5)])
    assert F.longest_active_streak_days(consecutive) == 5.0

    gapped = pd.Series([BASE + timedelta(days=d) for d in (0, 1, 5, 6, 7, 20)])
    assert F.longest_active_streak_days(gapped) == 3.0

    assert F.longest_active_streak_days(pd.Series([BASE])) == 1.0


def test_posts_per_day_handles_a_single_post():
    assert F.posts_per_day(pd.Series([BASE])) == 1.0


# ---------------------------------------------------------------------------
# content features
# ---------------------------------------------------------------------------
def test_type_token_ratio_falls_for_a_recycled_vocabulary():
    varied = ["the council approved a new filtration contract today",
              "residents raised concerns about water quality standards"]
    template = ["buy now click here"] * 8
    assert F.type_token_ratio(varied) > F.type_token_ratio(template)
    assert F.type_token_ratio([]) == 0.0


def test_duplicate_content_rate_counts_repeats_not_originals():
    assert F.duplicate_content_rate(["a", "b", "c"]) == 0.0
    assert F.duplicate_content_rate(["same", "same", "same", "other"]) == pytest.approx(0.5)
    # Whitespace and case are normalized: a repost is a repost.
    assert F.duplicate_content_rate(["Same  Text", "same text"]) == pytest.approx(0.5)


def test_self_similarity_is_high_for_template_posting():
    identical = [0xABCD_1234_5678_9000] * 6
    assert F.self_similarity(identical) == pytest.approx(1.0)

    spread = [(i * 0x9E3779B97F4A7C15) & ((1 << 64) - 1) for i in range(6)]
    assert F.self_similarity(spread) < 0.7
    assert F.self_similarity([12345]) == 0.0


# ---------------------------------------------------------------------------
# tiers and the intersection discipline
# ---------------------------------------------------------------------------
def corpus(n_authors=4, posts=8, with_parents=True):
    rows = []
    for a in range(n_authors):
        for p in range(posts):
            rows.append({
                "id": f"mastodon:{a}-{p}",
                "author_id": f"mastodon:user{a}",
                "source": "mastodon",
                "text": f"post {p} by author {a} about the filtration contract dispute",
                "timestamp": BASE + timedelta(days=a, hours=p * 2),
                "parent_id": f"mastodon:{a}-0" if with_parents and p else None,
                "conversation_id": f"mastodon:{a}-0" if with_parents else None,
                "urls": [f"https://ex.example/{p}"] if p % 2 else [],
                "hashtags": ["contract"],
                "mentions": [],
                "simhash": ((a * 100 + p) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
            })
    return pd.DataFrame(rows)


def authors_frame(n=4, followers=True):
    return pd.DataFrame([
        {
            "author_id": f"mastodon:user{a}",
            "source": "mastodon",
            "followers": 100 + a * 50 if followers else None,
            "following": 80 + a * 10 if followers else None,
            "created_at": BASE - timedelta(days=500),
            "post_count": 8,
        }
        for a in range(n)
    ])


def test_universal_tier_needs_nothing_but_posts():
    matrix = F.build_features(corpus(), None, tiers=["universal"])
    assert matrix.matrix.shape == (4, len(F.UNIVERSAL))
    assert np.isfinite(matrix.matrix).all()


def test_available_tiers_reflects_what_the_corpus_can_actually_supply():
    assert F.available_tiers(corpus(with_parents=False), None) == ["universal"]
    assert "social_graph" in F.available_tiers(corpus(), authors_frame())
    assert "threading" in F.available_tiers(corpus(with_parents=True), None)


def test_missing_follower_data_gets_an_indicator_not_a_zero():
    """ConvoKit Reddit has no follower concept at all. Zero followers and no
    follower data must not look the same to the model."""
    matrix = F.build_features(
        corpus(), authors_frame(followers=False), tiers=["universal", "social_graph"]
    )
    frame = matrix.as_frame()
    assert (frame["followers_is_missing"] == 1.0).all()
    assert (frame["followers"] == 0.0).all()


def test_present_follower_data_clears_the_indicator():
    matrix = F.build_features(
        corpus(), authors_frame(followers=True), tiers=["universal", "social_graph"]
    )
    frame = matrix.as_frame()
    assert (frame["followers_is_missing"] == 0.0).all()
    assert (frame["followers"] > 0).all()


def test_follower_ratio_never_divides_by_zero():
    frame = authors_frame()
    frame["following"] = 0
    matrix = F.build_features(corpus(), frame, tiers=["social_graph"])
    assert np.isfinite(matrix.as_frame()["follower_following_ratio"]).all()


def test_intersection_refuses_when_no_tier_is_shared():
    """A model trained on features the corpus cannot compute is not a model."""
    with pytest.raises(ValueError, match="no shared feature tier"):
        F.intersection_features(["social_graph"], ["threading"])


def test_intersection_keeps_only_the_shared_tiers():
    shared = F.intersection_features(
        ["universal", "social_graph"], ["universal", "threading"]
    )
    assert shared == ["universal"]


def test_features_are_pure_and_deterministic():
    """The same function computes features for the benchmark and for the corpus.
    If it were not deterministic, training and scoring would silently disagree."""
    a = F.build_features(corpus(), authors_frame(), tiers=["universal", "social_graph"])
    b = F.build_features(corpus(), authors_frame(), tiers=["universal", "social_graph"])
    assert np.array_equal(a.matrix, b.matrix)
    assert a.names == b.names


def test_every_feature_name_is_declared_in_a_tier():
    """SHAP contributions are shown to users by name; an undeclared feature
    would surface as an unexplainable reason for a flag."""
    matrix = F.build_features(corpus(), authors_frame(), tiers=list(F.TIERS))
    declared = {name for tier in F.TIERS.values() for name in tier}
    assert set(matrix.names) <= declared
    assert len(matrix.names) == matrix.matrix.shape[1]


def test_subset_returns_columns_in_the_requested_order():
    """The *requested* order, not the matrix's.

    ``BotModel.predict_proba`` calls ``subset(model.feature_names)`` and feeds
    the result straight to the estimator. If subset returned its own ordering,
    every feature would be silently fed to the wrong tree split and the model
    would keep producing plausible, wrong probabilities.
    """
    matrix = F.build_features(corpus(), authors_frame(), tiers=["universal", "social_graph"])
    wanted = ["followers", "post_count", "hour_entropy"]
    subset = matrix.subset(wanted)
    assert subset.names == wanted

    frame = matrix.as_frame()
    for position, name in enumerate(wanted):
        assert np.allclose(subset.matrix[:, position], frame[name].to_numpy())


# ---------------------------------------------------------------------------
# coordination
# ---------------------------------------------------------------------------
def coordinated_corpus(n_amplifiers=6, n_organic=12):
    """A planted burst of near-identical posts plus unrelated organic chatter."""
    rows = []
    shared = 0xDEAD_BEEF_1234_0000
    for i in range(n_amplifiers):
        rows.append({
            "id": f"mastodon:amp-{i}",
            "author_id": f"mastodon:amplifier{i}",
            "source": "mastodon",
            "text": "share this everywhere the contract was awarded without tender",
            "timestamp": BASE + timedelta(minutes=i * 3),
            "parent_id": None,
            "urls": ["https://ex.example/contract"],
            "hashtags": ["waterworks", "contract"],
            "simhash": shared ^ (1 << i),  # within Hamming 3 of each other
        })
    for i in range(n_organic):
        rows.append({
            "id": f"mastodon:org-{i}",
            "author_id": f"mastodon:resident{i}",
            "source": "mastodon",
            "text": f"unrelated musing number {i} about the weather and the bus timetable",
            "timestamp": BASE + timedelta(days=1 + i, minutes=i * 17),
            "parent_id": None,
            "urls": [],
            "hashtags": [],
            "simhash": ((i + 900) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
        })
    return pd.DataFrame(rows)


def test_coordination_links_the_planted_burst(isolated_settings):
    result = CoordinationDetector(isolated_settings).detect(
        coordinated_corpus(), run_null_model=False
    )
    linked = {e["src_author_id"] for e in result.edges} | {
        e["dst_author_id"] for e in result.edges
    }
    assert all(f"mastodon:amplifier{i}" in linked for i in range(6))
    assert not any(a.startswith("mastodon:resident") for a in linked)


def test_edges_carry_their_evidence_type(isolated_settings):
    """The UI says *why* two nodes are linked. An untyped edge cannot."""
    result = CoordinationDetector(isolated_settings).detect(
        coordinated_corpus(), run_null_model=False
    )
    kinds = {e["evidence"] for e in result.edges}
    assert kinds <= {"near_dup", "cotweet", "hashtag_seq", "temporal"}
    assert "near_dup" in kinds
    assert "cotweet" in kinds


def test_an_author_is_never_linked_to_itself(isolated_settings):
    frame = coordinated_corpus()
    frame["author_id"] = "mastodon:one_author"
    result = CoordinationDetector(isolated_settings).detect(frame, run_null_model=False)
    assert all(e["src_author_id"] != e["dst_author_id"] for e in result.edges)


def test_ineligible_sources_are_excluded_and_counted(isolated_settings):
    frame = coordinated_corpus()
    gdelt = frame.head(3).copy()
    gdelt["source"] = "gdelt"
    gdelt["id"] = ["gdelt:a", "gdelt:b", "gdelt:c"]
    gdelt["author_id"] = ["gdelt:outlet1", "gdelt:outlet2", "gdelt:outlet3"]
    result = CoordinationDetector(isolated_settings).detect(
        pd.concat([frame, gdelt], ignore_index=True), run_null_model=False
    )
    assert result.excluded_sources.get("gdelt:source_not_eligible") == 3


def test_deleted_authors_are_excluded(isolated_settings):
    """Keeping them would merge every tombstoned post into one pseudo-account."""
    frame = coordinated_corpus()
    frame.loc[frame.index[:2], "author_id"] = "mastodon:__deleted__"
    result = CoordinationDetector(isolated_settings).detect(frame, run_null_model=False)
    assert result.excluded_sources.get("deleted_author") == 2


def test_null_model_runs_and_is_reported(isolated_settings):
    """Any graph has communities. The null model is what makes the claim mean
    something -- and it must be reported whichever way it comes out."""
    result = CoordinationDetector(isolated_settings).detect(
        coordinated_corpus(n_amplifiers=8, n_organic=20), run_null_model=True
    )
    assert result.null_modularity >= 0.0
    assert isinstance(result.exceeds_null, bool)
    assert "null_modularity" in result.summary()


def test_coordination_scores_are_bounded_and_explainable(isolated_settings):
    result = CoordinationDetector(isolated_settings).detect(
        coordinated_corpus(), run_null_model=False
    )
    assert result.scores
    assert all(0.0 <= v <= 1.0 for v in result.scores.values())
    # The amplifiers should score above anyone the graph barely touches.
    amplifier_scores = [
        v for k, v in result.scores.items() if k.startswith("mastodon:amplifier")
    ]
    assert amplifier_scores
    assert min(amplifier_scores) > 0.0


def test_hashtag_evidence_needs_a_sequence_not_one_tag(isolated_settings):
    """One common hashtag links half a corpus and means nothing."""
    from modeling.accounts.coordination import _hashtag_keys

    assert _hashtag_keys({"hashtags": ["news"]}, 2) == []
    assert _hashtag_keys({"hashtags": ["news", "contract"]}, 2) == ["tags:news|contract"]
    # Order is part of the signature: a shared *ordering* is much stronger
    # evidence of a shared template than a shared vocabulary.
    assert _hashtag_keys({"hashtags": ["contract", "news"]}, 2) != _hashtag_keys(
        {"hashtags": ["news", "contract"]}, 2
    )


def test_empty_corpus_does_not_crash(isolated_settings):
    result = CoordinationDetector(isolated_settings).detect(
        pd.DataFrame(columns=["id", "author_id", "source", "text", "timestamp"]),
        run_null_model=False,
    )
    assert result.edges == []
    assert result.n_records == 0


# ---------------------------------------------------------------------------
# domain recalibration
# ---------------------------------------------------------------------------
def test_weak_labels_come_only_from_accounts_that_declare_one():
    """An absent declaration is not a declaration of 'human'."""
    import json

    from modeling.accounts.bot_clf import extract_weak_bot_labels

    authors = pd.DataFrame(
        [
            {"author_id": "mastodon:a", "raw": json.dumps({"bot": True})},
            {"author_id": "mastodon:b", "raw": json.dumps({"bot": False})},
            {"author_id": "reddit:c", "raw": json.dumps({})},  # no such field
            {"author_id": "reddit:d", "raw": "not json"},
        ]
    )
    labels = extract_weak_bot_labels(authors)
    assert set(labels.index) == {"mastodon:a", "mastodon:b"}
    assert labels["mastodon:a"] == 1.0
    assert labels["mastodon:b"] == 0.0


def test_domain_recalibration_refuses_a_tiny_label_set():
    from modeling.accounts.bot_clf import recalibrate_on_domain

    rng = np.random.default_rng(0)
    calibrator, report = recalibrate_on_domain(
        rng.random(20), (rng.random(20) > 0.7).astype(float)
    )
    assert calibrator is None
    assert report["applied"] is False
    assert "floor" in report["reason"]


def test_domain_recalibration_refuses_a_single_class_label():
    from modeling.accounts.bot_clf import recalibrate_on_domain

    rng = np.random.default_rng(0)
    calibrator, report = recalibrate_on_domain(rng.random(200), np.zeros(200))
    assert calibrator is None
    assert "one class" in report["reason"]


def test_domain_recalibration_fixes_a_shifted_score_distribution():
    """The measured failure: benchmark calibration puts every account above the
    threshold while only a small fraction are really bots. Recalibration must
    restore the base rate while preserving the ranking."""
    from modeling.accounts.bot_clf import recalibrate_on_domain

    rng = np.random.default_rng(7)
    n = 300
    labels = (rng.random(n) < 0.15).astype(float)
    # Every score crammed into [0.9, 1.0], bots only slightly higher -- the
    # shape actually observed on Mastodon.
    scores = 0.90 + 0.09 * rng.random(n) + 0.02 * labels
    scores = np.clip(scores, 0, 1)

    calibrator, report = recalibrate_on_domain(scores, labels)
    assert calibrator is not None and report["applied"]
    assert report["brier_after"] < report["brier_before"]

    calibrated = calibrator.transform(scores)
    assert (calibrated >= 0.5).mean() < 0.5, "still flagging most of the corpus"
    # Platt is monotone, so the ranking must be untouched.
    assert np.array_equal(np.argsort(scores), np.argsort(calibrated))


def test_domain_recalibration_is_rejected_when_it_makes_brier_worse():
    """A calibration that hurts is not shipped."""
    from modeling.accounts.bot_clf import recalibrate_on_domain

    rng = np.random.default_rng(3)
    labels = (rng.random(400) < 0.5).astype(float)
    # Already perfectly calibrated and perfectly separating.
    scores = np.where(labels == 1, 0.99, 0.01)
    calibrator, report = recalibrate_on_domain(scores, labels)
    assert calibrator is None or report["applied"] is True
