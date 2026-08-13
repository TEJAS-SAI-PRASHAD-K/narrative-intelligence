"""Narrative clustering: id stability, metrics, and the dedupe guard.

No embedding model is loaded here. The vectors are constructed by hand, which
makes the cluster structure known in advance -- the point is to test the
clustering *machinery*, and a test whose expected answer depends on what a
384-dim transformer happens to produce is not a test.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from modeling.text.cluster import (
    NarrativeClusterer,
    mean_pairwise_cosine,
    peak_posts_per_hour,
    weighted_percentile,
)
from modeling.text.embed import EmbeddingResult, l2_normalize

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


def synthetic_corpus(n_clusters=3, per_cluster=20, dim=16, seed=0, jitter=0.08):
    """Well-separated clusters in a small vector space.

    Each cluster is a tight ball around its own random direction, so the correct
    answer is known and any failure is in the code, not in the data.
    """
    rng = np.random.default_rng(seed)
    centres = l2_normalize(rng.normal(size=(n_clusters, dim)))
    vectors, ids, rows = [], [], []
    for c in range(n_clusters):
        for i in range(per_cluster):
            vector = centres[c] + rng.normal(scale=jitter, size=dim)
            record_id = f"mastodon:c{c}-p{i}"
            vectors.append(vector)
            ids.append(record_id)
            rows.append({
                "id": record_id,
                "source": "mastodon" if i % 3 else "reddit",
                "author_id": f"mastodon:author{c}-{i % 7}",
                "text": f"cluster {c} post {i} about a distinct topic with its own words",
                "timestamp": BASE + timedelta(days=c, minutes=i * 9),
                "engagement": {"likes": i, "shares": None, "replies": None, "views": None},
                "domains": [f"site{c}.example"],
                "hashtags": [f"topic{c}"],
                # Well-separated simhashes: nothing should collapse as a duplicate.
                "simhash": ((c * 1000 + i) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
            })
    embeddings = EmbeddingResult(
        vectors=l2_normalize(np.array(vectors, dtype=np.float32)),
        record_ids=ids,
        model_name="test",
        model_version="v0",
        dim=dim,
        was_truncated=np.zeros(len(ids), dtype=bool),
    )
    return pd.DataFrame(rows), embeddings


def small_clusterer(settings, **overrides):
    clusterer = NarrativeClusterer(settings)
    clusterer.min_cluster_size = overrides.pop("min_cluster_size", 5)
    clusterer.min_samples = overrides.pop("min_samples", 2)
    for key, value in overrides.items():
        setattr(clusterer, key, value)
    return clusterer


# ---------------------------------------------------------------------------
# basic clustering
# ---------------------------------------------------------------------------
def test_finds_the_planted_clusters(isolated_settings):
    records, embeddings = synthetic_corpus()
    result = small_clusterer(isolated_settings).fit(records, embeddings)
    assert len(result.narratives) == 3
    assert result.noise_ratio < 0.2


def test_every_record_is_either_clustered_or_noise(isolated_settings):
    records, embeddings = synthetic_corpus()
    result = small_clusterer(isolated_settings).fit(records, embeddings)
    assigned = {m for n in result.narratives for m in n.member_ids} | set(result.noise_ids)
    assert assigned == set(embeddings.record_ids)


def test_narrative_carries_platforms_and_authors(isolated_settings):
    records, embeddings = synthetic_corpus()
    result = small_clusterer(isolated_settings).fit(records, embeddings)
    for narrative in result.narratives:
        assert narrative.author_count > 0
        assert set(narrative.platforms) <= {"mastodon", "reddit"}
        assert narrative.first_seen is not None and narrative.last_seen is not None
        assert narrative.first_seen <= narrative.last_seen


def test_representatives_are_spread_across_platforms(isolated_settings):
    """Three representatives all from one platform make a cross-platform
    narrative look single-platform, and give the summarizer a one-sided view."""
    records, embeddings = synthetic_corpus()
    result = small_clusterer(isolated_settings).fit(records, embeddings)
    source_of = records.set_index("id")["source"].to_dict()
    for narrative in result.narratives:
        platforms = {source_of[r] for r in narrative.representative_ids}
        assert len(platforms) >= 2, "representatives collapsed onto one platform"


def test_centroid_is_unit_length(isolated_settings):
    """Downstream cosine comparisons -- id carry-forward especially -- assume it."""
    records, embeddings = synthetic_corpus()
    result = small_clusterer(isolated_settings).fit(records, embeddings)
    for narrative in result.narratives:
        assert np.linalg.norm(narrative.centroid) == pytest.approx(1.0, abs=1e-5)


def test_centroid_width_matches_the_embedding_dimension(isolated_settings):
    """Phase 4 sizes its pgvector column from this; a mismatch is a migration
    failure found in production."""
    records, embeddings = synthetic_corpus(dim=24)
    result = small_clusterer(isolated_settings).fit(records, embeddings)
    for narrative in result.narratives:
        assert len(narrative.centroid) == 24


# ---------------------------------------------------------------------------
# dedupe: one repost swarm must not become a narrative
# ---------------------------------------------------------------------------
def test_near_duplicates_collapse_for_clustering_but_keep_their_members(isolated_settings):
    records, embeddings = synthetic_corpus(n_clusters=2, per_cluster=12)
    rows = records.to_dict("records")
    swarm_vectors, swarm_ids = [], []
    base_vector = embeddings.vectors[0]
    for i in range(30):
        record_id = f"mastodon:swarm-{i}"
        swarm_ids.append(record_id)
        nudge = np.random.default_rng(i).normal(scale=0.001, size=len(base_vector))
        swarm_vectors.append(base_vector + nudge)
        rows.append({
            "id": record_id,
            "source": "mastodon",
            "author_id": f"mastodon:bot{i}",
            "text": "identical reposted claim",
            "timestamp": BASE + timedelta(minutes=i),
            "engagement": {"likes": 0, "shares": None, "replies": None, "views": None},
            "domains": ["swarm.example"],
            "hashtags": ["swarm"],
            "simhash": 0xABCD_1234_5678_9000,  # identical -> one representative
        })
    merged = EmbeddingResult(
        vectors=l2_normalize(
            np.vstack([embeddings.vectors, np.array(swarm_vectors, dtype=np.float32)])
        ),
        record_ids=embeddings.record_ids + swarm_ids,
        model_name="test",
        model_version="v0",
        dim=embeddings.dim,
    )
    frame = pd.DataFrame(rows)

    clusterer = small_clusterer(isolated_settings)
    representatives, members = clusterer.collapse_duplicates(frame, merged)
    assert len(representatives) < len(merged.record_ids)
    # The full member list survives: size must still reflect reality.
    all_members = {m for group in members.values() for m in group}
    assert all_members == set(merged.record_ids)

    result = clusterer.fit(frame, merged)
    total = sum(n.size for n in result.narratives) + len(result.noise_ids)
    assert total == len(merged.record_ids)


def test_records_without_a_simhash_are_never_merged(isolated_settings):
    records, embeddings = synthetic_corpus(n_clusters=2, per_cluster=8)
    records["simhash"] = None
    clusterer = small_clusterer(isolated_settings)
    representatives, _ = clusterer.collapse_duplicates(records, embeddings)
    assert len(representatives) == len(embeddings.record_ids)


# ---------------------------------------------------------------------------
# cross-run identity -- an acceptance criterion
# ---------------------------------------------------------------------------
def _previous_table(result):
    return pd.DataFrame(
        [
            {"narrative_id": n.narrative_id, "centroid": [float(v) for v in n.centroid]}
            for n in result.narratives
        ]
    )


def test_ids_are_stable_across_two_runs_on_identical_data(isolated_settings):
    records, embeddings = synthetic_corpus()
    clusterer = small_clusterer(isolated_settings)
    first = clusterer.fit(records, embeddings)
    second = clusterer.fit(records, embeddings, previous=_previous_table(first))
    assert {n.narrative_id for n in first.narratives} == {n.narrative_id for n in second.narratives}
    assert all(n.id_origin == "carried" for n in second.narratives)


def test_ids_are_stable_across_runs_on_overlapping_data(isolated_settings):
    """The real case: the corpus grew since the last run.

    The UI shows narrative ids and lets users rename them. Minting fresh ids
    because a few posts arrived would orphan every one of those edits.
    """
    records, embeddings = synthetic_corpus(per_cluster=20, seed=1)
    clusterer = small_clusterer(isolated_settings)
    first = clusterer.fit(records, embeddings)

    grown_records, grown_embeddings = synthetic_corpus(per_cluster=28, seed=1)
    second = clusterer.fit(grown_records, grown_embeddings, previous=_previous_table(first))

    carried = {n.narrative_id for n in second.narratives if n.id_origin == "carried"}
    assert carried == {n.narrative_id for n in first.narratives}


def test_a_genuinely_new_cluster_gets_a_new_id(isolated_settings):
    records, embeddings = synthetic_corpus(n_clusters=2, seed=3)
    clusterer = small_clusterer(isolated_settings)
    first = clusterer.fit(records, embeddings)

    records_3, embeddings_3 = synthetic_corpus(n_clusters=3, seed=3)
    second = clusterer.fit(records_3, embeddings_3, previous=_previous_table(first))
    origins = [n.id_origin for n in second.narratives]
    assert "new" in origins
    assert origins.count("carried") == len(first.narratives)


def test_a_disappeared_narrative_is_logged_as_a_death(isolated_settings):
    records, embeddings = synthetic_corpus(n_clusters=3, seed=5)
    clusterer = small_clusterer(isolated_settings)
    first = clusterer.fit(records, embeddings)

    subset, subset_embeddings = synthetic_corpus(n_clusters=2, seed=5)
    second = clusterer.fit(subset, subset_embeddings, previous=_previous_table(first))
    assert len(second.transitions["died"]) >= 1


def test_an_old_id_is_claimed_by_at_most_one_new_cluster(isolated_settings):
    """Two clusters matching one old id means the narrative split. The better
    match keeps the id; handing it to both would break the primary key."""
    records, embeddings = synthetic_corpus(n_clusters=3, seed=7)
    clusterer = small_clusterer(isolated_settings)
    first = clusterer.fit(records, embeddings)
    second = clusterer.fit(records, embeddings, previous=_previous_table(first))
    ids = [n.narrative_id for n in second.narratives]
    assert len(ids) == len(set(ids))


def test_id_minting_is_deterministic_without_a_previous_table(isolated_settings):
    records, embeddings = synthetic_corpus()
    clusterer = small_clusterer(isolated_settings)
    a = {n.narrative_id for n in clusterer.fit(records, embeddings).narratives}
    b = {n.narrative_id for n in clusterer.fit(records, embeddings).narratives}
    assert a == b


# ---------------------------------------------------------------------------
# cluster metrics
# ---------------------------------------------------------------------------
def test_mean_pairwise_cosine_matches_the_naive_computation():
    """The O(n*d) identity must agree with the O(n^2*d) version it replaces."""
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.normal(size=(40, 8)))
    similarity = vectors @ vectors.T
    n = len(vectors)
    naive = (similarity.sum() - n) / (n * (n - 1))
    assert mean_pairwise_cosine(vectors) == pytest.approx(naive, abs=1e-5)


def test_coherence_is_one_for_a_singleton_and_high_for_a_tight_cluster():
    assert mean_pairwise_cosine(np.array([[1.0, 0.0]])) == 1.0
    tight = l2_normalize(np.array([[1.0, 0.01], [1.0, 0.0], [1.0, -0.01]]))
    assert mean_pairwise_cosine(tight) > 0.99


def test_velocity_is_a_peak_not_a_mean():
    """A narrative that produced 200 posts in an hour then went quiet for a week
    is the interesting case; a lifetime mean erases it."""
    burst = pd.Series([BASE + timedelta(minutes=i) for i in range(20)])
    trickle = pd.Series([BASE + timedelta(days=i) for i in range(20)])
    assert peak_posts_per_hour(burst) == 20.0
    assert peak_posts_per_hour(trickle) == 1.0


def test_velocity_uses_real_seconds_not_raw_datetime_units():
    """pandas 3 stores datetime64 as microseconds. Dividing the raw int64 by 1e9
    compresses the timeline 1000x, so one "hour" swallows six weeks and every
    cluster reports its whole size as its peak velocity."""
    spread = pd.Series([BASE + timedelta(hours=3 * i) for i in range(12)])
    assert peak_posts_per_hour(spread) == 1.0


def test_velocity_handles_a_single_post():
    assert peak_posts_per_hour(pd.Series([BASE])) == 1.0
    assert peak_posts_per_hour(pd.Series([], dtype="datetime64[ns, UTC]")) == 0.0


def test_weighted_percentile_reduces_to_the_plain_one_with_equal_weights():
    values = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    weights = np.ones(5)
    assert weighted_percentile(values, weights, 50) == pytest.approx(
        np.percentile(values, 50), abs=0.05
    )


def test_weighted_percentile_follows_the_weight():
    values = np.array([0.1, 0.9])
    assert weighted_percentile(values, np.array([1.0, 99.0]), 50) > 0.8
    assert weighted_percentile(values, np.array([99.0, 1.0]), 50) < 0.2


# ---------------------------------------------------------------------------
# severity
# ---------------------------------------------------------------------------
def test_severity_is_null_when_the_misinfo_model_has_not_run(isolated_settings):
    """A severity computed from nothing would be a fabrication."""
    records, embeddings = synthetic_corpus()
    result = small_clusterer(isolated_settings).fit(records, embeddings, record_scores=None)
    assert all(n.severity is None for n in result.narratives)


def test_severity_is_null_when_every_member_score_is_null(isolated_settings):
    records, embeddings = synthetic_corpus()
    scores = pd.DataFrame({"record_id": records["id"], "misinfo_prob": [None] * len(records)})
    result = small_clusterer(isolated_settings).fit(records, embeddings, record_scores=scores)
    assert all(n.severity is None for n in result.narratives)


def _severity_of_cluster_zero(settings, probabilities_for_c0, *, engagement=None):
    """Score cluster 0's members as given, everything else neutral.

    Two planted clusters, not one: HDBSCAN is density-based, and a single
    undifferentiated blob has no density contrast to find. That is correct
    behaviour, so the fixture provides something to contrast against.

    Engagement defaults to unmeasurable (weight 1 for every post), which
    isolates the tail-mean aggregation from the engagement weighting. The
    weighting is tested separately -- mixing them makes a failure ambiguous.
    """
    records, embeddings = synthetic_corpus(n_clusters=2, per_cluster=len(probabilities_for_c0))
    in_c0 = records["id"].str.startswith("mastodon:c0-")
    values = []
    c0_iter = iter(probabilities_for_c0)
    for is_c0 in in_c0:
        values.append(next(c0_iter) if is_c0 else 0.01)
    if engagement is None:
        records = records.copy()
        records["engagement"] = [
            {"likes": None, "shares": None, "replies": None, "views": None}
        ] * len(records)
    else:
        records = records.copy()
        records["engagement"] = engagement
    scores = pd.DataFrame({"record_id": records["id"], "misinfo_prob": values})
    result = small_clusterer(settings).fit(records, embeddings, record_scores=scores)

    c0_ids = set(records.loc[in_c0, "id"])
    for narrative in result.narratives:
        if len(set(narrative.member_ids) & c0_ids) > len(narrative.member_ids) / 2:
            return narrative.severity
    pytest.fail("cluster 0 was not recovered")


def test_severity_is_not_dragged_down_by_a_neutral_tail(isolated_settings):
    """The whole reason for a high percentile rather than a mean: a narrative is
    a few alarming posts surrounded by discussion *about* them."""
    probabilities = [0.95] * 10 + [0.02] * 30
    severity = _severity_of_cluster_zero(isolated_settings, probabilities)
    assert severity is not None
    assert severity > np.mean(probabilities), "a plain mean would have buried the signal"


def test_severity_stays_below_the_maximum(isolated_settings):
    """One confident false positive must not set the whole narrative alight."""
    severity = _severity_of_cluster_zero(isolated_settings, [0.99] + [0.05] * 39)
    assert severity is not None
    assert severity < 0.99


def test_severity_follows_reach_when_engagement_is_measurable(isolated_settings):
    """A claim seen 50,000 times is more severe than the same claim seen twice.

    Same scores in both runs; only the reach of the alarming posts differs.
    """
    probabilities = [0.95] * 10 + [0.02] * 30
    amplified = [
        {"likes": 5000 if i < 10 else 1, "shares": None, "replies": None, "views": None}
        for i in range(40)
    ] * 2
    ignored = [
        {"likes": 1 if i < 10 else 5000, "shares": None, "replies": None, "views": None}
        for i in range(40)
    ] * 2
    loud = _severity_of_cluster_zero(isolated_settings, probabilities, engagement=amplified)
    quiet = _severity_of_cluster_zero(isolated_settings, probabilities, engagement=ignored)
    assert loud > quiet


def test_unmeasurable_engagement_is_weight_one_not_zero(isolated_settings):
    """ConvoKit Reddit exposes no engagement at all. Treating null as zero-reach
    would silently exclude an entire platform from severity."""
    from modeling.text.cluster import _engagement_weights

    rows = pd.DataFrame(
        {"engagement": [{"likes": None, "shares": None, "replies": None, "views": None}] * 3}
    )
    assert (_engagement_weights(rows) == 1.0).all()


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def test_diagnostics_report_silhouette_and_noise(isolated_settings):
    records, embeddings = synthetic_corpus()
    result = small_clusterer(isolated_settings).fit(records, embeddings)
    assert result.diagnostics["n_clusters"] == 3
    assert result.diagnostics["silhouette"] is not None
    assert 0.0 <= result.noise_ratio <= 1.0


def test_silhouette_is_declined_rather_than_faked_on_one_cluster(isolated_settings):
    records, embeddings = synthetic_corpus(n_clusters=1, per_cluster=20)
    result = small_clusterer(isolated_settings).fit(records, embeddings)
    assert result.diagnostics["silhouette"] is None
    assert "silhouette_note" in result.diagnostics
