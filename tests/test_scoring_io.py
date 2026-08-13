"""The output contract: schemas, joinability, idempotency, resumability.

Phase 4 loads these tables. Every assertion here is something that, if it broke
silently, would surface as a wrong number on a dashboard rather than as a crash.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from modeling.io import (
    TABLES,
    CorpusReader,
    OrphanRowsError,
    ScoredStore,
    _as_version_dict,
    empty_emotion,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    from modeling import config as C

    C.get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    settings = C.get_settings()
    settings.ensure_dirs()
    yield ScoredStore(settings)
    C.get_settings.cache_clear()


def make_record_scores(ids, *, versions=None, toxicity=0.1):
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    return pd.DataFrame(
        [
            {
                "record_id": rid,
                "source": rid.split(":", 1)[0],
                "toxicity": toxicity,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "emotion": empty_emotion(),
                "anomaly_score": 0.2,
                "skip_reasons": [],
                "model_versions": versions or {"toxicity": "v0.1.0"},
                "scored_at": now,
            }
            for rid in ids
        ]
    )


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------
def test_every_contract_table_has_keys_and_a_clock():
    for name, table in TABLES.items():
        fields = {f.name for f in table.schema}
        assert set(table.keys) <= fields, f"{name}: key column not in schema"
        assert fields & {"scored_at", "generated_at"}, f"{name}: no timestamp column"
        for key in table.keys:
            assert not table.schema.field(key).nullable, f"{name}.{key} must be non-nullable"


def test_round_trip_preserves_nested_types(store):
    frame = make_record_scores(["reddit:a", "reddit:b"])
    store.write("record_scores", frame, known_keys={"reddit:a", "reddit:b"})
    back = store.read("record_scores")
    assert len(back) == 2
    emotion = back.iloc[0]["emotion"]
    assert set(emotion) == set(empty_emotion())
    assert _as_version_dict(back.iloc[0]["model_versions"]) == {"toxicity": "v0.1.0"}


def test_null_survives_the_round_trip(store):
    """A null must come back a null. Phase 4 renders it as "not assessed"; a
    0.0 substituted anywhere in this path would render as a confident score."""
    frame = make_record_scores(["reddit:a"])
    frame["misinfo_prob"] = None
    frame["stance"] = None
    store.write("record_scores", frame, known_keys={"reddit:a"})
    back = store.read("record_scores")
    assert pd.isna(back.iloc[0]["misinfo_prob"])
    assert pd.isna(back.iloc[0]["stance"])


def test_unexpected_column_is_refused(store):
    frame = make_record_scores(["reddit:a"])
    frame["my_experimental_score"] = 0.5
    with pytest.raises(ValueError, match="output contract is closed"):
        store.write("record_scores", frame, known_keys={"reddit:a"})


# ---------------------------------------------------------------------------
# joinability
# ---------------------------------------------------------------------------
def test_orphan_record_id_is_rejected(store):
    frame = make_record_scores(["reddit:a", "reddit:ghost"])
    with pytest.raises(OrphanRowsError, match="orphan record_id"):
        store.write("record_scores", frame, known_keys={"reddit:a"})


def test_orphan_author_id_is_rejected(store):
    frame = pd.DataFrame(
        [
            {
                "author_id": "mastodon:ghost",
                "source": "mastodon",
                "bot_prob": 0.4,
                "model_versions": {"bot": "v0.1.0"},
                "scored_at": datetime.now(timezone.utc),
            }
        ]
    )
    with pytest.raises(OrphanRowsError, match="orphan author_id"):
        store.write("author_scores", frame, known_keys={"mastodon:real"})


# ---------------------------------------------------------------------------
# idempotency and resumability -- acceptance criteria
# ---------------------------------------------------------------------------
def test_rerunning_with_unchanged_input_writes_zero_rows(store):
    ids = ["reddit:a", "reddit:b", "mastodon:c"]
    known = set(ids)
    first = store.write("record_scores", make_record_scores(ids), known_keys=known)
    assert first["written"] == 3

    # Same content, different clock -- must still be a no-op.
    later = make_record_scores(ids)
    later["scored_at"] = datetime.now(timezone.utc) + timedelta(hours=5)
    second = store.write("record_scores", later, known_keys=known)
    assert second == {"written": 0, "updated": 0, "unchanged": 3}


def test_a_changed_score_is_an_update_not_a_duplicate(store):
    ids = ["reddit:a"]
    store.write("record_scores", make_record_scores(ids), known_keys=set(ids))
    changed = store.write(
        "record_scores", make_record_scores(ids, toxicity=0.9), known_keys=set(ids)
    )
    assert changed == {"written": 0, "updated": 1, "unchanged": 0}
    back = store.read("record_scores")
    assert len(back) == 1
    assert back.iloc[0]["toxicity"] == pytest.approx(0.9)


def test_already_scored_lets_a_killed_run_resume(store):
    ids = ["reddit:a", "reddit:b"]
    store.write("record_scores", make_record_scores(ids), known_keys=set(ids))
    done = store.already_scored("record_scores", {"toxicity": "v0.1.0"})
    assert done == {("reddit:a",), ("reddit:b",)}


def test_a_version_bump_invalidates_previous_scores(store):
    ids = ["reddit:a"]
    store.write("record_scores", make_record_scores(ids), known_keys=set(ids))
    # After a retrain, the old rows are no longer current and must be re-scored.
    assert store.already_scored("record_scores", {"toxicity": "v0.2.0"}) == set()


def test_partial_version_match_is_treated_as_current(store):
    """A row scored by aux+misinfo is still current for an aux-only rerun."""
    ids = ["reddit:a"]
    store.write(
        "record_scores",
        make_record_scores(ids, versions={"toxicity": "v0.1.0", "misinfo": "v0.1.0"}),
        known_keys=set(ids),
    )
    assert store.already_scored("record_scores", {"toxicity": "v0.1.0"}) == {("reddit:a",)}


def test_composite_key_tables_merge_on_all_key_columns(store):
    now = datetime.now(timezone.utc)
    edges = pd.DataFrame(
        [
            {
                "src_author_id": "mastodon:a",
                "dst_author_id": "mastodon:b",
                "weight": 0.5,
                "evidence": "near_dup",
                "observations": 3,
                "window_start": now,
                "window_end": now,
                "generated_at": now,
            },
            {
                "src_author_id": "mastodon:a",
                "dst_author_id": "mastodon:b",
                "weight": 0.2,
                "evidence": "cotweet",  # same pair, different evidence -> distinct row
                "observations": 1,
                "window_start": now,
                "window_end": now,
                "generated_at": now,
            },
        ]
    )
    store.write("coordination_edges", edges)
    assert len(store.read("coordination_edges")) == 2


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def test_manifest_records_rows_and_versions(store):
    ids = ["reddit:a"]
    store.write("record_scores", make_record_scores(ids), known_keys=set(ids))
    store.update_manifest(table="record_scores", rows=1, model_versions={"toxicity": "v0.1.0"})
    manifest = store.manifest()
    assert manifest["record_scores"]["rows"] == 1
    assert manifest["record_scores"]["model_versions"] == {"toxicity": "v0.1.0"}
    assert "input_manifest_hash" in manifest["record_scores"]


# ---------------------------------------------------------------------------
# corpus reader
# ---------------------------------------------------------------------------
def test_corpus_reader_is_empty_not_broken_without_a_corpus(tmp_path):
    reader = CorpusReader(root=tmp_path / "nothing-here")
    assert reader.available_sources() == []
    assert len(reader.records()) == 0
    assert reader.record_ids() == set()


def test_corpus_reader_reads_the_real_corpus_when_present():
    """Skips cleanly when the corpus has not been built -- pytest must pass on a
    clean clone with no network and no data."""
    reader = CorpusReader()
    if not reader.available_sources():
        pytest.skip("no Phase 1 corpus on disk")
    frame = reader.records(limit=50)
    assert len(frame) <= 50
    assert {"id", "source", "text", "timestamp", "author_id"} <= set(frame.columns)
    # Deterministic order is what makes two scoring runs byte-identical.
    assert frame["id"].tolist() == sorted(frame["id"].tolist())
