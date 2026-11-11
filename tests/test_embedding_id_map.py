"""The record-id map that ties cached vectors to posts.

The cache is keyed by the prepared (truncated) text, so a post's raw text hashes to the
wrong key whenever the article was long. These tests cover the id map that closes that gap
and the loader rules that keep it from ever mapping a post to the wrong vector. No weights
are loaded: the encoder is a fake.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
import pytest

from modeling import config as C
from modeling.text import embed as E
from tests.conftest_api import requires_postgres
from tests.test_embed_alternative import FakeEncoder

DEFAULT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LONG_TEXT = " ".join(f"Sentence number {i} about the claim under study." for i in range(120))


@pytest.fixture
def settings(tmp_path, monkeypatch):
    C.get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        E, "_load_sentence_transformer", lambda name, device, local_only: FakeEncoder(384)
    )
    yield C.get_settings()
    C.get_settings.cache_clear()


def _frame(rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"id": list(rows), "text": list(rows.values())})


def _embed(settings, rows: dict[str, str]) -> E.Embedder:
    embedder = E.Embedder(settings)
    embedder.embed_records(_frame(rows))
    return embedder


# ---------------------------------------------------------------------------
# writing the map
# ---------------------------------------------------------------------------
def test_embed_records_writes_an_id_map_keyed_by_the_prepared_text(settings):
    embedder = _embed(settings, {"news:long": LONG_TEXT, "news:short": "A short post."})
    prepared, truncated = embedder.prepare(LONG_TEXT)
    assert truncated and prepared != LONG_TEXT

    ids = json.loads(embedder.cache.ids_path.read_text(encoding="utf-8"))
    assert ids["news:long"]["key"] == E.EmbeddingCache.key(prepared)
    assert ids["news:long"]["key"] != E.EmbeddingCache.key(LONG_TEXT)
    assert ids["news:long"]["text_hash"] == E.EmbeddingCache.key(LONG_TEXT)
    assert ids["news:short"]["key"] == E.EmbeddingCache.key("A short post.")
    keys = json.loads(embedder.cache.index_path.read_text(encoding="utf-8"))
    assert {entry["key"] for entry in ids.values()} <= set(keys)


def test_a_second_run_merges_into_the_existing_id_map(settings):
    first = _embed(settings, {"news:a": "First post text."})
    second = _embed(settings, {"news:b": "Second post text."})

    ids = json.loads(second.cache.ids_path.read_text(encoding="utf-8"))
    assert set(ids) == {"news:a", "news:b"}
    assert ids["news:a"]["key"] == E.EmbeddingCache.key("First post text.")
    assert first.cache.ids_path == second.cache.ids_path


def test_the_id_map_write_leaves_no_temp_file(settings):
    embedder = _embed(settings, {"news:a": "First post text."})
    leftovers = [p.name for p in embedder.cache.ids_path.parent.glob("*.tmp")]
    assert leftovers == []


def test_a_failed_id_map_write_removes_its_unique_temp_file(settings, monkeypatch):
    embedder = _embed(settings, {"news:a": "First post text."})

    def refuse(*args, **kwargs):
        raise OSError("rename refused")

    monkeypatch.setattr(E.os, "replace", refuse)
    with pytest.raises(OSError):
        embedder.cache.link_ids({"news:b": ("First post text.", "First post text.")})
    assert list(embedder.cache.ids_path.parent.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# resolving posts to rows -- no database
# ---------------------------------------------------------------------------
def _resolve(posts, keys, ids, rows, **kwargs):
    from app.etl.scored_loader import _resolve_vector_rows

    return _resolve_vector_rows(posts, keys, ids, rows, E.EmbeddingCache.key, **kwargs)


def _entry(key: str, raw: str) -> dict[str, str]:
    return {"key": key, "text_hash": E.EmbeddingCache.key(raw)}


def test_a_long_post_resolves_through_the_id_map(settings):
    embedder = _embed(settings, {"news:long": LONG_TEXT})
    keys = json.loads(embedder.cache.index_path.read_text(encoding="utf-8"))
    ids = json.loads(embedder.cache.ids_path.read_text(encoding="utf-8"))

    with_ids, misses = _resolve([("news:long", LONG_TEXT)], keys, ids, len(keys))
    assert with_ids == [("news:long", 0)] and misses == {}

    # Without the map the raw text hashes to a key the cache never held.
    without, misses = _resolve([("news:long", LONG_TEXT)], keys, None, len(keys))
    assert without == [] and misses == {"no_cached_vector": 1}


def test_without_an_id_map_a_short_post_still_resolves_by_text(settings):
    embedder = _embed(settings, {"news:short": "A short post."})
    keys = json.loads(embedder.cache.index_path.read_text(encoding="utf-8"))

    matches, misses = _resolve([("news:short", "A short post.")], keys, None, len(keys))
    assert matches == [("news:short", 0)] and misses == {}


def test_a_stale_id_map_entry_is_a_counted_miss_not_a_wrong_vector():
    keys = {"k0": 0, "k1": 1}
    ids = {
        "p:absent": _entry("gone", "x"),
        "p:range": _entry("k1", "y"),
        "p:ok": _entry("k0", "z"),
    }
    posts = [("p:absent", "x"), ("p:range", "y"), ("p:ok", "z")]

    # Two rows in `keys` but a matrix of one row: k1 points past the end.
    matches, misses = _resolve(posts, keys, ids, 1)
    assert matches == [("p:ok", 0)]
    assert misses == {"id_map_key_absent": 1, "id_map_row_out_of_range": 1}


def test_a_stale_id_entry_does_not_fall_back_to_the_text_key():
    text = "Some text."
    keys = {E.EmbeddingCache.key(text): 0}
    matches, misses = _resolve([("p:1", text)], keys, {"p:1": _entry("gone", text)}, 1)
    assert matches == [] and misses == {"id_map_key_absent": 1}


def test_a_post_missing_from_the_id_map_falls_back_to_its_text_key():
    text = "Some text."
    keys = {E.EmbeddingCache.key(text): 0}
    matches, misses = _resolve([("p:1", text)], keys, {"p:other": _entry("k", "q")}, 1)
    assert matches == [("p:1", 0)] and misses == {}


def test_an_edited_post_is_a_miss_not_the_old_vector():
    old, new = "The original text.", "The edited text."
    keys = {E.EmbeddingCache.key(old): 0, E.EmbeddingCache.key(new): 1}
    ids = {"p:1": _entry(E.EmbeddingCache.key(old), old)}

    matches, misses = _resolve([("p:1", new)], keys, ids, 2)
    assert matches == [] and misses == {"id_map_text_changed": 1}


def test_the_old_flat_id_map_format_is_a_counted_miss():
    text = "Some text."
    key = E.EmbeddingCache.key(text)
    matches, misses = _resolve([("p:1", text)], {key: 0}, {"p:1": key}, 1)
    assert matches == [] and misses == {"id_map_legacy_format": 1}


def test_an_unreadable_map_is_reported_apart_from_never_embedded():
    matches, misses = _resolve([("p:1", "unseen")], {}, None, 0, map_unreadable=True)
    assert matches == [] and misses == {"id_map_unreadable": 1}


def test_cache_consistency_check():
    from app.etl.scored_loader import _cache_is_consistent

    assert _cache_is_consistent({"a": 0, "b": 1}, 2) is None
    assert "keys but" in _cache_is_consistent({"a": 0}, 2)
    assert "duplicated" in _cache_is_consistent({"a": 0, "b": 0}, 2)
    assert "duplicated" in _cache_is_consistent({"a": 0, "b": 5}, 2)


# ---------------------------------------------------------------------------
# the loader end to end
# ---------------------------------------------------------------------------
@pytest.fixture
def project(migrated_db):
    from app.db import sync_session
    from app.models.core import Project

    project_id = uuid.uuid4()
    with sync_session() as session:
        session.add(Project(id=project_id, slug="embed-map", name="Embed map"))
    return project_id


def _load_posts(project_id, tmp_path, rows):
    from app.db import sync_session
    from app.etl.parquet_loader import load_source
    from tests.test_etl import _row, _write_partition

    path = _write_partition(
        tmp_path / "normalized",
        "reddit",
        "2026-05-01",
        [_row(id=i, native_id=i.split(":")[1], text=t) for i, t in rows.items()],
    )
    with sync_session() as session:
        load_source(
            session,
            project_id=project_id,
            source="reddit",
            files=[path],
            rejects_dir=tmp_path / "rejects",
        )


def _loaded_ids(project_id) -> set[str]:
    from sqlalchemy import text

    from app.db import sync_session

    with sync_session() as session:
        return {
            row[0]
            for row in session.execute(
                text(
                    "SELECT e.post_id FROM post_embeddings_384 e "
                    "JOIN posts p ON p.id = e.post_id WHERE p.project_id = :p"
                ),
                {"p": project_id},
            )
        }


@requires_postgres
def test_load_embeddings_gives_a_long_post_its_vector_by_id(project, settings, tmp_path):
    from app.db import sync_session
    from app.etl.scored_loader import load_embeddings

    rows = {"reddit:t3_long": LONG_TEXT, "reddit:t3_short": "A short post."}
    _load_posts(project, tmp_path, rows)
    _embed(settings, rows)

    with sync_session() as session:
        load_embeddings(session, settings.data_dir, project)
    assert _loaded_ids(project) == set(rows)


@requires_postgres
def test_load_embeddings_on_an_old_cache_uses_text_keys_only(project, settings, tmp_path):
    from app.db import sync_session
    from app.etl.scored_loader import load_embeddings

    rows = {"reddit:t3_long": LONG_TEXT, "reddit:t3_short": "A short post."}
    _load_posts(project, tmp_path, rows)
    embedder = _embed(settings, rows)
    embedder.cache.ids_path.unlink()

    with sync_session() as session:
        load_embeddings(session, settings.data_dir, project)
    assert _loaded_ids(project) == {"reddit:t3_short"}


@requires_postgres
def test_load_embeddings_skips_a_stale_id_entry(project, settings, tmp_path, caplog):
    from app.db import sync_session
    from app.etl.scored_loader import load_embeddings

    rows = {"reddit:t3_long": LONG_TEXT, "reddit:t3_short": "A short post."}
    _load_posts(project, tmp_path, rows)
    embedder = _embed(settings, rows)
    ids = json.loads(embedder.cache.ids_path.read_text(encoding="utf-8"))
    ids["reddit:t3_long"]["key"] = "0" * 32
    embedder.cache.ids_path.write_text(json.dumps(ids), encoding="utf-8")

    with caplog.at_level("WARNING"), sync_session() as session:
        load_embeddings(session, settings.data_dir, project)
    assert _loaded_ids(project) == {"reddit:t3_short"}
    assert "id_map_key_absent=1" in caplog.text


def test_embedding_matrix_rows_are_untouched_by_linking(settings):
    embedder = _embed(settings, {"news:a": "First post text."})
    before = np.load(embedder.cache.matrix_path)
    embedder.cache.link_ids({"news:b": ("First post text.", "First post text.")})
    assert np.array_equal(np.load(embedder.cache.matrix_path), before)


def _copy_cache_as(embedder: E.Embedder, slug: str, *, scale: float) -> None:
    """A second cache of the same width, as compare-embedder leaves behind."""
    directory = embedder.cache.matrix_path.parent
    np.save(directory / f"{slug}.npy", np.load(embedder.cache.matrix_path) * scale)
    (directory / f"{slug}.keys.json").write_text(
        embedder.cache.index_path.read_text(encoding="utf-8"), encoding="utf-8"
    )


def _stored_models(project_id) -> set[str]:
    from sqlalchemy import text

    from app.db import sync_session

    with sync_session() as session:
        return {
            row[0]
            for row in session.execute(
                text(
                    "SELECT e.model FROM post_embeddings_384 e "
                    "JOIN posts p ON p.id = e.post_id WHERE p.project_id = :p"
                ),
                {"p": project_id},
            )
        }


@requires_postgres
def test_only_the_configured_cache_is_written(project, settings, tmp_path, caplog):
    from app.db import sync_session
    from app.etl.scored_loader import load_embeddings

    rows = {"reddit:t3_a": "First post text.", "reddit:t3_b": "Second post text."}
    _load_posts(project, tmp_path, rows)
    embedder = _embed(settings, rows)
    _copy_cache_as(embedder, "other-multilingual-v9", scale=-1.0)

    with caplog.at_level("WARNING"), sync_session() as session:
        load_embeddings(session, settings.data_dir, project)

    assert _stored_models(project) == {embedder.cache.matrix_path.stem}
    assert caplog.text.count("skipping embedding cache other-multilingual-v9.keys.json") == 1


@requires_postgres
def test_an_absent_configured_cache_loads_nothing_and_says_so(project, settings, tmp_path, caplog):
    from app.db import sync_session
    from app.etl.scored_loader import load_embeddings

    rows = {"reddit:t3_a": "First post text."}
    _load_posts(project, tmp_path, rows)
    embedder = _embed(settings, rows)
    _copy_cache_as(embedder, "other-multilingual-v9", scale=1.0)
    embedder.cache.matrix_path.unlink()
    embedder.cache.index_path.unlink()

    with caplog.at_level("WARNING"), sync_session() as session:
        load_embeddings(session, settings.data_dir, project)

    assert _loaded_ids(project) == set()
    assert "configured embedding cache" in caplog.text


@requires_postgres
def test_a_cache_whose_keys_and_matrix_disagree_is_skipped(project, settings, tmp_path, caplog):
    from app.db import sync_session
    from app.etl.scored_loader import load_embeddings

    rows = {"reddit:t3_a": "First post text.", "reddit:t3_b": "Second post text."}
    _load_posts(project, tmp_path, rows)
    embedder = _embed(settings, rows)
    np.save(embedder.cache.matrix_path, np.load(embedder.cache.matrix_path)[:1])

    with caplog.at_level("ERROR"), sync_session() as session:
        load_embeddings(session, settings.data_dir, project)

    assert _loaded_ids(project) == set()
    assert "keys but 1 matrix rows" in caplog.text


@requires_postgres
def test_an_edited_post_gets_no_vector_from_the_loader(project, settings, tmp_path):
    from app.db import sync_session
    from app.etl.scored_loader import load_embeddings

    _load_posts(project, tmp_path, {"reddit:t3_a": "The original text."})
    _embed(settings, {"reddit:t3_a": "The original text."})
    from sqlalchemy import text

    with sync_session() as session:
        session.execute(
            text("UPDATE posts SET text = 'The edited text.' WHERE id = 'reddit:t3_a'")
        )

    with sync_session() as session:
        load_embeddings(session, settings.data_dir, project)
    assert _loaded_ids(project) == set()


@requires_postgres
def test_an_unreadable_id_map_is_named_in_the_warning(project, settings, tmp_path, caplog):
    from app.db import sync_session
    from app.etl.scored_loader import load_embeddings

    rows = {"reddit:t3_long": LONG_TEXT}
    _load_posts(project, tmp_path, rows)
    embedder = _embed(settings, rows)
    embedder.cache.ids_path.write_text("{not json", encoding="utf-8")

    with caplog.at_level("WARNING"), sync_session() as session:
        load_embeddings(session, settings.data_dir, project)
    assert "id_map_unreadable=1" in caplog.text
    assert "id map absent" not in caplog.text


# ---------------------------------------------------------------------------
# a batch that repeats a text
# ---------------------------------------------------------------------------
SAME = "Two records share exactly this text."


def test_a_repeated_text_in_one_batch_gets_one_row_and_one_key(settings):
    embedder = _embed(settings, {"news:a": SAME, "news:b": SAME, "news:c": "Another post."})

    keys = json.loads(embedder.cache.index_path.read_text(encoding="utf-8"))
    matrix = np.load(embedder.cache.matrix_path)
    assert len(keys) == 2 and matrix.shape[0] == 2
    assert sorted(keys.values()) == [0, 1]


def test_a_cache_with_repeated_texts_survives_a_reload(settings, caplog):
    _embed(settings, {"news:a": SAME, "news:b": SAME})

    with caplog.at_level("WARNING"):
        reloaded = E.EmbeddingCache(DEFAULT_NAME, E.Embedder(settings).version, 384, settings)
    assert reloaded.get(SAME) is not None
    assert "discarding" not in caplog.text


def test_both_records_of_a_duplicate_pair_resolve_to_the_shared_row(settings):
    embedder = _embed(settings, {"news:a": SAME, "news:b": SAME})
    keys = json.loads(embedder.cache.index_path.read_text(encoding="utf-8"))
    ids = json.loads(embedder.cache.ids_path.read_text(encoding="utf-8"))

    matches, misses = _resolve([("news:a", SAME), ("news:b", SAME)], keys, ids, len(keys))
    assert matches == [("news:a", 0), ("news:b", 0)] and misses == {}


def test_a_row_key_mismatch_warning_states_the_real_reason(settings, caplog):
    embedder = _embed(settings, {"news:a": "First.", "news:b": "Second."})
    np.save(embedder.cache.matrix_path, np.load(embedder.cache.matrix_path)[:1])

    with caplog.at_level("WARNING"):
        E.EmbeddingCache(embedder.model_name, embedder.version, 384, settings)
    assert "1 matrix rows but 2 keys" in caplog.text
