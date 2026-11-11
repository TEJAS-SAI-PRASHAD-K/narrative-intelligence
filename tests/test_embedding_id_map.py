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
    assert ids["news:long"] == E.EmbeddingCache.key(prepared)
    assert ids["news:long"] != E.EmbeddingCache.key(LONG_TEXT)
    assert ids["news:short"] == E.EmbeddingCache.key("A short post.")
    keys = json.loads(embedder.cache.index_path.read_text(encoding="utf-8"))
    assert set(ids.values()) <= set(keys)


def test_a_second_run_merges_into_the_existing_id_map(settings):
    first = _embed(settings, {"news:a": "First post text."})
    second = _embed(settings, {"news:b": "Second post text."})

    ids = json.loads(second.cache.ids_path.read_text(encoding="utf-8"))
    assert set(ids) == {"news:a", "news:b"}
    assert ids["news:a"] == E.EmbeddingCache.key("First post text.")
    assert first.cache.ids_path == second.cache.ids_path


def test_the_id_map_write_leaves_no_temp_file(settings):
    embedder = _embed(settings, {"news:a": "First post text."})
    leftovers = [p.name for p in embedder.cache.ids_path.parent.glob("*.tmp")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# resolving posts to rows -- no database
# ---------------------------------------------------------------------------
def _resolve(posts, keys, ids, rows):
    from app.etl.scored_loader import _resolve_vector_rows

    return _resolve_vector_rows(posts, keys, ids, rows, E.EmbeddingCache.key)


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
    ids = {"p:absent": "gone", "p:range": "k1", "p:ok": "k0"}
    posts = [("p:absent", "x"), ("p:range", "y"), ("p:ok", "z")]

    # Two rows in `keys` but a matrix of one row: k1 points past the end.
    matches, misses = _resolve(posts, keys, ids, 1)
    assert matches == [("p:ok", 0)]
    assert misses == {"id_map_key_absent": 1, "id_map_row_out_of_range": 1}


def test_a_stale_id_entry_does_not_fall_back_to_the_text_key():
    text = "Some text."
    keys = {E.EmbeddingCache.key(text): 0}
    matches, misses = _resolve([("p:1", text)], keys, {"p:1": "gone"}, 1)
    assert matches == [] and misses == {"id_map_key_absent": 1}


def test_a_post_missing_from_the_id_map_falls_back_to_its_text_key():
    text = "Some text."
    keys = {E.EmbeddingCache.key(text): 0}
    matches, misses = _resolve([("p:1", text)], keys, {"p:other": "k"}, 1)
    assert matches == [("p:1", 0)] and misses == {}


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
    ids["reddit:t3_long"] = "0" * 32
    embedder.cache.ids_path.write_text(json.dumps(ids), encoding="utf-8")

    with caplog.at_level("WARNING"), sync_session() as session:
        load_embeddings(session, settings.data_dir, project)
    assert _loaded_ids(project) == {"reddit:t3_short"}
    assert "id_map_key_absent=1" in caplog.text


def test_embedding_matrix_rows_are_untouched_by_linking(settings):
    embedder = _embed(settings, {"news:a": "First post text."})
    before = np.load(embedder.cache.matrix_path)
    embedder.cache.link_ids({"news:b": "First post text."})
    assert np.array_equal(np.load(embedder.cache.matrix_path), before)
