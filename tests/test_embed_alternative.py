"""The config-selected embedder and its dimension check.

No weights are loaded. `_load_sentence_transformer` is replaced by a fake that
reports a chosen vector width, which is all the load-time check needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling import config as C
from modeling.text import embed as E

DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FakeEncoder:
    """Deterministic stand-in: a hash-seeded vector of a chosen width."""

    device = "cpu"

    def __init__(self, dim: int):
        self.dim = dim
        self.max_seq_length = 0
        self.encoded: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            rows.append(rng.normal(size=self.dim))
        out = np.array(rows, dtype=np.float32)
        if kwargs.get("normalize_embeddings"):
            out = E.l2_normalize(out)
        return out


@pytest.fixture
def settings(tmp_path, monkeypatch):
    C.get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    yield C.get_settings()
    C.get_settings.cache_clear()


def _install(monkeypatch, dims: dict[str, int]):
    loaded: list[str] = []

    def fake(name, device, local_only):
        loaded.append(name)
        return FakeEncoder(dims[name])

    monkeypatch.setattr(E, "_load_sentence_transformer", fake)
    return loaded


def test_default_model_is_unchanged_and_the_alternative_is_named():
    config = C.module_config("embed")
    assert config["model_name"] == DEFAULT
    assert MULTILINGUAL in config["alternatives"]
    assert config["vector_dim"] == 384


def test_default_loads_and_reports_384(settings, monkeypatch):
    _install(monkeypatch, {DEFAULT: 384})
    embedder = E.Embedder(settings)
    assert embedder.model_name == DEFAULT
    assert embedder.load() is True
    assert embedder.dim == 384


def test_named_alternative_loads_when_it_is_384(settings, monkeypatch):
    loaded = _install(monkeypatch, {MULTILINGUAL: 384})
    embedder = E.Embedder(settings, model_name=MULTILINGUAL)
    assert embedder.load() is True
    assert loaded == [MULTILINGUAL]
    assert embedder.dim == 384


def test_config_selects_the_model(settings, monkeypatch):
    body = C.models_config()
    patched = {**body, "embed": {**body["embed"], "model_name": MULTILINGUAL}}
    monkeypatch.setattr(C, "models_config", lambda: patched)
    assert E.Embedder(settings).model_name == MULTILINGUAL


def test_wrong_width_fails_loudly_at_load(settings, monkeypatch):
    """bge-base is 768-dim. It must not reach the 384-wide vector column."""
    _install(monkeypatch, {"BAAI/bge-base-en-v1.5": 768})
    embedder = E.Embedder(settings, model_name="BAAI/bge-base-en-v1.5")
    with pytest.raises(E.EmbeddingDimensionMismatch, match="768-dim.*384-dim"):
        embedder.load()
    # a second call raises again instead of reporting a half-loaded model
    with pytest.raises(E.EmbeddingDimensionMismatch):
        embedder.load()
    with pytest.raises(E.EmbeddingDimensionMismatch):
        embedder.embed_texts(["anything"])


def test_wrong_width_writes_nothing_to_the_cache(settings, monkeypatch):
    _install(monkeypatch, {"BAAI/bge-base-en-v1.5": 768})
    embedder = E.Embedder(settings, model_name="BAAI/bge-base-en-v1.5")
    with pytest.raises(E.EmbeddingDimensionMismatch):
        embedder.embed_records(pd.DataFrame([{"id": "a", "text": "a long enough sentence"}]))
    assert not list(settings.embeddings_dir.glob("*.npy"))


def test_missing_vector_dim_is_an_error_not_a_pass(settings, monkeypatch):
    body = C.models_config()
    without = {**body, "embed": {k: v for k, v in body["embed"].items() if k != "vector_dim"}}
    monkeypatch.setattr(C, "models_config", lambda: without)
    _install(monkeypatch, {DEFAULT: 384})
    with pytest.raises(E.EmbeddingDimensionMismatch, match="vector_dim"):
        E.Embedder(settings).load()


def test_check_dimension_accepts_a_match_and_names_both_widths_otherwise():
    E.check_dimension("m", 384, 384)
    with pytest.raises(E.EmbeddingDimensionMismatch, match="m produces 768"):
        E.check_dimension("m", 768, 384)


def test_alternative_embeds_hindi_to_unit_vectors_with_its_own_cache(settings, monkeypatch):
    _install(monkeypatch, {DEFAULT: 384, MULTILINGUAL: 384})
    frame = pd.DataFrame(
        [
            {"id": "en", "text": "the vaccine claim resurfaced again this week"},
            {"id": "hi", "text": "टीके के बारे में दावा इस सप्ताह फिर सामने आया"},
        ]
    )
    default = E.Embedder(settings).embed_records(frame)
    alternative = E.Embedder(settings, model_name=MULTILINGUAL).embed_records(frame)

    assert alternative.model_name == MULTILINGUAL
    assert alternative.dim == 384 and alternative.vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(alternative.vectors, axis=1), 1.0, atol=1e-5)
    # one cache file per model, so the two never mix
    assert len(list(settings.embeddings_dir.glob("*.npy"))) == 2
    assert default.model_name == DEFAULT
