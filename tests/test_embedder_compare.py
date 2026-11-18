"""The embedder comparison command, with a fake encoder and the demo corpus.

The fake maps every text to one of three tight groups, so the real cluster stage
finds three narratives without any weights being loaded.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from modeling import config as C
from modeling.cli import app
from modeling.eval import embedder_compare as X
from modeling.text import embed as E

DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATIVE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WIDE = "BAAI/bge-base-en-v1.5"


class GroupedEncoder:
    device = "cpu"

    def __init__(self, dim: int):
        self.dim = dim
        self.max_seq_length = 0

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.zeros(self.dim, dtype=np.float32)
            vector[digest[0] % 3] = 1.0
            vector += np.frombuffer(digest[1:9], dtype=np.uint8).astype(np.float32).repeat(
                self.dim // 8
            )[: self.dim] * 1e-4
            rows.append(vector)
        return E.l2_normalize(np.array(rows))


@pytest.fixture
def env(tmp_path, monkeypatch):
    C.get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    body = C.models_config()
    small = {**body, "cluster": {**body["cluster"], "min_cluster_size": 5, "min_samples": 2}}
    monkeypatch.setattr(C, "models_config", lambda: small)
    C.clear_language_cache()
    dims = {DEFAULT: 384, ALTERNATIVE: 384, WIDE: 768}
    loaded: list[str] = []

    def fake(name, device, local_only):
        loaded.append(name)
        return GroupedEncoder(dims[name])

    monkeypatch.setattr(E, "_load_sentence_transformer", fake)
    settings = C.get_settings()
    settings.ensure_dirs()
    yield settings, tmp_path, loaded
    C.get_settings.cache_clear()


def test_alternative_is_clustered_and_written_only_under_the_output_directory(env):
    settings, tmp_path, loaded = env
    out = tmp_path / "compare"
    shared = settings.scored_dir.parent / "scored_demo"
    result = X.compare_embedder(ALTERNATIVE, demo=True, out_dir=out, settings=settings)

    assert result.clustered
    assert result.alternative_checks["embed_model"] == ALTERNATIVE
    assert result.alternative_checks["embedding_dim"] == 384
    assert result.alternative_checks["n_clusters"] >= 2
    for name in ("clustering.json", "ablation.md", "ablation.json", "comparison.md"):
        assert (out / name).exists(), name
    assert (out / "scored" / "narratives").exists()
    assert (out / "scored" / "manifest.json").exists()
    # the shared stores and the shared manifest were not created or touched
    assert not shared.exists()
    assert not settings.scored_manifest_path.exists()
    assert not (settings.eval_dir / "ablation").exists()
    payload = json.loads((out / "clustering.json").read_text())
    assert payload["alternative"]["embed_model"] == ALTERNATIVE
    text = (out / "comparison.md").read_text()
    assert ALTERNATIVE in text and DEFAULT in text and "Ablation, alternative" in text


def test_default_output_path_is_separate_from_the_ablation_directory(env):
    settings, _, _ = env
    path = X.default_out_dir(ALTERNATIVE, settings)
    assert path.parent == settings.eval_dir / "embedder"
    assert "ablation" not in path.parts


def test_a_wrong_width_model_fails_before_anything_is_written(env):
    settings, tmp_path, _ = env
    out = tmp_path / "compare"
    with pytest.raises(E.EmbeddingDimensionMismatch):
        X.compare_embedder(WIDE, demo=True, out_dir=out, settings=settings)
    assert not out.exists()


def test_the_configured_model_is_not_a_comparison(env):
    settings, tmp_path, _ = env
    with pytest.raises(ValueError, match="already the configured"):
        X.compare_embedder(DEFAULT, demo=True, out_dir=tmp_path / "c", settings=settings)


@pytest.mark.parametrize("where", ["scored", "scored_demo", "eval/ablation", "scored/inside"])
def test_output_inside_or_over_a_shared_directory_is_refused(env, where):
    settings, _, _ = env
    base = {
        "scored": settings.scored_dir,
        "scored_demo": settings.scored_dir.parent / "scored_demo",
        "eval/ablation": settings.eval_dir / "ablation",
        "scored/inside": settings.scored_dir / "inside",
    }[where]
    with pytest.raises(ValueError, match="overlaps"):
        X.compare_embedder(ALTERNATIVE, demo=True, out_dir=base, settings=settings)


def test_cli_runs_the_comparison_and_reports_the_files(env):
    _, tmp_path, _ = env
    out = tmp_path / "cli"
    result = CliRunner().invoke(
        app, ["compare-embedder", "--model", ALTERNATIVE, "--demo", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "comparison.md" in result.output
    assert (out / "comparison.md").exists()


def test_cli_exits_nonzero_with_the_reason_for_a_wrong_width(env):
    _, tmp_path, _ = env
    result = CliRunner().invoke(
        app, ["compare-embedder", "--model", WIDE, "--demo", "--out", str(tmp_path / "w")]
    )
    assert result.exit_code == 1
    assert "768-dim" in result.output


def _membership(pairs):
    import pandas as pd

    return pd.DataFrame(pairs, columns=["record_id", "narrative_id"])


def test_language_purity_per_cluster_weighted_mean_and_mixed_count():
    lang = {f"a{i}": "hi" for i in range(9)} | {"a9": "en"}
    lang |= {f"b{i}": "en" for i in range(4)} | {f"c{i}": "hi" for i in range(2)}
    lang |= {"c2": "en", "c3": "en"}
    pairs = [(f"a{i}", "n-a") for i in range(10)]
    pairs += [(f"b{i}", "n-b") for i in range(4)]
    pairs += [(f"c{i}", "n-c") for i in range(4)]
    purity = X.language_purity(_membership(pairs), lang)

    by_id = {row["narrative_id"]: row for row in purity["clusters"]}
    assert by_id["n-a"]["purity"] == 0.9 and by_id["n-a"]["majority_lang"] == "hi"
    assert by_id["n-b"]["purity"] == 1.0
    assert by_id["n-c"]["purity"] == 0.5
    assert purity["mixed_clusters"] == 1
    assert purity["weighted_mean_purity"] == pytest.approx((9 + 4 + 2) / 18, abs=1e-4)
    assert purity["n_clusters"] == 3


def test_language_purity_counts_members_without_a_language_and_never_hides_them():
    lang = {"a": "hi", "b": "hi", "c": "und", "d": None}
    pairs = [("a", "n1"), ("b", "n1"), ("c", "n1"), ("d", "n2"), ("zz", "n2")]
    purity = X.language_purity(_membership(pairs), lang)

    assert purity["lang_unknown"] == 3
    rows = {row["narrative_id"]: row for row in purity["clusters"]}
    assert rows["n1"]["purity"] == 1.0 and rows["n1"]["lang_unknown"] == 1
    assert rows["n2"]["purity"] is None
    assert purity["weighted_mean_purity"] == 1.0
    assert X.language_purity(_membership([]), lang)["status"] == "no_clusters"


def test_sweep_runs_both_embedders_at_each_size_and_reports_purity(env):
    settings, tmp_path, loaded = env
    out = tmp_path / "sweep"
    result = X.compare_embedder(
        ALTERNATIVE, demo=True, out_dir=out, min_cluster_sizes=[4, 6], settings=settings
    )

    assert [(r["embedder"], r["min_cluster_size"]) for r in result.sweep] == [
        ("baseline", 4), ("alternative", 4), ("baseline", 6), ("alternative", 6),
    ]
    assert all("weighted_mean_purity" in r["purity"] or r["purity"]["status"] for r in result.sweep)
    payload = json.loads((out / "clustering.json").read_text())
    assert len(payload["sweep"]) == 4
    assert "language_purity" in payload
    text = (out / "comparison.md").read_text()
    assert "## Language purity" in text and "## Sweep over min_cluster_size" in text
    assert (out / "sweep" / "alternative-mcs4").exists()


def test_cli_accepts_a_sweep_and_rejects_a_bad_size(env):
    import re

    _, tmp_path, _ = env
    ok = CliRunner().invoke(
        app,
        ["compare-embedder", "--model", ALTERNATIVE, "--demo", "--out", str(tmp_path / "s"),
         "--min-cluster-size", "4,6"],
    )
    assert ok.exit_code == 0, ok.output
    bad = CliRunner().invoke(
        app,
        ["compare-embedder", "--model", ALTERNATIVE, "--demo", "--out", str(tmp_path / "t"),
         "--min-cluster-size", "four"],
    )
    assert bad.exit_code != 0
    assert "whole numbers" in re.sub(r"\x1b\[[0-9;]*m", "", bad.output)
