"""The embedding claim dedup pass: links, counts by reason, and the hand check."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest

from modeling.eval import claim_dedup as cd
from modeling.eval.factcheck_eval import (
    CLAIM_COLLAPSED_EMBEDDING,
    EvalInputError,
    ManifestMismatch,
    build_claim_universe,
    build_manifest,
    freeze_manifest_from_data,
    load_claim_universe,
    read_manifest,
    verify_manifest,
)
from tests.test_factcheck_manifest import _write_factchecks

BASE = 0x0123456789ABCDEF
CONFIG = cd.ClaimDedupConfig(
    enabled=True, model_name="fake-model", min_similarity=0.8, sample_size=10
)


def distinct(index: int) -> int:
    """A 64-bit value far enough from the others that simhash never links them."""
    return (0x9E3779B97F4A7C15 * (index + 1)) & ((1 << 64) - 1)


def row(record_id: str, claim: str, *, simhash: int | None = None, label: str = "false"):
    if simhash is None:
        simhash = distinct(len(record_id) + ord(record_id[0]))
    return {
        "record_id": record_id,
        "claim": claim,
        "claim_simhash": simhash,
        "label": label,
    }


def unit(similarity: float) -> list[float]:
    """A unit vector whose cosine with ``[1, 0]`` is ``similarity``."""
    return [similarity, float(np.sqrt(max(0.0, 1.0 - similarity**2)))]


class FakeEmbedder:
    """Vectors keyed by claim text. No model, no network, no cache."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = {
            text: np.asarray(vector, dtype=np.float32) for text, vector in vectors.items()
        }
        self.seen: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.seen.append(list(texts))
        return np.stack([self.vectors[text] for text in texts])


class FailingEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("model fell over")


def ids(result: cd.ClaimDedupResult) -> list[str]:
    return [str(item["record_id"]) for item in result.kept]


# ---------------------------------------------------------------------------
# The two passes
# ---------------------------------------------------------------------------
def test_a_reworded_claim_outside_the_simhash_radius_is_linked():
    rows = [
        row("a", "video shows flooding in the city", simhash=distinct(0)),
        row("b", "video shows the city flooded", simhash=distinct(1)),
    ]
    embedder = FakeEmbedder({rows[0]["claim"]: [1.0, 0.0], rows[1]["claim"]: unit(0.9)})

    result = cd.run_claim_dedup(rows, config=CONFIG, embedder=embedder)

    assert ids(result) == ["a"]
    assert result.counts["simhash"] == 0
    assert result.counts["embedding"] == 1
    assert result.clusters == {"a": "a", "b": "a"}
    (link,) = result.links
    assert link.reason == cd.DedupPass.EMBEDDING.value
    assert link.similarity == pytest.approx(0.9, abs=1e-6)


def test_claims_in_different_languages_link_when_their_embeddings_are_close():
    rows = [
        row("a", "video shows flooding in the city", simhash=distinct(0)),
        row("b", "वीडियो में शहर में बाढ़ दिखाई दे रही है", simhash=distinct(1)),
    ]
    embedder = FakeEmbedder({rows[0]["claim"]: [1.0, 0.0], rows[1]["claim"]: unit(0.93)})

    result = cd.run_claim_dedup(rows, config=CONFIG, embedder=embedder)

    assert ids(result) == ["a"]
    assert result.counts["embedding"] == 1
    assert "claim" not in result.links[0].as_dict()


def test_a_pair_below_the_threshold_is_kept_and_the_zero_is_measured():
    rows = [
        row("a", "video shows flooding in the city", simhash=distinct(0)),
        row("b", "a different claim about a different flood", simhash=distinct(1)),
    ]
    embedder = FakeEmbedder({rows[0]["claim"]: [1.0, 0.0], rows[1]["claim"]: unit(0.7)})

    result = cd.run_claim_dedup(rows, config=CONFIG, embedder=embedder)

    assert ids(result) == ["a", "b"]
    assert result.counts["embedding"] == 0
    outcome = result.outcome(cd.DedupPass.EMBEDDING)
    assert outcome is not None and outcome.status == "ok" and outcome.collapsed == 0


def test_a_moved_group_attributes_the_representative_to_embedding_and_its_followers_to_simhash():
    # a and b share a simhash; c and d share another; the two groups are linked by
    # embedding. Simhash removes b and d (2). Embedding removes c (1), the only
    # row that would have been kept without the second relation.
    rows = [
        row("a", "flood video from the city", simhash=BASE),
        row("b", "flood video from the city again", simhash=BASE),
        row("c", "city flooding video", simhash=BASE ^ ((1 << 64) - 1)),
        row("d", "city flooding video shared widely", simhash=BASE ^ ((1 << 64) - 1)),
    ]
    embedder = FakeEmbedder(
        {rows[0]["claim"]: [1.0, 0.0], rows[2]["claim"]: unit(0.95)}
    )

    result = cd.run_claim_dedup(rows, config=CONFIG, embedder=embedder)

    assert ids(result) == ["a"]
    assert result.counts["simhash"] == 2
    assert result.counts["embedding"] == 1
    reasons = {link.record_id: link.reason for link in result.links}
    assert reasons == {
        "b": cd.DedupPass.SIMHASH.value,
        "c": cd.DedupPass.EMBEDDING.value,
        "d": cd.DedupPass.SIMHASH.value,
    }
    assert sum(result.counts[name] for name in ("simhash", "embedding")) == len(result.links)


def test_a_chain_links_and_the_representative_is_the_smallest_record_id():
    rows = [
        row("n", "claim one", simhash=distinct(0)),
        row("m", "claim two", simhash=distinct(1)),
        row("z", "claim three", simhash=distinct(2)),
    ]
    embedder = FakeEmbedder(
        {
            rows[0]["claim"]: unit(0.82),
            rows[1]["claim"]: [1.0, 0.0],
            rows[2]["claim"]: [0.82, -0.572363],
        }
    )

    result = cd.run_claim_dedup(rows, config=CONFIG, embedder=embedder)

    assert ids(result) == ["m"]
    assert result.counts["embedding"] == 2
    assert result.clusters == {"m": "m", "n": "m", "z": "m"}


def test_result_does_not_depend_on_input_order():
    rows = [
        row("a", "claim one", simhash=distinct(0)),
        row("b", "claim two", simhash=distinct(1)),
        row("c", "claim three", simhash=distinct(2)),
        row("d", "claim four", simhash=distinct(3)),
    ]
    vectors = {
        rows[0]["claim"]: [1.0, 0.0],
        rows[1]["claim"]: unit(0.9),
        rows[2]["claim"]: unit(0.85),
        rows[3]["claim"]: [0.0, 1.0],
    }
    expected = cd.run_claim_dedup(rows, config=CONFIG, embedder=FakeEmbedder(vectors))
    rng = random.Random(11)
    for _ in range(10):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        result = cd.run_claim_dedup(shuffled, config=CONFIG, embedder=FakeEmbedder(vectors))
        assert result.kept == expected.kept
        assert result.links == expected.links
        assert result.clusters == expected.clusters
        assert result.counts == expected.counts


def test_a_disabled_pass_is_not_measured_and_does_not_load_a_model():
    rows = [row("a", "claim one", simhash=distinct(0)), row("b", "claim two", simhash=distinct(1))]
    embedder = FakeEmbedder({rows[0]["claim"]: [1.0, 0.0], rows[1]["claim"]: [1.0, 0.0]})

    result = cd.run_claim_dedup(
        rows, config=cd.config_without_embedding(CONFIG), embedder=embedder
    )

    outcome = result.outcome(cd.DedupPass.EMBEDDING)
    assert outcome is not None
    assert outcome.status == "disabled"
    assert outcome.collapsed is None
    assert outcome.reason == cd.EmbeddingReason.DISABLED.value
    assert "embedding" not in result.counts
    assert ids(result) == ["a", "b"]
    assert embedder.seen == []


def test_a_missing_dependency_is_a_reason_code_not_an_exception(monkeypatch):
    monkeypatch.setattr(
        cd,
        "default_embedder",
        lambda model, settings=None: (None, cd.EmbeddingReason.DEPENDENCY_MISSING.value),
    )
    rows = [row("a", "claim one", simhash=distinct(0)), row("b", "claim two", simhash=distinct(1))]

    result = cd.run_claim_dedup(rows, config=CONFIG)

    outcome = result.outcome(cd.DedupPass.EMBEDDING)
    assert outcome is not None
    assert outcome.status == "unavailable"
    assert outcome.collapsed is None
    assert outcome.reason == "sentence_transformers_not_installed"
    assert "embedding" not in result.counts
    assert ids(result) == ["a", "b"]


def test_an_encode_failure_is_an_unavailable_pass_with_a_reason_code():
    rows = [row("a", "claim one", simhash=distinct(0)), row("b", "claim two", simhash=distinct(1))]

    result = cd.run_claim_dedup(rows, config=CONFIG, embedder=FailingEmbedder())

    outcome = result.outcome(cd.DedupPass.EMBEDDING)
    assert outcome is not None
    assert outcome.status == "unavailable"
    assert outcome.collapsed is None
    assert outcome.reason == cd.EmbeddingReason.ENCODE_FAILED.value
    assert ids(result) == ["a", "b"]


def test_a_row_with_no_claim_text_is_counted_and_left_to_the_simhash_pass():
    rows = [
        row("a", "", simhash=distinct(0)),
        row("b", "claim two", simhash=distinct(1)),
        row("c", "claim three", simhash=distinct(2)),
    ]
    embedder = FakeEmbedder({rows[1]["claim"]: [1.0, 0.0], rows[2]["claim"]: unit(0.9)})

    result = cd.run_claim_dedup(rows, config=CONFIG, embedder=embedder)

    assert result.outcome(cd.DedupPass.EMBEDDING).skipped == {"no_claim_text": 1}
    assert result.counts["embedding"] == 1
    assert sorted(ids(result)) == ["a", "b"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "block, embed, match",
    [
        ({}, {"alternatives": ["m"]}, "min_similarity"),
        ({"min_similarity": 0.5}, {"alternatives": ["m"]}, "model_name"),
        ({"min_similarity": 0.0}, {"claim_dedup_model": "m", "alternatives": ["m"]}, "above 0"),
        ({"min_similarity": 1.5}, {"claim_dedup_model": "m", "alternatives": ["m"]}, "at most 1"),
        (
            {"min_similarity": 0.5},
            {"claim_dedup_model": "other", "alternatives": ["m"]},
            "not listed",
        ),
        (
            {"min_similarity": 0.5, "sample_size": 0},
            {"claim_dedup_model": "m", "alternatives": ["m"]},
            "sample_size",
        ),
    ],
)
def test_config_refuses_a_guessed_value(block, embed, match):
    with pytest.raises(cd.ClaimDedupConfigError, match=match):
        cd.parse_claim_dedup_config(block, embed)


def test_a_disabled_pass_needs_no_threshold_or_model():
    config = cd.parse_claim_dedup_config({"enabled": False}, {})
    assert not config.enabled and config.min_similarity is None


def test_the_repo_config_resolves_and_names_the_multilingual_alternative():
    config = cd.claim_dedup_config()
    assert config.enabled
    assert config.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert config.min_similarity is not None and 0.0 < config.min_similarity <= 1.0
    assert config.sample_size >= 1


# ---------------------------------------------------------------------------
# Artifact, staleness and the frozen set
# ---------------------------------------------------------------------------
def write_links(
    data_dir: Path,
    clusters: dict[str, str],
    *,
    source: Path | None = None,
    sample: Path | None = None,
    passes: dict | None = None,
    counts: dict | None = None,
) -> Path:
    payload = {
        "schema_version": cd.LINKS_SCHEMA_VERSION,
        "generated_at": "2026-11-18T00:00:00+00:00",
        "rows": len(clusters),
        "passes": passes
        or {
            "simhash": {"name": "simhash", "status": "ok", "considered": 2, "collapsed": 0},
            "embedding": {
                "name": "embedding",
                "status": "ok",
                "considered": 2,
                "collapsed": 1,
                "threshold": 0.85,
                "model_name": "fake-model",
            },
        },
        "counts": counts or {"simhash": 0, "embedding": 1},
        "clusters": clusters,
        "links": [
            {"record_id": record_id, "kept_record_id": representative, "reason": "embedding"}
            for record_id, representative in clusters.items()
            if record_id != representative
        ],
    }
    if source is not None:
        payload["source"] = {"file": source.name, "sha256": cd.sha256_file(source)}
    if sample is not None:
        payload["sample"] = str(sample)
    path = cd.claim_links_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_links_artifact_round_trips_and_keeps_no_claim_text(tmp_path):
    result = cd.run_claim_dedup(
        [
            row("a", "claim one", simhash=distinct(0)),
            row("b", "claim two", simhash=distinct(1)),
        ],
        config=CONFIG,
        embedder=FakeEmbedder({"claim one": [1.0, 0.0], "claim two": unit(0.9)}),
    )
    source = tmp_path / "factchecks.parquet"
    source.write_bytes(b"parquet bytes")
    path = cd.write_claim_links(result, tmp_path / "links.json", source=source)

    payload = cd.read_claim_links(path)
    assert payload is not None
    assert payload["schema_version"] == cd.LINKS_SCHEMA_VERSION
    assert payload["clusters"] == {"a": "a", "b": "a"}
    assert payload["counts"]["embedding"] == 1
    assert payload["source"]["sha256"] == cd.sha256_file(source)
    assert "claim one" not in path.read_text(encoding="utf-8")

    cd.ensure_source_matches(payload, source)
    source.write_bytes(b"changed")
    with pytest.raises(cd.ClaimLinksStale, match="different factchecks file"):
        cd.ensure_source_matches(payload, source)


def test_a_malformed_artifact_is_refused(tmp_path):
    path = tmp_path / "links.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(cd.ClaimLinksError, match="schema_version"):
        cd.read_claim_links(path)


@pytest.mark.parametrize(
    "clusters, match",
    [
        ({"a": "a"}, "does not describe"),
        ({"a": "a", "b": "a", "d": "d"}, "does not describe"),
        ({"a": "b", "b": "a", "c": "c"}, "not canonical"),
        ({"a": "a", "b": "a", "c": "x"}, "not a record"),
    ],
)
def test_the_eval_refuses_a_cluster_map_that_does_not_cover_the_rows(clusters, match):
    with pytest.raises(cd.ClaimLinksStale, match=match):
        cd.clusters_for_ids(clusters, {"a", "b", "c"})


def test_final_rows_mark_only_the_rows_the_embedding_pass_removed():
    kept = [{"record_id": "a"}, {"record_id": "c"}]
    collapsed = [("b", "a", 0)]
    final, embedding = cd.final_rows_and_embedding_collapses(
        kept, collapsed, {"a": "a", "b": "a", "c": "a"}
    )
    assert [item["record_id"] for item in final] == ["a"]
    assert embedding == {"c": "a"}


def test_universe_gives_an_embedding_collapsed_row_its_own_reason():
    universe = build_claim_universe(
        [{"record_id": "a", "claim": "x", "label": "false"}],
        [],
        collapsed_embedding={"b": "a"},
    )
    assert universe.eval_ids == {"a"}
    assert universe.reason_for_exclusion("b") == CLAIM_COLLAPSED_EMBEDDING


def test_a_manifest_row_that_became_an_embedding_duplicate_stops():
    manifest = build_manifest([{"record_id": "b", "claim": "x", "label": "false", "url": "u"}])
    with pytest.raises(ManifestMismatch, match="embedding claim pass"):
        verify_manifest(
            manifest,
            [{"record_id": "a", "claim": "x", "label": "false", "url": "u"}],
            [],
            collapsed_embedding={"b": "a"},
        )


def test_load_applies_links_and_reports_both_passes(tmp_path):
    data = tmp_path / "data"
    _write_factchecks(
        data,
        [
            {"record_id": "a", "claim": "claim one", "simhash": distinct(0)},
            {"record_id": "b", "claim": "claim two", "simhash": distinct(1)},
        ],
    )
    sample = data / "factchecks" / cd.HAND_CHECK_FILENAME
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,is_same_claim\n",
        encoding="utf-8",
    )
    write_links(data, {"a": "a", "b": "a"}, sample=sample)

    universe = load_claim_universe(data, tmp_path / "no-manifest.csv")

    assert universe.eval_ids == {"a"}
    assert universe.reason_for_exclusion("b") == CLAIM_COLLAPSED_EMBEDDING
    assert universe.dedup_counts["simhash"] == 0
    assert universe.dedup_counts["embedding"] == 1
    assert universe.claim_dedup["passes"]["embedding"]["collapsed"] == 1
    assert universe.claim_dedup["precision"]["status"] == "not_measured"
    assert universe.claim_dedup["precision"]["precision"] is None


def test_load_refuses_a_links_artifact_for_other_rows(tmp_path):
    data = tmp_path / "data"
    _write_factchecks(data, [{"record_id": "a", "claim": "claim one", "simhash": distinct(0)}])
    write_links(data, {"a": "a", "gone": "a"})

    with pytest.raises(EvalInputError, match="does not describe"):
        load_claim_universe(data, tmp_path / "no-manifest.csv")


def test_freezing_uses_the_final_rows_when_links_exist(tmp_path):
    data = tmp_path / "data"
    _write_factchecks(
        data,
        [
            {"record_id": "a", "claim": "claim one", "simhash": distinct(0)},
            {"record_id": "b", "claim": "claim two", "simhash": distinct(1)},
        ],
    )
    write_links(data, {"a": "a", "b": "a"})
    manifest = tmp_path / "manifest.csv"

    _, count = freeze_manifest_from_data(data, manifest)

    assert count == 1
    assert [item["record_id"] for item in read_manifest(manifest)] == ["a"]


# ---------------------------------------------------------------------------
# Hand-check sample and precision
# ---------------------------------------------------------------------------
def sample_result() -> cd.ClaimDedupResult:
    links = [
        cd.ClaimLink(
            record_id=f"r{index:02d}",
            kept_record_id="keep",
            reason=cd.DedupPass.EMBEDDING.value,
            similarity=0.80 + index / 100,
        )
        for index in range(10)
    ]
    return cd.ClaimDedupResult(
        kept=[{"record_id": "keep"}],
        links=links,
        clusters={"keep": "keep", **{link.record_id: "keep" for link in links}},
        passes={},
        counts={"embedding": len(links)},
    )


def test_the_hand_check_sample_spans_the_similarity_range(tmp_path):
    rows = [{"record_id": "keep", "claim": "kept claim", "label": "false"}]
    rows += [
        {"record_id": f"r{index:02d}", "claim": f"claim {index}", "label": "false"}
        for index in range(10)
    ]
    path = tmp_path / "sample.csv"

    cd.write_hand_check_sample(sample_result(), rows, path, size=4)

    with path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [item["similarity"] for item in written] == [
        "0.800000",
        "0.830000",
        "0.860000",
        "0.890000",
    ]
    assert all(item["is_same_claim"] == "" for item in written)
    assert written[0]["claim"] == "claim 0"
    assert written[0]["kept_claim"] == "kept claim"


def test_precision_with_no_labels_is_not_measured(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,is_same_claim\n"
        "a,b,0.9,false,x,false,y,\n",
        encoding="utf-8",
    )

    precision = cd.embedding_precision(path)

    assert precision.status == "not_measured"
    assert precision.precision is None
    assert precision.reason == cd.PrecisionReason.SAMPLE_NOT_LABELLED.value


def test_precision_counts_labelled_pairs_and_a_zero_is_measured(tmp_path):
    path = tmp_path / "sample.csv"
    header = "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,is_same_claim\n"
    path.write_text(
        header
        + "a,b,0.9,false,x,false,y,yes\n"
        + "c,d,0.88,false,x,false,y,no\n"
        + "e,f,0.86,false,x,false,y,\n",
        encoding="utf-8",
    )
    precision = cd.embedding_precision(path)
    # An older sample has no labeller_kind column. It reads as unknown and is not measured.
    assert precision.status == "provisional"
    assert precision.labeller_kind == "unknown"
    assert precision.reason == "labeller_not_human"
    assert precision.precision == pytest.approx(0.5)
    assert (precision.n_same, precision.n_labelled) == (1, 2)

    path.write_text(header + "a,b,0.9,false,x,false,y,no\n", encoding="utf-8")
    zero = cd.embedding_precision(path)
    assert zero.precision == 0.0
    assert zero.status == "provisional"


def test_an_unrecognised_hand_label_is_refused(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,is_same_claim\n"
        "a,b,0.9,false,x,false,y,maybe\n",
        encoding="utf-8",
    )
    with pytest.raises(cd.ClaimSampleError, match="use yes or no"):
        cd.embedding_precision(path)


def test_the_report_block_says_not_run_when_no_artifact_exists(tmp_path):
    block = cd.embedding_report(
        None,
        simhash_count=3,
        simhash_considered=10,
        reason=cd.EmbeddingReason.LINKS_ABSENT.value,
        sample_path=tmp_path / "absent.csv",
    )
    assert block["passes"]["simhash"]["collapsed"] == 3
    assert block["passes"]["embedding"]["collapsed"] is None
    assert block["passes"]["embedding"]["reason"] == "links_artifact_absent"
    assert block["precision"]["precision"] is None


def test_the_cli_writes_links_and_a_sample_for_a_simhash_only_run(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from modeling import cli
    from modeling.config import get_settings

    data = tmp_path / "data"
    _write_factchecks(
        data,
        [
            {"record_id": "a", "claim": "claim one", "simhash": BASE},
            {"record_id": "b", "claim": "claim one again", "simhash": BASE},
        ],
    )
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()
    factchecks = data / "factchecks" / "factchecks.parquet"
    before = factchecks.read_bytes()
    try:
        result = CliRunner().invoke(
            cli.app, ["dedup-claims", "--no-embedding", "--sample-size", "5"]
        )
    finally:
        get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    assert factchecks.read_bytes() == before
    payload = cd.read_claim_links(cd.claim_links_path(data))
    assert payload is not None
    assert payload["counts"]["simhash"] == 1
    assert payload["passes"]["embedding"]["collapsed"] is None
    assert payload["passes"]["embedding"]["reason"] == "disabled_in_config"
    assert cd.hand_check_sample_path(data).exists()
    assert "embedding: not measured" in result.output


NEW_HEADER = ",".join(cd.SAMPLE_COLUMNS) + "\n"


def _sample_row(record_id, kind, l1, l2, a, b, same):
    # record_id, kept_record_id, similarity, label, claim, kept_label, kept_claim,
    # labeller_kind, labeller_1, labeller_2, label_1, label_2, is_same_claim
    return f"{record_id},k,0.9,false,x,false,y,{kind},{l1},{l2},{a},{b},{same}\n"


def _write_sample(tmp_path, *rows):
    path = tmp_path / "sample.csv"
    path.write_text(NEW_HEADER + "".join(rows), encoding="utf-8")
    return path


def test_the_written_sample_has_blank_labeller_columns(tmp_path):
    path = tmp_path / "sample.csv"
    rows = [{"record_id": "keep", "claim": "kept claim", "label": "false"}]
    cd.write_hand_check_sample(sample_result(), rows, path, size=4)
    with path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    for column in ("labeller_kind", "labeller_1", "labeller_2", "label_1", "label_2"):
        assert column in cd.SAMPLE_COLUMNS
        assert all(item[column] == "" for item in written)


def test_the_kind_list_is_the_tier_4_list():
    from ingest import title_predicate_eval as tpe

    assert cd.LABELLER_KINDS is tpe.LABELLER_KINDS
    assert cd.MEASURED_KINDS is tpe.MEASURED_KINDS


def test_human_labels_are_measured_with_interval_and_agreement(tmp_path):
    path = _write_sample(
        tmp_path,
        _sample_row("a", "human", "ana", "raj", "yes", "yes", "yes"),
        _sample_row("b", "human", "ana", "raj", "no", "no", "no"),
        _sample_row("c", "human", "ana", "raj", "yes", "no", "yes"),
        _sample_row("d", "human", "ana", "raj", "yes", "yes", "yes"),
        _sample_row("e", "", "", "", "", "", ""),
    )
    result = cd.embedding_precision(path)
    assert result.status == "measured"
    assert result.reason is None
    assert result.labeller_kind == "human"
    assert result.precision == pytest.approx(0.75)
    assert 0.0 < result.ci_low < 0.75 < result.ci_high <= 1.0
    assert result.n_double_labelled == 4
    assert result.agreement == pytest.approx(0.75)
    assert result.kappa == pytest.approx(0.5)
    text = cd.describe_precision(result.as_dict())
    assert "labeller: human" in text
    assert "PROVISIONAL" not in text


@pytest.mark.parametrize("kind", ["model-unreviewed", "model-reviewed"])
def test_model_labels_are_provisional_not_measured(tmp_path, kind):
    path = _write_sample(tmp_path, _sample_row("a", kind, "m1", "m2", "yes", "yes", "yes"))
    result = cd.embedding_precision(path)
    assert result.status == "provisional"
    assert result.reason == cd.PrecisionReason.LABELLER_NOT_HUMAN.value
    assert result.labeller_kind == kind
    assert "not a measured precision" in cd.describe_precision(result.as_dict())


def test_model_unreviewed_text_names_the_labeller(tmp_path):
    path = _write_sample(
        tmp_path, _sample_row("a", "model-unreviewed", "m1", "m2", "yes", "yes", "yes")
    )
    text = cd.describe_precision(cd.embedding_precision(path).as_dict())
    assert "labeller: model, unreviewed by a person" in text
    assert "PROVISIONAL" in text
    assert "labeller_not_human" in text


def test_a_sample_with_mixed_kinds_is_not_measured(tmp_path):
    path = _write_sample(
        tmp_path,
        _sample_row("a", "human", "ana", "raj", "yes", "yes", "yes"),
        _sample_row("b", "model-unreviewed", "m1", "m2", "no", "no", "no"),
    )
    result = cd.embedding_precision(path)
    assert result.labeller_kind == "mixed"
    assert result.status == "provisional"
    assert result.reason == "labeller_not_human"


def test_kappa_is_none_without_double_labels(tmp_path):
    path = _write_sample(tmp_path, _sample_row("a", "human", "ana", "", "yes", "", "yes"))
    result = cd.embedding_precision(path)
    assert result.status == "measured"
    assert result.kappa is None
    assert result.agreement is None
    assert result.n_double_labelled == 0


@pytest.mark.parametrize(
    "row, message",
    [
        (_sample_row("a", "", "ana", "raj", "yes", "yes", "yes"), "labeller_kind"),
        (_sample_row("a", "robot", "ana", "raj", "yes", "yes", "yes"), "labeller_kind"),
        (_sample_row("a", "human", "ana", "raj", "yes", "yes", "maybe"), "use yes or no"),
        (_sample_row("a", "human", "ana", "raj", "maybe", "yes", "yes"), "label_1"),
        (_sample_row("a", "human", "Ana", "ana", "yes", "yes", "yes"), "same labeller"),
    ],
)
def test_an_invalid_filled_row_is_refused(tmp_path, row, message):
    path = _write_sample(tmp_path, row)
    with pytest.raises(cd.ClaimSampleError, match=message):
        cd.embedding_precision(path)

