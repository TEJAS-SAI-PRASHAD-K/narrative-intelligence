"""Tier 4 stays out of the primary evaluation unless the config puts it in."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ingest.config import CONFIG_DIR, Settings
from ingest.factcheck import FactCheck, collapse_duplicate_claims, write_factchecks
from ingest.title_predicate import pattern_set_hash
from modeling.eval.factcheck_eval import (
    CLAIM_TIER4_EXCLUDED,
    ClaimUniverse,
    EvalInputError,
    ManifestMismatch,
    build_claim_universe,
    build_manifest,
    freeze_manifest_from_data,
    load_claim_universe,
    publisher_shares_from_data,
    render_markdown,
    title_predicate_settings,
)
from tests._factcheck_eval_support import PUBLISHERS, pair, r, run


def real_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "factcheck.yaml").read_text(encoding="utf-8"))


# --- claim collapse keeps tier 4 apart ---------------------------------------


def row(record_id, tier, simhash):
    return {"record_id": record_id, "tier": tier, "claim_simhash": simhash, "label": "false"}


def test_a_tier_4_row_never_absorbs_a_structured_verdict():
    rows = [row("a-headline", "title_predicate", 0b1010), row("b-stated", "claimreview", 0b1010)]
    result = collapse_duplicate_claims(rows)
    assert [r["record_id"] for r in result.kept] == ["a-headline", "b-stated"]
    assert result.collapsed == []


def test_a_tier_4_row_never_bridges_two_structured_verdicts():
    far = 0b111111  # 6 bits apart from 0
    rows = [
        row("a", "claimreview", 1 << 10),
        row("b", "claimreview", (1 << 10) | far),
        row("m", "title_predicate", (1 << 10) | 0b11),  # near both
    ]
    result = collapse_duplicate_claims(rows)
    assert [r["record_id"] for r in result.kept] == ["a", "b", "m"]
    merged = collapse_duplicate_claims(rows, merge_tier4=True)
    assert [r["record_id"] for r in merged.kept] == ["a", "b"]
    assert merged.collapsed == [("m", "a", 2)]


def test_with_the_switch_on_a_headline_collapses_into_the_stated_verdict():
    rows = [row("a-headline", "title_predicate", 0b1010), row("b-stated", "claimreview", 0b1010)]
    result = collapse_duplicate_claims(rows, merge_tier4=True)
    assert [r["record_id"] for r in result.kept] == ["b-stated"]
    assert result.collapsed == [("a-headline", "b-stated", 0)]
    assert result.counts["duplicate_claim"] == 1


def test_tier_4_rows_still_collapse_among_themselves():
    rows = [row("a", "title_predicate", 0b1010), row("b", "title_predicate", 0b1010)]
    result = collapse_duplicate_claims(rows)
    assert [r["record_id"] for r in result.kept] == ["a"]
    assert result.collapsed == [("b", "a", 0)]
    assert result.counts["duplicate_claim"] == 1


def test_collapse_without_tier_4_is_unchanged():
    rows = [row("b", "claimreview", 0b1010), row("a", "dom_block", 0b1010)]
    for merge in (False, True):
        result = collapse_duplicate_claims(rows, merge_tier4=merge)
        assert [r["record_id"] for r in result.kept] == ["a"]


def test_the_embedding_pass_skips_tier_4_rows():
    import numpy as np

    from modeling.eval import claim_dedup as cd

    class Embedder:
        def encode(self, texts, **_):
            return np.ones((len(texts), 4), dtype="float32")

    rows = [
        {**row("a-headline", "title_predicate", 1), "claim": "same words"},
        {**row("b-stated", "claimreview", 1 << 40), "claim": "same words"},
    ]
    result = cd.run_claim_dedup(rows, config=cd.claim_dedup_config(), embedder=Embedder())
    assert {r["record_id"] for r in result.kept} == {"a-headline", "b-stated"}
    skipped = result.outcome("embedding").skipped or {}
    assert skipped.get(cd.EmbeddingReason.TIER_TITLE_PREDICATE.value) == 1


# --- the claim universe ------------------------------------------------------


def eval_rows():
    base = {"claim": "x", "url": "u", "label": "false"}
    return [
        {**base, "record_id": "c1", "tier": "claimreview"},
        {**base, "record_id": "t1", "tier": "title_predicate"},
        {**base, "record_id": "t2", "tier": "title_predicate"},
    ]


def test_tier_4_claims_are_excluded_from_the_primary_set_by_default():
    universe = build_claim_universe(eval_rows(), [])
    assert universe.eval_ids == frozenset({"c1"})
    assert universe.excluded["t1"] == CLAIM_TIER4_EXCLUDED
    assert universe.title_predicate_ids == frozenset({"t1", "t2"})
    assert universe.tier_counts == {"claimreview": 1, "title_predicate": 2}


def test_the_switch_puts_tier_4_into_the_primary_set():
    universe = build_claim_universe(
        eval_rows(), [], title_predicate={"include_in_primary_evaluation": True}
    )
    assert universe.eval_ids == frozenset({"c1", "t1", "t2"})
    assert universe.title_predicate_included is True


def test_the_manifest_leaves_tier_4_out_unless_the_switch_is_on():
    assert [m["record_id"] for m in build_manifest(eval_rows())] == ["c1"]
    on = build_manifest(eval_rows(), include_title_predicate=True)
    assert [m["record_id"] for m in on] == ["c1", "t1", "t2"]


def test_a_manifest_that_holds_tier_4_is_refused_while_the_switch_is_off():
    manifest = build_manifest(eval_rows(), include_title_predicate=True)
    with pytest.raises(ManifestMismatch, match="tier title_predicate"):
        build_claim_universe(eval_rows(), [], manifest=manifest)


def test_the_error_rate_block_travels_into_the_universe():
    block = {"measured_error_rate": {"status": "not_measured", "reason": "no sheet"}}
    universe = build_claim_universe(eval_rows(), [], title_predicate=block)
    assert universe.title_predicate_error_rate["reason"] == "no sheet"


# --- the config switch and the measured block ---------------------------------


def write_config(tmp_path: Path, mutate) -> Path:
    config = real_config()
    mutate(config["title_predicate"])
    path = tmp_path / "factcheck.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def measured(config_path: Path, **overrides) -> dict:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block = {
        "status": "measured",
        "labeller_kind": "human",
        "pattern_hash": pattern_set_hash(data),
        "cohen_kappa": 0.7,
        "n_headlines": 100,
        "n_by_language": {"en": 60, "hi": 40},
        "precision": 0.9,
        "recall": 0.5,
    }
    block.update(overrides)
    return block


def test_a_string_switch_is_refused(tmp_path):
    path = write_config(tmp_path, lambda b: b.update(include_in_primary_evaluation="false"))
    with pytest.raises(EvalInputError):
        title_predicate_settings(path)


def test_a_complete_human_measurement_is_accepted(tmp_path):
    path = write_config(tmp_path, lambda b: None)
    block = measured(path)
    path = write_config(tmp_path, lambda b: b.update(measured_error_rate=block))
    assert title_predicate_settings(path)["measured_error_rate"]["status"] == "measured"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"labeller_kind": "model-reviewed"}, "not 'human'"),
        ({"labeller_kind": "model-unreviewed"}, "not 'human'"),
        ({"pattern_hash": "0" * 64}, "pattern_hash"),
        ({"n_headlines": 99}, "fewer than 100"),
        ({"cohen_kappa": 0.59}, "cohen_kappa"),
        ({"cohen_kappa": None}, "cohen_kappa"),
        ({"n_by_language": {"en": 71, "hi": 29}}, "fewer than 30 sampled hi"),
        ({"n_by_language": {"en": 29, "hi": 71}}, "fewer than 30 sampled en"),
    ],
)
def test_a_measured_block_that_does_not_hold_up_is_refused(tmp_path, override, message):
    path = write_config(tmp_path, lambda b: None)
    block = measured(path, **override)
    path = write_config(tmp_path, lambda b: b.update(measured_error_rate=block))
    with pytest.raises(EvalInputError, match=message):
        title_predicate_settings(path)


def test_a_measured_block_goes_stale_when_a_pattern_changes(tmp_path):
    path = write_config(tmp_path, lambda b: None)
    block = measured(path)

    def edit(b):
        b["measured_error_rate"] = block
        b["patterns"]["en"][0]["pattern"] += "|zzz"

    with pytest.raises(EvalInputError, match="pattern_hash"):
        title_predicate_settings(write_config(tmp_path, edit))


def test_the_checked_in_config_is_not_measured_and_passes():
    block = title_predicate_settings()
    assert block["measured_error_rate"]["status"] == "not_measured"
    assert block["include_in_primary_evaluation"] is False


# --- tier 4 rows on disk -----------------------------------------------------


def factcheck(
    record_id, tier, claim, *, domain="altnews.in", lang="en", simhash=1 << 20, label="false"
):
    return FactCheck(
        record_id=record_id,
        url=f"https://{domain}/{record_id}",
        domain=domain,
        publisher=domain,
        tier=tier,
        claim=claim,
        claim_lang=lang,
        rating_raw="False",
        label=label,
        claim_simhash=simhash,
        extracted_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def data_dir(tmp_path):
    verdicts = [
        factcheck("c1", "claimreview", "stated claim", domain="boomlive.in", lang="hi",
                  simhash=0xF),
        factcheck("c2", "claimreview", "another stated claim", domain="boomlive.in", lang="hi",
                  simhash=0xF0),
        factcheck("t1", "title_predicate", "Video Is Fake", domain="altnews.in", lang="en",
                  simhash=0xF00),
        factcheck("t2", "title_predicate", "वीडियो फर्जी है", domain="altnews.in", lang="hi",
                  simhash=0xF000),
    ]
    write_factchecks(verdicts, Settings(data_dir=tmp_path))
    return tmp_path


def test_freezing_leaves_tier_4_out_of_the_manifest(data_dir, tmp_path):
    path, count = freeze_manifest_from_data(data_dir, tmp_path / "manifest.csv")
    assert count == 2
    assert "t1" not in path.read_text() and "c1" in path.read_text()


def test_freezing_with_the_switch_on_keeps_tier_4_and_loading_agrees(
    data_dir, tmp_path, monkeypatch
):
    import modeling.config

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    write_config(tmp_path, lambda b: b.update(include_in_primary_evaluation=True))
    (tmp_path / "factcheck.yaml").replace(config_dir / "factcheck.yaml")
    monkeypatch.setattr(modeling.config, "CONFIG_DIR", config_dir)
    path, count = freeze_manifest_from_data(data_dir, tmp_path / "manifest.csv")
    assert count == 4
    universe = load_claim_universe(data_dir, path)
    assert universe.eval_ids == frozenset({"c1", "c2", "t1", "t2"})
    assert universe.frozen


def test_a_manifest_frozen_without_tier_4_loads_with_tier_4_outside_it(data_dir, tmp_path):
    path, _ = freeze_manifest_from_data(data_dir, tmp_path / "manifest.csv")
    universe = load_claim_universe(data_dir, path)
    assert universe.eval_ids == frozenset({"c1", "c2"})
    assert universe.title_predicate_ids == frozenset({"t1", "t2"})
    assert universe.excluded["t1"] == CLAIM_TIER4_EXCLUDED
    assert universe.tier_counts["title_predicate"] == 2


def test_the_hindi_shares_leave_tier_4_out_unless_the_switch_is_on(data_dir):
    shares = publisher_shares_from_data(data_dir)
    assert shares["n_hindi_claims"] == 2
    assert [p["publisher"] for p in shares["publishers"]] == ["boomlive.in"]


def test_the_hindi_shares_count_tier_4_when_the_switch_is_on(data_dir, tmp_path, monkeypatch):
    import modeling.config

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    write_config(tmp_path, lambda b: b.update(include_in_primary_evaluation=True))
    (tmp_path / "factcheck.yaml").replace(config_dir / "factcheck.yaml")
    monkeypatch.setattr(modeling.config, "CONFIG_DIR", config_dir)
    shares = publisher_shares_from_data(data_dir)
    assert shares["n_hindi_claims"] == 3


# --- the report --------------------------------------------------------------


def test_the_report_shows_tier_4_as_a_separate_cell_and_says_what_it_measures():
    extra = [r("claims", "t1", 0.95), r("claims", "t2", 0.1, lang="hi")]
    universe = ClaimUniverse(
        eval_ids=frozenset({"c1", "c2", "c3", "c4", "c5"}),
        excluded={"t1": CLAIM_TIER4_EXCLUDED, "t2": CLAIM_TIER4_EXCLUDED},
        frozen=False,
        title_predicate_ids=frozenset({"t1", "t2"}),
        title_predicate_error_rate={"status": "not_measured", "reason": "sheet not labelled"},
    )
    result = run(pair(extra), universe=universe, publishers=PUBLISHERS)
    block = result["title_predicate"]
    assert block["included_in_primary"] is False and block["n_claims"] == 2
    cell = block["cells"]["cand"]["all"]["claims"]
    assert cell["n_scored"] == 2 and cell["recall"]["value"] == 0.5
    assert result["cells"]["cand"]["all"]["claims"]["n_rows"] == 5
    text = render_markdown(result)
    assert "## Tier 4 claims read from headlines (separate cell)" in text
    assert "not measured (sheet not labelled)" in text
    assert "debunk headline" in text and "not the false claim itself" in text
    assert "debunk language" in text


def test_a_report_with_no_tier_4_claim_says_there_is_no_cell():
    assert "No tier 4 claim is present" in render_markdown(run(pair()))
