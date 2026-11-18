"""The Hindi share per publisher, in the frozen set summary and in the report."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from ingest.factcheck import FACTCHECK_SCHEMA
from modeling import cli
from modeling.config import get_settings
from modeling.eval.factcheck_eval import (
    HINDI_SINGLE_PUBLISHER,
    build_claim_universe,
    build_manifest,
    claims_by_publisher,
    hindi_publisher_shares,
    publisher_shares_from_data,
    run_comparison,
)
from tests._factcheck_eval_support import COLUMNS, VAL_A, make_test_rows, small_case_rows

runner = CliRunner()


def row(record_id, domain, lang, label="false"):
    return {
        "record_id": record_id,
        "claim": f"claim {record_id}",
        "label": label,
        "url": f"https://{domain}/{record_id}",
        "domain": domain,
        "claim_lang": lang,
    }


MIXED = [
    row("a", "vishvasnews.com", "hi"),
    row("b", "vishvasnews.com", "hi"),
    row("c", "vishvasnews.com", "hi"),
    row("d", "boomlive.in", "hi"),
    row("e", "boomlive.in", "en"),
    row("f", "boomlive.in", None),
    row("g", "vishvasnews.com", "hi", label="true"),
]


def test_claims_are_counted_per_publisher_and_language_over_false_rows_only():
    assert claims_by_publisher(MIXED) == {
        "boomlive.in": {"en": 1, "hi": 1, "undetected": 1},
        "vishvasnews.com": {"hi": 3},
    }


def test_a_row_without_a_domain_is_counted_under_unknown_not_dropped():
    assert claims_by_publisher([{**row("x", "a.org", "hi"), "domain": None}]) == {
        "unknown": {"hi": 1}
    }


def test_hindi_shares_are_per_publisher_largest_first_and_sum_to_one():
    shares = hindi_publisher_shares(claims_by_publisher(MIXED))
    assert shares["n_hindi_claims"] == 4
    assert shares["n_publishers"] == 2
    assert shares["reason"] is None
    assert [(p["publisher"], p["n_hindi_claims"]) for p in shares["publishers"]] == [
        ("vishvasnews.com", 3),
        ("boomlive.in", 1),
    ]
    assert shares["publishers"][0]["share"] == pytest.approx(0.75)
    assert sum(p["share"] for p in shares["publishers"]) == pytest.approx(1.0)


def test_an_undetected_language_is_never_counted_as_hindi():
    shares = hindi_publisher_shares({"boomlive.in": {"undetected": 5}})
    assert shares["n_hindi_claims"] == 0
    assert shares["publishers"] == []
    assert shares["reason"] == HINDI_SINGLE_PUBLISHER


def test_one_publisher_carries_the_single_publisher_reason():
    shares = hindi_publisher_shares({"vishvasnews.com": {"hi": 10}})
    assert shares["reason"] == HINDI_SINGLE_PUBLISHER
    assert shares["publishers"][0]["share"] == 1.0


def test_the_universe_counts_only_its_evaluation_ids():
    manifest = build_manifest([r for r in MIXED if r["record_id"] != "e"])
    universe = build_claim_universe(MIXED, [], manifest=manifest)
    assert universe.claims_by_publisher["boomlive.in"] == {"hi": 1, "undetected": 1}
    assert universe.claims_by_publisher["vishvasnews.com"] == {"hi": 3}


def _write(data_dir, rows):
    out = data_dir / "factchecks"
    out.mkdir(parents=True, exist_ok=True)
    full = [
        {
            "record_id": r["record_id"],
            "url": r["url"],
            "domain": r["domain"],
            "publisher": r["domain"],
            "tier": "claimreview",
            "claim": r["claim"],
            "claim_lang": r["claim_lang"],
            "rating_raw": "False",
            "label": r["label"],
            "claim_simhash": r["simhash"],
            "reviewed_at": None,
            "extracted_at": "2026-01-01T00:00:00+00:00",
        }
        for r in rows
    ]
    table = pa.Table.from_pylist(full, schema=FACTCHECK_SCHEMA())
    pq.write_table(table, out / "factchecks.parquet")


SIMHASHES = [0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0xF0F0F0F0F0F0F0F0, 0x0F0F0F0F0F0F0F0F]


def stored_rows():
    domains = ["vishvasnews.com", "vishvasnews.com", "vishvasnews.com", "boomlive.in"]
    return [
        {**row(f"c{i}", d, "hi"), "simhash": h}
        for i, (d, h) in enumerate(zip(domains, SIMHASHES, strict=True), start=1)
    ]


def test_shares_from_the_parquet_on_disk(tmp_path):
    _write(tmp_path, stored_rows())
    shares = publisher_shares_from_data(tmp_path)
    assert shares["n_hindi_claims"] == 4
    assert [p["publisher"] for p in shares["publishers"]] == ["vishvasnews.com", "boomlive.in"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts))
    get_settings.cache_clear()
    scores = tmp_path / "scores"
    scores.mkdir()
    frame = pd.DataFrame(
        VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + small_case_rows(), columns=COLUMNS
    )
    for name in ("tfidf", "cand"):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, scores / f"{name}.parquet")
    _write(data, stored_rows())
    yield {"scores": scores, "data": data, "manifest": tmp_path / "manifest.csv"}
    get_settings.cache_clear()


def test_the_report_lists_hindi_claims_by_publisher(env):
    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=1
    )
    payload = json.loads(paths[0].read_text())
    shares = payload["evaluation_set"]["hindi_publisher_shares"]
    assert shares["n_publishers"] == 2
    assert payload["evaluation_set"]["claims_by_publisher"]["boomlive.in"] == {"hi": 1}
    text = paths[1].read_text()
    assert "## Hindi claims by publisher" in text
    assert "| vishvasnews.com | 3 | 75.0% |" in text
    assert "| boomlive.in | 1 | 25.0% |" in text
    assert HINDI_SINGLE_PUBLISHER not in text
    assert result["evaluation_set"]["hindi_publisher_shares"]["reason"] is None


def test_the_report_warns_when_hindi_rests_on_one_publisher(env):
    _write(
        env["data"],
        [{**r, "domain": "vishvasnews.com", "url": r["url"]} for r in stored_rows()],
    )
    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=1
    )
    assert any(n.startswith(HINDI_SINGLE_PUBLISHER) for n in result["notices"])
    assert HINDI_SINGLE_PUBLISHER in paths[1].read_text()


def test_freeze_prints_the_hindi_share_per_publisher(env):
    result = runner.invoke(cli.app, ["freeze-factchecks", "--out", str(env["manifest"])])
    assert result.exit_code == 0, result.output
    assert "Hindi claims: 4 from 2 publisher(s)" in result.output
    assert "vishvasnews.com: 3 (75.0% of Hindi claims)" in result.output
    assert "boomlive.in: 1 (25.0% of Hindi claims)" in result.output
