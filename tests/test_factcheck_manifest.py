"""The frozen evaluation set: build, verify, and stop on any change."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.factcheck import FACTCHECK_SCHEMA
from modeling.eval.factcheck_eval import (
    CLAIM_COLLAPSED_DUPLICATE,
    CLAIM_NOT_IN_FROZEN_SET,
    CLAIM_NOT_KNOWN_FALSE,
    CLAIM_UNKNOWN_ID,
    EvalInputError,
    ManifestMismatch,
    build_claim_universe,
    build_manifest,
    claim_hash,
    freeze_manifest_from_data,
    load_claim_universe,
    read_manifest,
    verify_manifest,
    write_manifest,
)

# SHA-256 of the three bytes "abc", from FIPS 180-4 test vectors.
ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def kept(record_id, claim, label="false", url=None):
    return {
        "record_id": record_id,
        "claim": claim,
        "label": label,
        "url": url or f"https://example.org/{record_id}",
    }


def test_claim_hash_is_sha256_of_the_utf8_text():
    assert claim_hash("abc") == ABC


def test_build_manifest_keeps_only_false_rows_sorted_and_without_claim_text():
    rows = build_manifest([kept("b", "abc"), kept("a", "x", label="true"), kept("c", "abc")])
    assert [r["record_id"] for r in rows] == ["b", "c"]
    assert rows[0] == {"record_id": "b", "url": "https://example.org/b", "claim_sha256": ABC}
    assert "claim" not in rows[0]


def test_write_then_read_round_trips_and_has_no_claim_text(tmp_path):
    path = tmp_path / "m.csv"
    write_manifest(build_manifest([kept("a", "abc")]), path)
    assert path.read_text() == (
        "record_id,url,claim_sha256\n" f"a,https://example.org/a,{ABC}\n"
    )
    assert read_manifest(path) == [
        {"record_id": "a", "url": "https://example.org/a", "claim_sha256": ABC}
    ]


def test_missing_manifest_reads_as_none(tmp_path):
    assert read_manifest(tmp_path / "absent.csv") is None


def test_write_refuses_to_replace_a_frozen_set(tmp_path):
    path = tmp_path / "m.csv"
    rows = build_manifest([kept("a", "abc")])
    write_manifest(rows, path)
    with pytest.raises(EvalInputError, match="already exists"):
        write_manifest(rows, path)
    write_manifest(rows, path, overwrite=True)


def test_empty_manifest_is_not_written(tmp_path):
    with pytest.raises(EvalInputError, match="empty"):
        write_manifest([], tmp_path / "m.csv")


@pytest.mark.parametrize(
    "text",
    [
        "id,url,hash\na,u,x\n",
        "record_id,url,claim_sha256\n",
        f"record_id,url,claim_sha256\na,u,{ABC}\na,u,{ABC}\n",
        "record_id,url,claim_sha256\na,u,not-a-hash\n",
    ],
)
def test_malformed_manifest_refuses(tmp_path, text):
    path = tmp_path / "m.csv"
    path.write_text(text)
    with pytest.raises(ManifestMismatch):
        read_manifest(path)


def test_verify_passes_when_every_row_matches():
    manifest = build_manifest([kept("a", "abc")])
    verify_manifest(manifest, [kept("a", "abc"), kept("z", "new claim")], [])


def test_changed_claim_text_stops():
    manifest = build_manifest([kept("a", "abc")])
    with pytest.raises(ManifestMismatch, match="claim hash differs: a"):
        verify_manifest(manifest, [kept("a", "abd")], [])


def test_missing_row_stops():
    manifest = build_manifest([kept("a", "abc"), kept("b", "xyz")])
    with pytest.raises(ManifestMismatch, match="missing from the fact-check rows: b"):
        verify_manifest(manifest, [kept("a", "abc")], [])


def test_changed_url_stops():
    manifest = build_manifest([kept("a", "abc")])
    with pytest.raises(ManifestMismatch, match="URL differs: a"):
        verify_manifest(manifest, [kept("a", "abc", url="https://elsewhere.org/a")], [])


def test_row_that_became_a_collapsed_duplicate_stops():
    manifest = build_manifest([kept("b", "abc")])
    with pytest.raises(ManifestMismatch, match="now a duplicate of a"):
        verify_manifest(manifest, [kept("a", "abc")], [("b", "a", 0)])


def test_universe_without_manifest_is_the_deduplicated_false_rows():
    universe = build_claim_universe(
        [kept("a", "x"), kept("c", "y", label="true")], [("b", "a", 0)]
    )
    assert universe.eval_ids == {"a"}
    assert not universe.frozen
    assert universe.reason_for_exclusion("a") is None
    assert universe.reason_for_exclusion("b") == CLAIM_COLLAPSED_DUPLICATE
    assert universe.reason_for_exclusion("c") == CLAIM_NOT_KNOWN_FALSE
    assert universe.reason_for_exclusion("never-seen") == CLAIM_UNKNOWN_ID


def test_frozen_universe_is_exactly_the_manifest_ids():
    manifest = build_manifest([kept("a", "abc")])
    universe = build_claim_universe(
        [kept("a", "abc"), kept("z", "later claim")], [], manifest=manifest
    )
    assert universe.frozen
    assert universe.eval_ids == {"a"}
    assert universe.reason_for_exclusion("z") == CLAIM_NOT_IN_FROZEN_SET


def test_frozen_universe_with_a_changed_row_stops():
    manifest = build_manifest([kept("a", "abc")])
    with pytest.raises(ManifestMismatch):
        build_claim_universe([kept("a", "tampered")], [], manifest=manifest)


def _write_factchecks(data_dir, rows):
    out = data_dir / "factchecks"
    out.mkdir(parents=True, exist_ok=True)
    full = [
        {
            "record_id": r["record_id"],
            "url": f"https://example.org/{r['record_id']}",
            "domain": "example.org",
            "publisher": "Example",
            "tier": "claimreview",
            "claim": r["claim"],
            "claim_lang": "en",
            "rating_raw": "False",
            "label": r.get("label", "false"),
            "claim_simhash": r["simhash"],
            "reviewed_at": None,
            "extracted_at": "2026-01-01T00:00:00+00:00",
        }
        for r in rows
    ]
    table = pa.Table.from_pylist(full, schema=FACTCHECK_SCHEMA())
    pq.write_table(table, out / "factchecks.parquet")


def test_freeze_and_load_from_a_real_parquet_collapses_duplicates(tmp_path):
    base = 0x0123456789ABCDEF
    _write_factchecks(
        tmp_path / "data",
        [
            {"record_id": "a", "claim": "one", "simhash": base},
            {"record_id": "b", "claim": "one again", "simhash": base},  # duplicate of a
            {"record_id": "c", "claim": "two", "simhash": base ^ ((1 << 64) - 1)},
        ],
    )
    path = tmp_path / "manifest.csv"
    _, count = freeze_manifest_from_data(tmp_path / "data", path)
    assert count == 2
    assert [r["record_id"] for r in read_manifest(path)] == ["a", "c"]

    universe = load_claim_universe(tmp_path / "data", path)
    assert universe.frozen and universe.eval_ids == {"a", "c"}
    assert universe.reason_for_exclusion("b") == CLAIM_COLLAPSED_DUPLICATE

    # Edit one claim on disk after freezing: the harness must stop.
    _write_factchecks(
        tmp_path / "data",
        [
            {"record_id": "a", "claim": "one EDITED", "simhash": base},
            {"record_id": "c", "claim": "two", "simhash": base ^ ((1 << 64) - 1)},
        ],
    )
    with pytest.raises(ManifestMismatch, match="claim hash differs"):
        load_claim_universe(tmp_path / "data", path)


def test_load_without_a_manifest_is_not_frozen(tmp_path):
    _write_factchecks(tmp_path / "data", [{"record_id": "a", "claim": "x", "simhash": 5}])
    universe = load_claim_universe(tmp_path / "data", tmp_path / "absent.csv")
    assert not universe.frozen and universe.eval_ids == {"a"}
