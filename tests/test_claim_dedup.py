"""Claim collapse over stored fact-check rows. The artifact itself stays lossless."""

from __future__ import annotations

import random

import pytest

from ingest.factcheck import (
    FACTCHECK_SCHEMA,
    CollapseReason,
    FactCheck,
    collapse_duplicate_claims,
    read_deduplicated_factchecks,
)

DUP = CollapseReason.DUPLICATE_CLAIM.value
BASE = 0x0123456789ABCDEF


def flip(value: int, *bits: int) -> int:
    for bit in bits:
        value ^= 1 << bit
    return value


def row(
    record_id: str, simhash: int | None, extracted_at: str | None = "2026-01-01T00:00:00+00:00"
):
    return {"record_id": record_id, "claim_simhash": simhash, "extracted_at": extracted_at}


def ids(result) -> list[str]:
    return [r["record_id"] for r in result.kept]


def test_exact_duplicates_collapse_to_the_earliest():
    rows = [
        row("b", BASE, "2026-01-02T00:00:00+00:00"),
        row("a", BASE, "2026-01-03T00:00:00+00:00"),
        row("c", BASE, "2026-01-01T00:00:00+00:00"),
    ]
    result = collapse_duplicate_claims(rows)
    assert ids(result) == ["c"]
    assert result.counts[DUP] == 2
    assert result.collapsed == [("a", "c", 0), ("b", "c", 0)]


def test_equal_timestamps_break_on_record_id():
    result = collapse_duplicate_claims([row("z", BASE), row("m", BASE)])
    assert ids(result) == ["m"]


def test_missing_timestamp_never_wins():
    result = collapse_duplicate_claims([row("a", BASE, None), row("z", BASE)])
    assert ids(result) == ["z"]


def test_distance_three_collapses():
    result = collapse_duplicate_claims([row("a", BASE), row("b", flip(BASE, 0, 20, 63))])
    assert ids(result) == ["a"]
    assert result.counts[DUP] == 1


def test_distance_four_does_not_collapse():
    result = collapse_duplicate_claims([row("a", BASE), row("b", flip(BASE, 0, 20, 40, 63))])
    assert ids(result) == ["a", "b"]
    assert result.counts[DUP] == 0


@pytest.mark.parametrize("bits", [(0, 1, 2), (1, 17, 33), (16, 32, 48), (61, 62, 63)])
def test_distance_three_found_whatever_blocks_differ(bits):
    result = collapse_duplicate_claims([row("a", BASE), row("b", flip(BASE, *bits))])
    assert result.counts[DUP] == 1


def test_hash_zero_never_collapses():
    rows = [row("a", 0), row("b", 0), row("c", None), row("d", flip(0, 0))]
    result = collapse_duplicate_claims(rows)
    assert ids(result) == ["a", "b", "c", "d"]
    assert result.counts[DUP] == 0
    assert result.counts["unhashed_kept"] == 3


def test_hash_zero_does_not_absorb_a_near_zero_hash():
    result = collapse_duplicate_claims([row("a", 0), row("b", 0b111)])
    assert ids(result) == ["a", "b"]


def test_transitive_chain_is_one_group():
    # a-b is 3 bits, b-c is 3 bits, a-c is 6 bits.
    a = BASE
    b = flip(a, 0, 1, 2)
    c = flip(b, 10, 11, 12)
    rows = [
        row("a", a, "2026-01-03T00:00:00+00:00"),
        row("b", b, "2026-01-02T00:00:00+00:00"),
        row("c", c, "2026-01-01T00:00:00+00:00"),
    ]
    result = collapse_duplicate_claims(rows)
    assert ids(result) == ["c"]
    assert result.counts[DUP] == 2
    assert ("a", "c", 6) in result.collapsed


def test_result_does_not_depend_on_input_order():
    a = BASE
    b = flip(a, 0, 1, 2)
    c = flip(b, 10, 11, 12)
    rows = [
        row("a", a, "2026-01-03T00:00:00+00:00"),
        row("b", b, "2026-01-02T00:00:00+00:00"),
        row("c", c, "2026-01-01T00:00:00+00:00"),
        row("d", flip(a, 30, 31, 32, 33, 34, 35)),
        row("e", flip(a, 30, 31, 32, 33, 34, 35)),
        row("f", 0),
    ]
    expected = collapse_duplicate_claims(rows)
    rng = random.Random(7)
    for _ in range(25):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        result = collapse_duplicate_claims(shuffled)
        assert result.kept == expected.kept
        assert result.collapsed == expected.collapsed
        assert result.counts == expected.counts


def test_hashes_above_2_pow_63_collapse():
    high = (1 << 63) | BASE
    top = (1 << 64) - 1
    result = collapse_duplicate_claims(
        [row("a", high), row("b", flip(high, 63)), row("c", top), row("d", flip(top, 0, 63))]
    )
    assert ids(result) == ["a", "c"]
    assert result.counts[DUP] == 2


def test_hash_outside_unsigned_64_bit_is_refused():
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        collapse_duplicate_claims([row("a", 1 << 64)])


def test_accepts_factcheck_objects():
    def make(record_id, simhash):
        return FactCheck(
            record_id=record_id, url="u", domain="d", publisher="p", tier="rss_tag",
            claim="c", claim_lang="en", rating_raw="False", label="false", claim_simhash=simhash,
            extracted_at="2026-01-01T00:00:00+00:00",
        )

    result = collapse_duplicate_claims([make("b", BASE), make("a", BASE)])
    assert ids(result) == ["a"]


def test_reader_collapses_and_leaves_the_file_untouched(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from ingest.config import Settings

    settings = Settings(data_dir=tmp_path)
    out = tmp_path / "factchecks"
    out.mkdir()
    base = {
        "url": "u", "domain": "d", "publisher": "p", "tier": "rss_tag", "claim": "c",
        "claim_lang": "en", "rating_raw": "False", "label": "false", "reviewed_at": None,
        "extracted_at": "2026-01-01T00:00:00+00:00",
    }
    high = (1 << 63) | BASE
    rows = [
        {**base, "record_id": "x", "claim_simhash": high},
        {**base, "record_id": "y", "claim_simhash": flip(high, 5)},
        {**base, "record_id": "z", "claim_simhash": 0},
    ]
    path = out / "factchecks.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=FACTCHECK_SCHEMA()), path)
    before = path.read_bytes()

    result = read_deduplicated_factchecks(settings)

    assert [r["record_id"] for r in result.kept] == ["x", "z"]
    assert result.counts[DUP] == 1
    assert path.read_bytes() == before


def test_reader_with_no_file_is_empty(tmp_path):
    from ingest.config import Settings

    result = read_deduplicated_factchecks(Settings(data_dir=tmp_path))
    assert result.kept == [] and result.counts[DUP] == 0
