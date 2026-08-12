"""Round-tripping the corpus. If this is wrong, everything downstream is wrong."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ingest.schema import Record, make_id
from ingest.store import ARROW_SCHEMA, Manifest, ParquetStore, sha256_tree

TS = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(settings) -> ParquetStore:
    settings.ensure_dirs()
    return ParquetStore(settings)


def make_record(n: int, *, source="reddit", when: datetime = TS) -> Record:
    return Record(
        native_id=f"id{n}",
        source=source,
        source_detail="news",
        content_type="post" if source == "reddit" else "article",
        text=f"record number {n} with enough text to be interesting",
        author_id=make_id(source, f"author{n % 3}"),
        timestamp=when,
        engagement={"likes": n, "views": None},
        urls=["https://example.com/a"],
        domains=["example.com"],
        simhash=n,
        raw={"n": n},
    )


class TestRoundTrip:
    def test_write_then_read(self, store):
        result = store.write_records([make_record(i) for i in range(5)])
        assert result == {"written": 5, "duplicates": 0}
        df = store.read_all()
        assert len(df) == 5
        assert set(df["id"]) == {f"reddit:id{i}" for i in range(5)}

    def test_types_survive(self, store, sample_record):
        store.write_records([sample_record])
        table = store.dataset().to_table()
        assert table.schema.field("timestamp").type == ARROW_SCHEMA.field("timestamp").type
        row = table.to_pylist()[0]
        assert row["timestamp"] == sample_record.timestamp
        assert row["timestamp"].tzinfo is not None
        assert row["engagement"] == {"likes": 12, "shares": None, "replies": 3, "views": None}
        assert row["urls"] == ["https://example.com/a"]
        assert row["simhash"] == 12345678901234567890  # uint64, not silently float64

    def test_raw_is_json_and_recoverable(self, store, sample_record):
        import json

        store.write_records([sample_record])
        row = store.dataset().to_table().to_pylist()[0]
        assert json.loads(row["raw"])["nested"] == {"a": [1, 2]}

    def test_null_engagement_stays_null_not_zero(self, store):
        store.write_records([make_record(0)])
        row = store.dataset().to_table().to_pylist()[0]
        assert row["engagement"]["views"] is None
        assert row["engagement"]["likes"] == 0  # measured zero survives as zero


class TestPartitioning:
    def test_partitions_by_source_and_utc_date(self, store, settings):
        store.write_records(
            [
                make_record(1, source="reddit", when=TS),
                make_record(2, source="news", when=TS + timedelta(days=1)),
            ]
        )
        root = settings.normalized_dir
        assert (root / "source=reddit" / "date=2024-05-01").is_dir()
        assert (root / "source=news" / "date=2024-05-02").is_dir()

    def test_read_scoped_to_one_source(self, store):
        store.write_records([make_record(1, source="reddit"), make_record(2, source="news")])
        assert len(store.read_all("news")) == 1
        assert len(store.read_all()) == 2

    def test_empty_corpus_reads_as_empty_frame(self, store):
        df = store.read_all()
        assert len(df) == 0
        assert "id" in df.columns


class TestDedupe:
    def test_duplicate_ids_within_a_batch(self, store):
        result = store.write_records([make_record(1), make_record(1)])
        assert result == {"written": 1, "duplicates": 1}

    def test_duplicate_ids_across_runs(self, store):
        store.write_records([make_record(i) for i in range(3)])
        result = store.write_records([make_record(i) for i in range(5)])
        assert result == {"written": 2, "duplicates": 3}
        assert len(store.read_all()) == 5

    def test_rerun_is_idempotent(self, store):
        # This is the "kill it mid-fetch and rerun" acceptance criterion.
        batch = [make_record(i) for i in range(4)]
        store.write_records(batch)
        store.write_records(batch)
        store.write_records(batch)
        assert len(store.read_all()) == 4

    def test_dedupe_can_be_disabled(self, store):
        store.write_records([make_record(1)])
        store.write_records([make_record(1)], dedupe=False)
        assert len(store.read_all()) == 2

    def test_existing_ids_scoped_to_partition(self, store):
        store.write_records([make_record(1, source="reddit"), make_record(2, source="news")])
        assert store.existing_ids(source="reddit") == {"reddit:id1"}
        assert store.existing_ids(source="reddit", date="2024-05-01") == {"reddit:id1"}
        assert store.existing_ids(source="reddit", date="2000-01-01") == set()


class TestStats:
    def test_per_source_summary(self, store):
        store.write_records(
            [
                make_record(1, source="reddit"),
                make_record(2, source="reddit", when=TS + timedelta(days=2)),
                make_record(3, source="news"),
            ]
        )
        stats = {row["source"]: row for row in store.stats()}
        assert stats["reddit"]["records"] == 2
        assert stats["reddit"]["partitions"] == 2
        assert stats["news"]["records"] == 1
        assert stats["reddit"]["first"].startswith("2024-05-01")
        assert stats["reddit"]["last"].startswith("2024-05-03")

    def test_stats_on_empty_corpus(self, store):
        assert store.stats() == []


class TestManifest:
    def test_records_checksum_rows_and_bytes(self, tmp_path):
        artifact = tmp_path / "raw.jsonl"
        artifact.write_text('{"a":1}\n', encoding="utf-8")
        manifest = Manifest(tmp_path / "manifest.json")
        entry = manifest.record_artifact(
            "reddit_convokit:reddit-corpus-small",
            path=artifact,
            url="https://zissou.infosci.cornell.edu/convokit/datasets/reddit-corpus-small/",
            rows=1,
        )
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] == artifact.stat().st_size
        assert entry["rows"] == 1
        assert entry["fetched_at"].endswith("+00:00")

    def test_persisted_and_reloadable(self, tmp_path):
        path = tmp_path / "manifest.json"
        Manifest(path).record_artifact("a", url="https://x")
        assert Manifest(path).get("a")["url"] == "https://x"

    def test_directory_artifacts_get_a_tree_checksum(self, tmp_path):
        tree = tmp_path / "corpus"
        (tree / "sub").mkdir(parents=True)
        (tree / "sub" / "utterances.jsonl").write_text("{}\n", encoding="utf-8")
        digest, size = sha256_tree(tree)
        assert len(digest) == 64 and size > 0
        manifest = Manifest(tmp_path / "manifest.json")
        assert manifest.record_artifact("tree", path=tree)["checksum_kind"] == "sha256-tree"

    def test_corrupt_manifest_does_not_crash_the_run(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        assert Manifest(path).as_dict() == {}


class TestAuthorRollup:
    def test_merge_across_runs(self, store):
        from ingest.schema import Author

        a1 = Author(
            author_id="reddit:alice",
            source="reddit",
            first_seen=TS,
            last_seen=TS,
            post_count=2,
        )
        a2 = Author(
            author_id="reddit:alice",
            source="reddit",
            handle="alice",
            followers=10,
            first_seen=TS - timedelta(days=1),
            last_seen=TS + timedelta(days=1),
            post_count=3,
        )
        store.write_authors([a1], "reddit")
        store.write_authors([a2], "reddit")
        rows = store.read_authors("reddit")
        assert len(rows) == 1
        assert rows[0]["post_count"] == 5
        assert rows[0]["followers"] == 10
        assert rows[0]["first_seen"] == TS - timedelta(days=1)
        assert rows[0]["last_seen"] == TS + timedelta(days=1)
