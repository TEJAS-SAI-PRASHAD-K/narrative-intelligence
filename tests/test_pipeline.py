"""Run-loop mechanics: rate limiting, checkpoints, quota, and BaseSource.run()."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest

from ingest.checkpoint import Checkpoint, QuotaExhausted, QuotaLedger
from ingest.ratelimit import TokenBucket, with_retry
from ingest.schema import DropReason, Record, make_id
from ingest.sources.base import BaseSource, SourceUnavailable
from ingest.store import ParquetStore

TS = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)


class TestTokenBucket:
    def test_burst_is_immediate(self):
        bucket = TokenBucket(rate_per_sec=10, burst=5)
        start = time.monotonic()
        for _ in range(5):
            bucket.acquire()
        assert time.monotonic() - start < 0.05

    def test_sustained_rate_is_enforced(self):
        bucket = TokenBucket(rate_per_sec=50, burst=1)
        start = time.monotonic()
        for _ in range(4):
            bucket.acquire()
        # 1 burst token + 3 refills at 50/s => at least ~60ms of waiting.
        assert time.monotonic() - start >= 0.05

    def test_rejects_nonsense_rate(self):
        with pytest.raises(ValueError):
            TokenBucket(0)

    def test_context_manager_acquires(self):
        with TokenBucket(1000) as bucket:
            assert isinstance(bucket, TokenBucket)


class TestRetry:
    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        @with_retry(attempts=3, initial_wait=0.001, max_wait=0.002)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 3

    def test_reraises_after_exhausting_attempts(self):
        @with_retry(attempts=2, initial_wait=0.001, max_wait=0.002)
        def always_fails():
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            always_fails()


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        cp = Checkpoint("mastodon", tmp_path)
        cp.set("max_id", "110234")
        assert Checkpoint("mastodon", tmp_path).get("max_id") == "110234"

    def test_missing_key_default(self, tmp_path):
        assert Checkpoint("gdelt", tmp_path).get("cursor", "none") == "none"

    def test_completed_units(self, tmp_path):
        cp = Checkpoint("news_rss", tmp_path)
        cp.mark_done("https://feeds.example/rss")
        assert Checkpoint("news_rss", tmp_path).is_done("https://feeds.example/rss")
        assert not cp.is_done("https://other.example/rss")

    def test_bump(self, tmp_path):
        cp = Checkpoint("youtube", tmp_path)
        assert cp.bump("pages") == 1
        assert cp.bump("pages", 2) == 3

    def test_corrupt_file_does_not_crash(self, tmp_path):
        (tmp_path / "gdelt.json").write_text("{oh no", encoding="utf-8")
        assert Checkpoint("gdelt", tmp_path).as_dict() == {}

    def test_clear(self, tmp_path):
        cp = Checkpoint("gdelt", tmp_path)
        cp.set("a", 1)
        cp.clear()
        assert not cp.path.exists()
        assert Checkpoint("gdelt", tmp_path).as_dict() == {}


class TestQuotaLedger:
    def test_charges_accumulate(self, tmp_path):
        ledger = QuotaLedger(Checkpoint("youtube", tmp_path), daily_limit=10_000)
        ledger.charge(100, call="search.list")
        ledger.charge(1, call="videos.list")
        assert ledger.spent == 101
        assert ledger.remaining == 9899
        assert ledger.count("search.list") == 1

    def test_hard_stops_before_overspending(self, tmp_path):
        ledger = QuotaLedger(Checkpoint("youtube", tmp_path), daily_limit=150)
        ledger.charge(100, call="search.list")
        with pytest.raises(QuotaExhausted, match="only 50"):
            ledger.charge(100, call="search.list")
        # The failed charge must not have been deducted.
        assert ledger.spent == 100

    def test_can_afford(self, tmp_path):
        ledger = QuotaLedger(Checkpoint("youtube", tmp_path), daily_limit=100)
        assert ledger.can_afford(100)
        assert not ledger.can_afford(101)

    def test_resets_on_new_utc_day(self, tmp_path):
        cp = Checkpoint("youtube", tmp_path)
        ledger = QuotaLedger(cp, daily_limit=1000)
        ledger.charge(900, call="search.list")
        cp.set("quota", {"date": "2000-01-01", "units": 900, "calls": {}})
        assert ledger.spent == 0  # stale day is discarded, not carried forward

    def test_survives_a_restart(self, tmp_path):
        QuotaLedger(Checkpoint("youtube", tmp_path), daily_limit=10_000).charge(2000, call="s")
        assert QuotaLedger(Checkpoint("youtube", tmp_path), daily_limit=10_000).spent == 2000


# --- BaseSource ----------------------------------------------------------


class FakeSource(BaseSource):
    """Minimal adapter used to exercise the run loop without any network."""

    name = "reddit_convokit"  # a source whose credential gate is always open
    source = "reddit"

    def __init__(self, payloads, **kwargs):
        super().__init__(**kwargs)
        self.payloads = payloads
        self.fetched_ids: list[str] = []

    def fetch(self) -> Iterator[dict]:
        for payload in self.payloads:
            self.fetched_ids.append(payload.get("id", "?"))
            self.checkpoint.set("last_seen", payload.get("id"))
            yield payload

    def to_record(self, raw: dict) -> Record | None:
        if raw.get("text") == "[deleted]":
            self.drop(DropReason.DELETED_TEXT, raw["id"])
            return None
        return Record(
            native_id=raw["id"],
            source="reddit",
            source_detail="news",
            content_type="comment",
            text=raw["text"],
            author_id=make_id("reddit", raw.get("author", "anon")),
            timestamp=raw.get("timestamp", TS),
        )


def payloads(n: int, start: int = 0) -> list[dict]:
    return [{"id": f"c{i}", "text": f"comment number {i}"} for i in range(start, start + n)]


@pytest.fixture
def store(settings) -> ParquetStore:
    settings.ensure_dirs()
    return ParquetStore(settings)


class TestRunLoop:
    def test_writes_records_and_reports(self, settings, store):
        result = FakeSource(payloads(3), settings=settings, store=store).run()
        assert result.ok
        assert (result.fetched, result.written, result.duplicates) == (3, 3, 0)
        assert len(store.read_all("reddit")) == 3

    def test_drops_are_counted_not_silent(self, settings, store):
        data = payloads(2) + [{"id": "c9", "text": "[deleted]"}]
        result = FakeSource(data, settings=settings, store=store).run()
        assert result.written == 2
        assert result.dropped[DropReason.DELETED_TEXT.value] == 1
        assert result.total_dropped == 1

    def test_validation_error_drops_one_record_not_the_run(self, settings, store):
        data = payloads(2) + [{"id": "", "text": "no id"}] + payloads(1, start=50)
        result = FakeSource(data, settings=settings, store=store).run()
        assert result.written == 3
        assert result.dropped[DropReason.VALIDATION_ERROR.value] == 1

    def test_unexpected_mapping_error_is_contained(self, settings, store):
        class Boom(FakeSource):
            def to_record(self, raw):
                if raw["id"] == "c1":
                    raise KeyError("unexpected shape")
                return super().to_record(raw)

        result = Boom(payloads(3), settings=settings, store=store).run()
        assert result.ok and result.written == 2
        assert result.dropped[DropReason.VALIDATION_ERROR.value] == 1

    def test_limit_stops_early(self, settings, store):
        result = FakeSource(payloads(100), settings=settings, store=store, limit=5).run()
        assert result.written == 5
        assert len(store.read_all("reddit")) == 5

    def test_flush_every_writes_in_batches(self, settings, store):
        source = FakeSource(payloads(10), settings=settings, store=store)
        assert source.run(flush_every=3).written == 10

    def test_rerun_resumes_without_duplicating(self, settings, store):
        # Simulates the acceptance criterion: die mid-fetch, rerun, no dupes.
        first = FakeSource(payloads(5), settings=settings, store=store, limit=3).run()
        assert first.written == 3
        second = FakeSource(payloads(5), settings=settings, store=store).run()
        assert second.duplicates == 3 and second.written == 2
        assert len(store.read_all("reddit")) == 5

    def test_checkpoint_persists_across_runs(self, settings, store):
        FakeSource(payloads(4), settings=settings, store=store).run()
        assert Checkpoint("reddit_convokit", settings.checkpoint_dir).get("last_seen") == "c3"

    def test_source_error_is_captured_not_raised(self, settings, store):
        class Exploding(FakeSource):
            def fetch(self):
                yield from payloads(1)
                raise RuntimeError("upstream died")

        result = Exploding([], settings=settings, store=store).run()
        assert not result.ok
        assert "upstream died" in result.error
        assert result.written == 1  # buffered work is still flushed

    def test_missing_credentials_skip_cleanly(self, settings, store):
        class Gated(FakeSource):
            name = "youtube"
            source = "youtube"

            def preflight(self):
                raise SourceUnavailable("youtube: YOUTUBE_API_KEY absent")

        result = Gated(payloads(2), settings=settings, store=store).run()
        assert result.ok and result.written == 0
        assert "YOUTUBE_API_KEY absent" in result.skipped_reason
        assert result.as_row()["status"] == "skipped"

    def test_missing_optional_package_skips_cleanly(self, settings, store):
        class NeedsPackage(FakeSource):
            requires_package = "definitely_not_installed_xyz"

        result = NeedsPackage(payloads(2), settings=settings, store=store).run()
        assert result.ok and result.written == 0
        assert "not installed" in result.skipped_reason

    def test_author_rollup_written(self, settings, store):
        from ingest.schema import Author

        class WithAuthors(FakeSource):
            def to_author(self, raw, record):
                return Author(
                    author_id=record.author_id,
                    source="reddit",
                    handle=raw.get("author", "anon"),
                    first_seen=record.timestamp,
                    last_seen=record.timestamp,
                    post_count=1,
                )

        data = [
            {"id": "c1", "text": "one", "author": "alice"},
            {"id": "c2", "text": "two", "author": "alice"},
            {"id": "c3", "text": "three", "author": "bob"},
        ]
        result = WithAuthors(data, settings=settings, store=store).run()
        assert result.authors == 2
        rows = {r["author_id"]: r for r in store.read_authors("reddit")}
        assert rows["reddit:alice"]["post_count"] == 2

    def test_run_result_row_shape(self, settings, store):
        row = FakeSource(payloads(1), settings=settings, store=store).run().as_row()
        assert set(row) == {
            "source",
            "fetched",
            "written",
            "duplicates",
            "dropped",
            "status",
            "note",
        }
        assert row["status"] == "ok"


class TestQuotaStopsRunCleanly:
    def test_quota_exhaustion_keeps_what_was_fetched(self, settings, store):
        class Quota(FakeSource):
            def fetch(self):
                yield from payloads(2)
                raise QuotaExhausted("daily budget spent")

        result = Quota([], settings=settings, store=store).run()
        assert result.ok and result.written == 2
        assert "daily budget spent" in result.skipped_reason


def test_authors_merge_keeps_widest_window(settings, store):
    from ingest.schema import Author

    source = FakeSource([], settings=settings, store=store)
    base = dict(author_id="reddit:a", source="reddit", post_count=1)
    source.note_author(Author(**base, first_seen=TS, last_seen=TS))
    source.note_author(
        Author(**base, first_seen=TS - timedelta(days=2), last_seen=TS + timedelta(days=1))
    )
    merged = source._authors["reddit:a"]
    assert merged.post_count == 2
    assert merged.first_seen == TS - timedelta(days=2)
    assert merged.last_seen == TS + timedelta(days=1)
