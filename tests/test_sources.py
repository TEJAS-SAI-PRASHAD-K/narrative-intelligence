"""Adapter mapping tests, run against recorded fixtures only.

No adapter test in this file makes a live call. Each one feeds recorded payload
dicts straight into ``to_record``, which is exactly why ``fetch()`` flattens
client objects into plain dicts before mapping them.
"""

from __future__ import annotations

import json

import pytest

from ingest.schema import DropReason, Record
from ingest.store import ParquetStore


def load_fixture(fixtures_dir, name: str):
    with (fixtures_dir / name).open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def store(settings) -> ParquetStore:
    settings.ensure_dirs()
    return ParquetStore(settings)


# --- reddit / convokit ----------------------------------------------------


@pytest.fixture
def convokit_payloads(fixtures_dir):
    return load_fixture(fixtures_dir, "reddit_convokit_utterances.json")


@pytest.fixture
def convokit(settings, store):
    from ingest.sources.reddit_convokit import ConvoKitSource

    return ConvoKitSource(settings=settings, store=store)


class TestConvoKitMapping:
    def test_root_utterance_becomes_a_post(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[0])
        assert record.content_type == "post"
        assert record.parent_id is None
        assert record.is_root

    def test_reply_becomes_a_threaded_comment(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[1])
        assert record.content_type == "comment"
        assert record.parent_id == "reddit:t1_d4e5f6g"
        assert record.conversation_id == "reddit:t3_9x1abc"

    def test_ids_are_namespaced(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[0])
        assert record.id == "reddit:t3_9x1abc"
        assert record.author_id == "reddit:concerned_citizen"

    def test_unix_timestamp_becomes_utc_aware(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[0])
        assert record.timestamp.tzinfo is not None
        assert record.timestamp.isoformat() == "2024-05-01T12:00:00+00:00"

    def test_score_maps_to_likes_and_negatives_survive(self, convokit, convokit_payloads):
        assert convokit.to_record(convokit_payloads[0]).engagement.likes == 1284
        assert convokit.to_record(convokit_payloads[1]).engagement.likes == -12

    def test_unavailable_metrics_are_null_not_zero(self, convokit, convokit_payloads):
        engagement = convokit.to_record(convokit_payloads[0]).engagement
        assert engagement.shares is None and engagement.views is None

    def test_subreddit_becomes_source_detail(self, convokit, convokit_payloads):
        assert convokit.to_record(convokit_payloads[0]).source_detail == "conspiracy"

    def test_platform_specifics_land_in_raw_only(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[0])
        assert record.raw["meta"]["permalink"].startswith("/r/conspiracy/")
        assert record.raw["meta"]["gilded"] == 1
        assert "permalink" not in record.model_dump(exclude={"raw"})

    def test_urls_and_domains_extracted_and_canonicalized(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[0])
        assert record.urls == ["https://www.example-news.com/story"]
        assert record.domains == ["example-news.com"]

    def test_removed_body_is_dropped_with_a_reason(self, convokit, convokit_payloads):
        assert convokit.to_record(convokit_payloads[2]) is None
        assert convokit.dropped[DropReason.DELETED_TEXT.value] == 1

    def test_deleted_author_keeps_the_text_and_flags_the_author(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[3])
        assert record is not None
        assert record.text.startswith("I saved screenshots")
        assert record.author_is_deleted
        assert record.author_handle is None
        assert convokit.flags["author_deleted"] == 1
        # It is a flag, not a drop: the text is still evidence.
        assert convokit.dropped.total() == 0

    def test_deleted_author_produces_no_author_rollup(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[3])
        assert convokit.to_author(convokit_payloads[3], record) is None

    def test_link_post_title_carries_the_claim(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[4])
        assert record.text == "Health agency retracts guidance after review"
        assert record.raw["title"] == "Health agency retracts guidance after review"

    def test_structured_url_from_meta_is_used(self, convokit, convokit_payloads):
        # The link target only exists in meta.url; the selftext is empty, so a
        # regex over the text would find nothing.
        record = convokit.to_record(convokit_payloads[4])
        assert record.urls == ["https://trib.al/xY9zQw2"]
        assert record.domains == ["trib.al"]

    def test_missing_timestamp_is_dropped_not_stamped_with_now(self, convokit, convokit_payloads):
        assert convokit.to_record(convokit_payloads[5]) is None
        assert convokit.dropped[DropReason.MISSING_TIMESTAMP.value] == 1

    def test_author_rollup_shape(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[0])
        author = convokit.to_author(convokit_payloads[0], record)
        assert author.author_id == "reddit:concerned_citizen"
        assert author.post_count == 1
        # Reddit corpora carry no follower graph and no account creation date;
        # claiming otherwise would fabricate coordination features.
        assert author.followers is None and author.created_at is None
        assert author.raw["num_posts"] == 412

    def test_simhash_and_lang_populated(self, convokit, convokit_payloads):
        record = convokit.to_record(convokit_payloads[0])
        assert record.lang == "en"
        assert isinstance(record.simhash, int) and record.simhash > 0

    def test_every_fixture_either_validates_or_drops_with_a_reason(
        self, convokit, convokit_payloads
    ):
        records = [convokit.to_record(payload) for payload in convokit_payloads]
        kept = [r for r in records if r is not None]
        assert all(isinstance(r, Record) for r in kept)
        assert len(kept) + convokit.dropped.total() == len(convokit_payloads)

    def test_ms_timestamps_are_handled(self, convokit, convokit_payloads):
        payload = dict(convokit_payloads[0])
        payload["timestamp"] = 1714564800000  # milliseconds
        assert convokit.to_record(payload).timestamp.year == 2024

    def test_records_round_trip_to_parquet(self, convokit, convokit_payloads, store):
        records = [r for r in map(convokit.to_record, convokit_payloads) if r is not None]
        assert store.write_records(records)["written"] == len(records)
        assert len(store.read_all("reddit")) == len(records)
