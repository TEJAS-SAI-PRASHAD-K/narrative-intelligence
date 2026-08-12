"""Schema is the contract. These tests are the enforcement of that contract."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ingest.schema import (
    DELETED_AUTHOR,
    Author,
    EngagementMetrics,
    Record,
    make_id,
    utcnow,
)

TS = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)


def minimal(**overrides) -> Record:
    kwargs = dict(
        native_id="abc123",
        source="reddit",
        source_detail="r/news",
        content_type="post",
        text="hello world",
        author_id=make_id("reddit", "u1"),
        timestamp=TS,
    )
    kwargs.update(overrides)
    return Record(**kwargs)


class TestIdNamespacing:
    def test_id_derived_from_source_and_native_id(self):
        assert minimal().id == "reddit:abc123"

    def test_explicit_matching_id_is_kept(self):
        assert minimal(id="reddit:abc123").id == "reddit:abc123"

    def test_wrongly_namespaced_id_rejected(self):
        with pytest.raises(ValidationError, match="namespaced"):
            minimal(id="mastodon:abc123")

    def test_bare_author_id_rejected(self):
        with pytest.raises(ValidationError, match="author_id"):
            minimal(author_id="u1")

    @pytest.mark.parametrize("field", ["parent_id", "conversation_id"])
    def test_threading_ids_must_be_namespaced(self, field):
        with pytest.raises(ValidationError, match=field):
            minimal(**{field: "t3_root"})
        assert getattr(minimal(**{field: "reddit:t3_root"}), field) == "reddit:t3_root"

    def test_cross_source_ids_cannot_collide(self):
        a = minimal(native_id="1")
        b = minimal(
            native_id="1",
            source="mastodon",
            author_id=make_id("mastodon", "u1"),
            content_type="post",
        )
        assert a.id != b.id


class TestTimestamps:
    def test_naive_timestamp_rejected_not_coerced(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            minimal(timestamp=datetime(2024, 5, 1, 12, 0))

    def test_non_utc_offset_converted_to_utc(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        rec = minimal(timestamp=datetime(2024, 5, 1, 17, 30, tzinfo=ist))
        assert rec.timestamp == TS
        assert rec.timestamp.tzinfo is timezone.utc

    def test_naive_ingested_at_rejected(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            minimal(ingested_at=datetime(2024, 5, 1))

    def test_ingested_at_defaults_to_aware_now(self):
        assert minimal().ingested_at.tzinfo is not None

    def test_date_partition_is_utc(self):
        tz = timezone(timedelta(hours=-8))
        rec = minimal(timestamp=datetime(2024, 5, 1, 20, 0, tzinfo=tz))  # 2024-05-02 04:00Z
        assert rec.date_partition == "2024-05-02"


class TestEngagement:
    def test_all_keys_present_and_null_by_default(self):
        assert minimal().engagement.model_dump() == {
            "likes": None,
            "shares": None,
            "replies": None,
            "views": None,
        }

    def test_null_is_not_zero(self):
        measured = EngagementMetrics(likes=0)
        assert measured.likes == 0 and measured.views is None
        assert measured.likes is not None

    def test_negative_score_preserved(self):
        # A downvoted Reddit post is signal, not noise to be clamped.
        assert EngagementMetrics(likes=-12).likes == -12

    def test_unknown_metric_rejected(self):
        with pytest.raises(ValidationError):
            EngagementMetrics(retweets=3)


class TestFieldHygiene:
    def test_extra_top_level_fields_rejected(self):
        # Platform-specific fields belong in `raw`, never at the top level.
        with pytest.raises(ValidationError):
            minimal(subreddit="r/news")

    @pytest.mark.parametrize(
        "value,expected",
        [("EN", "en"), ("zh-cn", "zh"), ("pt_BR", "pt"), ("und", None), (None, None)],
    )
    def test_lang_normalization(self, value, expected):
        assert minimal(lang=value).lang == expected

    def test_bad_lang_rejected(self):
        with pytest.raises(ValidationError, match="ISO 639"):
            minimal(lang="english!")

    def test_empty_required_strings_rejected(self):
        with pytest.raises(ValidationError):
            minimal(native_id="   ")
        with pytest.raises(ValidationError):
            minimal(source_detail="")

    def test_list_fields_dedupe_preserving_order(self):
        rec = minimal(hashtags=["a", "b", "a", " b ", ""])
        assert rec.hashtags == ["a", "b"]

    def test_simhash_must_fit_uint64(self):
        assert minimal(simhash=2**64 - 1).simhash == 2**64 - 1
        with pytest.raises(ValidationError, match="64-bit"):
            minimal(simhash=2**64)

    def test_bad_source_or_content_type_rejected(self):
        with pytest.raises(ValidationError):
            minimal(source="twitter")
        with pytest.raises(ValidationError):
            minimal(content_type="tweet")

    def test_raw_passthrough_is_untouched(self):
        payload = {"score": 3, "nested": {"a": [1, 2]}}
        assert minimal(raw=payload).raw == payload

    def test_deleted_author_sentinel(self):
        rec = minimal(author_id=make_id("reddit", DELETED_AUTHOR))
        assert rec.author_is_deleted
        assert not minimal().author_is_deleted

    def test_is_root(self):
        assert minimal().is_root
        assert not minimal(parent_id="reddit:t1_x").is_root


class TestAuthorRollup:
    def test_valid(self):
        a = Author(
            author_id="mastodon:acct1",
            source="mastodon",
            handle="user@instance.tld",
            created_at=TS,
            followers=10,
            following=3,
            post_count=2,
            first_seen=TS,
            last_seen=TS,
        )
        assert a.post_count == 2

    def test_naive_created_at_rejected(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            Author(
                author_id="mastodon:acct1",
                source="mastodon",
                created_at=datetime(2020, 1, 1),
                first_seen=TS,
                last_seen=TS,
            )

    def test_last_seen_before_first_seen_rejected(self):
        with pytest.raises(ValidationError, match="precedes"):
            Author(
                author_id="mastodon:acct1",
                source="mastodon",
                first_seen=TS,
                last_seen=TS - timedelta(days=1),
            )

    def test_null_counters_stay_null(self):
        a = Author(author_id="reddit:u1", source="reddit", first_seen=TS, last_seen=TS)
        assert a.followers is None and a.following is None


def test_utcnow_is_aware():
    assert utcnow().tzinfo is not None
