"""Telegram adapter: mapping from a recorded payload, flattening, and the skip paths.

Nothing here needs Telethon. ``fetch`` is driven through a fake client, and the
one test that touches Telethon's error classes skips when it is not installed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ingest.schema import DropReason
from ingest.sources.base import SourceUnavailable
from ingest.sources.telegram import TelegramSource, _flatten_channel, _flatten_message
from ingest.store import ParquetStore


@pytest.fixture
def payloads(fixtures_dir):
    with (fixtures_dir / "telegram_messages.json").open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def telegram(settings):
    settings.ensure_dirs()
    return TelegramSource(settings=settings, store=ParquetStore(settings))


@pytest.fixture
def credentialed(settings):
    settings.telegram_api_id = "12345"
    settings.telegram_api_hash = "hash"
    settings.telegram_session = "session"
    return settings


class TestMapping:
    def test_post_maps_onto_the_schema(self, telegram, payloads):
        record = telegram.to_record(payloads[0])
        assert record.id == "telegram:1234567890:41872"
        assert record.source == "telegram"
        assert record.content_type == "post"
        assert record.source_detail == "example_news_in"
        assert record.author_id == "telegram:1234567890"
        assert record.author_handle == "example_news_in"
        assert record.timestamp == datetime(2026, 10, 12, 8, 15, tzinfo=timezone.utc)
        assert record.conversation_id == "telegram:1234567890:41872"
        assert record.parent_id is None

    def test_text_hashtags_and_urls_come_from_the_shared_pipeline(self, telegram, payloads):
        record = telegram.to_record(payloads[0])
        assert record.text.startswith("चुनाव आयोग")
        assert "election" in record.hashtags
        assert "example-news.in" in record.domains

    def test_engagement_is_measured_where_exposed(self, telegram, payloads):
        engagement = telegram.to_record(payloads[0]).engagement
        assert engagement.views == 18250
        assert engagement.shares == 412
        assert engagement.replies == 37
        assert engagement.likes == 90

    def test_unexposed_metrics_are_none_not_zero(self, telegram, payloads):
        engagement = telegram.to_record(payloads[1]).engagement
        assert engagement.views == 9120
        assert engagement.shares == 0  # forwards was sent as a measured zero
        assert engagement.replies is None
        assert engagement.likes is None

    def test_measured_zeros_stay_zero(self, telegram, payloads):
        engagement = telegram.to_record(payloads[2]).engagement
        assert (engagement.views, engagement.shares, engagement.replies, engagement.likes) == (
            0,
            0,
            0,
            0,
        )

    def test_fully_unmeasured_post_has_no_numbers_at_all(self, telegram, payloads):
        engagement = telegram.to_record(payloads[3]).engagement
        assert engagement.model_dump() == {
            "likes": None,
            "shares": None,
            "replies": None,
            "views": None,
        }

    def test_platform_fields_live_in_raw(self, telegram, payloads):
        raw = telegram.to_record(payloads[1]).raw
        assert raw["message_id"] == 41873
        assert raw["channel_id"] == "1234567890"
        assert raw["channel_username"] == "example_news_in"
        assert raw["permalink"] == "https://t.me/example_news_in/41873"
        assert raw["media_type"] == "MessageMediaPhoto"
        assert raw["forwarded_from"]["channel_id"] == 998877
        assert raw["edit_date"] is None
        assert telegram.to_record(payloads[0]).raw["edit_date"] == "2026-10-12T08:40:00+00:00"

    def test_reaction_breakdown_is_kept_in_raw(self, telegram, payloads):
        assert telegram.to_record(payloads[0]).raw["reactions"] == [
            {"reaction": "👍", "count": 64},
            {"reaction": "🔥", "count": 26},
        ]
        assert telegram.to_record(payloads[1]).raw["reactions"] is None

    def test_channel_without_username_has_no_permalink_and_falls_back_to_id(
        self, telegram, payloads
    ):
        record = telegram.to_record(payloads[3])
        assert record.source_detail == "555000111"
        assert record.author_handle is None
        assert record.raw["permalink"] is None

    def test_reply_threads_within_the_channel(self, telegram, payloads):
        record = telegram.to_record(payloads[3])
        assert record.parent_id == "telegram:555000111:5"
        assert record.conversation_id is None

    def test_same_message_id_in_two_channels_does_not_collide(self, telegram, payloads):
        other = json.loads(json.dumps(payloads[0]))
        other["channel"]["id"] = 42
        assert telegram.to_record(payloads[0]).id != telegram.to_record(other).id

    def test_empty_text_is_dropped_with_a_reason(self, telegram, payloads):
        assert telegram.to_record(payloads[4]) is None
        assert telegram.dropped[DropReason.EMPTY_TEXT.value] == 1

    def test_missing_timestamp_is_dropped_not_stamped_with_now(self, telegram, payloads):
        assert telegram.to_record(payloads[5]) is None
        assert telegram.dropped[DropReason.MISSING_TIMESTAMP.value] == 1

    def test_missing_channel_id_is_dropped(self, telegram, payloads):
        broken = dict(payloads[0], channel={"username": "x"})
        assert telegram.to_record(broken) is None
        assert telegram.dropped[DropReason.MISSING_ID.value] == 1

    def test_author_rollup_uses_subscribers_when_known(self, telegram, payloads):
        record = telegram.to_record(payloads[0])
        author = telegram.to_author(payloads[0], record)
        assert author.followers == 482113
        assert author.following is None
        assert telegram.to_author(payloads[3], telegram.to_record(payloads[3])).followers is None

    def test_every_fixture_validates_or_drops_with_a_reason(self, telegram, payloads):
        mapped = [telegram.to_record(p) for p in payloads]
        assert sum(r is not None for r in mapped) == 4
        assert sum(telegram.dropped.values()) == 2

    def test_records_round_trip_to_parquet(self, telegram, payloads):
        records = [r for r in map(telegram.to_record, payloads) if r is not None]
        assert telegram.store.write_records(records)["written"] == len(records)
        assert len(telegram.store.read_all("telegram")) == len(records)


def _message(**overrides):
    base = {
        "id": 10,
        "date": datetime(2026, 10, 12, 8, 0, tzinfo=timezone.utc),
        "edit_date": None,
        "message": "body",
        "views": None,
        "forwards": None,
        "replies": None,
        "reactions": None,
        "fwd_from": None,
        "reply_to": None,
        "media": None,
        "grouped_id": None,
        "post_author": None,
        "action": None,
    }
    return SimpleNamespace(**(base | overrides))


CHANNEL = {"id": 1, "username": "chan", "title": "Chan", "date": None, "participants_count": None}


class TestFlattening:
    def test_absent_counters_stay_none(self):
        flat = _flatten_message(_message(), CHANNEL)
        assert flat["views"] is None
        assert flat["forwards"] is None
        assert flat["replies"] is None
        assert flat["reactions_total"] is None

    def test_reply_count_is_read_from_the_replies_object(self):
        flat = _flatten_message(_message(replies=SimpleNamespace(replies=0)), CHANNEL)
        assert flat["replies"] == 0

    def test_reactions_are_summed(self):
        reactions = SimpleNamespace(
            results=[
                SimpleNamespace(reaction=SimpleNamespace(emoticon="👍"), count=3),
                SimpleNamespace(reaction=SimpleNamespace(emoticon="🔥"), count=4),
            ]
        )
        flat = _flatten_message(_message(reactions=reactions), CHANNEL)
        assert flat["reactions_total"] == 7

    def test_present_but_empty_reactions_are_a_measured_zero(self):
        flat = _flatten_message(_message(reactions=SimpleNamespace(results=[])), CHANNEL)
        assert flat["reactions_total"] == 0

    def test_service_message_is_not_flattened(self):
        assert _flatten_message(_message(action=object()), CHANNEL) is None

    def test_output_is_plain_json(self):
        flat = _flatten_message(
            _message(
                edit_date=datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc),
                media=SimpleNamespace(),
                grouped_id=123,
                reply_to=SimpleNamespace(reply_to_msg_id=4),
                fwd_from=SimpleNamespace(
                    from_id=SimpleNamespace(channel_id=9),
                    from_name=None,
                    channel_post=2,
                    date=datetime(2026, 10, 12, 7, 0, tzinfo=timezone.utc),
                ),
            ),
            CHANNEL,
        )
        json.dumps(flat)
        assert flat["media_type"] == "SimpleNamespace"
        assert flat["grouped_id"] == "123"
        assert flat["reply_to_id"] == 4
        assert flat["forwarded_from"]["channel_id"] == 9

    def test_channel_username_falls_back_to_the_configured_handle(self):
        entity = SimpleNamespace(id=5, username=None, title="T")
        assert _flatten_channel(entity, fallback_username="cfg")["username"] == "cfg"


class TestSkipWithoutCredentials:
    def test_missing_credentials_skip_with_a_reason_code(self, telegram):
        with pytest.raises(SourceUnavailable, match="missing_credentials"):
            telegram.preflight()

    def test_message_names_every_missing_variable(self, telegram):
        with pytest.raises(SourceUnavailable) as info:
            telegram.preflight()
        for env in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION"):
            assert env in str(info.value)

    def test_partial_credentials_still_skip(self, settings):
        settings.telegram_api_id = "1"
        settings.telegram_api_hash = "h"
        source = TelegramSource(settings=settings, store=ParquetStore(settings))
        with pytest.raises(SourceUnavailable, match="TELEGRAM_SESSION"):
            source.preflight()
        assert not settings.has_credentials("telegram")

    def test_run_without_credentials_is_skipped_not_failed(self, telegram):
        result = telegram.run()
        assert result.ok
        assert result.skipped_reason and "missing_credentials" in result.skipped_reason
        assert result.written == 0

    def test_complete_credentials_pass_the_gate(self, credentialed):
        assert credentialed.has_credentials("telegram")

    def test_blank_env_values_count_as_absent(self, tmp_path):
        from ingest.config import Settings

        settings = Settings(
            data_dir=tmp_path,
            telegram_api_id="",
            telegram_api_hash="",
            telegram_session="",
            _env_file=None,
        )
        assert settings.telegram_api_id is None
        assert not settings.has_credentials("telegram")


class FakeClient:
    def __init__(self, history: dict[str, list], authorized: bool = True, fail: dict | None = None):
        self.history = history
        self.authorized = authorized
        self.fail = fail or {}
        self.min_ids: dict[str, int] = {}
        self.disconnected = False

    async def connect(self):
        return None

    async def is_user_authorized(self):
        return self.authorized

    async def disconnect(self):
        self.disconnected = True

    async def get_entity(self, name):
        if name in self.fail:
            raise self.fail[name]
        return SimpleNamespace(id=hash(name) % 1000 + 1, username=name, title=name.title())

    async def iter_messages(self, entity, limit, min_id):
        self.min_ids[entity.username] = min_id
        count = 0
        for message in self.history[entity.username]:
            if message.id <= min_id:
                continue
            if count >= limit:
                return
            count += 1
            yield message


@pytest.fixture
def fast(monkeypatch):
    """No pacing sleeps in tests."""
    monkeypatch.setattr("ingest.sources.telegram.TokenBucket.acquire", lambda self, tokens=1.0: 0.0)


def _fetching(settings, client, channels, **kwargs):
    source = TelegramSource(
        settings=settings, store=ParquetStore(settings), channels=channels, **kwargs
    )
    source._client = lambda: client
    return source


class TestFetch:
    def test_messages_are_flattened_before_mapping(self, credentialed, fast):
        client = FakeClient({"alpha": [_message(id=3, views=5), _message(id=2)]})
        source = _fetching(credentialed, client, ["alpha"])
        rows = list(source.fetch())
        assert [r["id"] for r in rows] == [3, 2]
        json.dumps(rows)
        records = [source.to_record(r) for r in rows]
        assert records[0].engagement.views == 5
        assert records[1].engagement.views is None
        assert client.disconnected

    def test_checkpoint_resumes_from_the_newest_seen_id(self, credentialed, fast):
        client = FakeClient({"alpha": [_message(id=3), _message(id=2)]})
        first = _fetching(credentialed, client, ["alpha"])
        list(first.fetch())
        first.checkpoint.save()

        client2 = FakeClient({"alpha": [_message(id=4), _message(id=3), _message(id=2)]})
        second = _fetching(credentialed, client2, ["alpha"])
        rows = list(second.fetch())
        assert client2.min_ids["alpha"] == 3
        assert [r["id"] for r in rows] == [4]

    def test_force_ignores_the_checkpoint(self, credentialed, fast):
        client = FakeClient({"alpha": [_message(id=3)]})
        first = _fetching(credentialed, client, ["alpha"])
        list(first.fetch())
        first.checkpoint.save()

        client2 = FakeClient({"alpha": [_message(id=3)]})
        second = _fetching(credentialed, client2, ["alpha"], force=True)
        assert len(list(second.fetch())) == 1
        assert client2.min_ids["alpha"] == 0

    def test_cut_short_run_does_not_advance_the_cursor(self, credentialed, fast):
        client = FakeClient({"alpha": [_message(id=3), _message(id=2)]})
        source = _fetching(credentialed, client, ["alpha"])
        stream = source.fetch()
        next(stream)
        stream.close()
        assert source.checkpoint.get("last_id.alpha") is None

    def test_per_channel_cap_follows_the_limit(self, credentialed, fast):
        client = FakeClient({"alpha": [_message(id=i) for i in range(10, 0, -1)]})
        source = _fetching(credentialed, client, ["alpha"])
        source.limit = 3
        assert len(list(source.fetch())) == 3

    def test_service_messages_are_counted_not_dropped_silently(self, credentialed, fast):
        client = FakeClient({"alpha": [_message(id=2, action=object()), _message(id=1)]})
        source = _fetching(credentialed, client, ["alpha"])
        assert [r["id"] for r in source.fetch()] == [1]
        assert source.flags["service_message"] == 1

    def test_expired_session_skips_with_a_reason(self, credentialed, fast):
        client = FakeClient({"alpha": []}, authorized=False)
        source = _fetching(credentialed, client, ["alpha"])
        with pytest.raises(SourceUnavailable, match="session_not_authorized"):
            list(source.fetch())
        assert client.disconnected

    def test_no_channels_configured_is_counted(self, credentialed, fast, monkeypatch):
        monkeypatch.setattr("ingest.sources.telegram.sources_config", lambda: {})
        source = _fetching(credentialed, FakeClient({}), [])
        assert list(source.fetch()) == []
        assert source.flags["no_channels_configured"] == 1

    def test_unreachable_channels_are_skipped_with_reason_codes(self, credentialed, fast):
        errors = pytest.importorskip("telethon.errors")
        client = FakeClient(
            {"good": [_message(id=1)]},
            fail={
                "gone": errors.UsernameNotOccupiedError(request=None),
                "closed": errors.ChannelPrivateError(request=None),
            },
        )
        source = _fetching(credentialed, client, ["gone", "closed", "good"])
        assert [r["id"] for r in source.fetch()] == [1]
        assert source.flags["channel_not_found"] == 1
        assert source.flags["channel_private"] == 1

    def test_channel_names_are_cleaned_and_deduplicated(self, credentialed):
        source = _fetching(credentialed, FakeClient({}), [" @alpha", "alpha", "beta", ""])
        assert source._channels() == ["alpha", "beta"]


class TestRegistration:
    def test_registered_and_requires_telethon(self):
        from ingest.sources import REGISTRY, get_source_class

        assert "telegram" in REGISTRY
        cls = get_source_class("telegram")
        assert cls.name == "telegram" and cls.requires_package == "telethon"

    def test_source_literal_accepts_telegram(self):
        from ingest.schema import SOURCES

        assert "telegram" in SOURCES
