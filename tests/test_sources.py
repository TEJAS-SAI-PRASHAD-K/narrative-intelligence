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


# --- mastodon -------------------------------------------------------------


@pytest.fixture
def mastodon_payloads(fixtures_dir):
    return load_fixture(fixtures_dir, "mastodon_statuses.json")


@pytest.fixture
def mastodon(settings, store):
    from ingest.sources.mastodon import MastodonSource

    return MastodonSource(settings=settings, store=store)


class TestMastodonMapping:
    def test_html_stripped_but_original_kept_in_raw(self, mastodon, mastodon_payloads):
        record = mastodon.to_record(mastodon_payloads[0])
        assert "<p>" not in record.text
        assert record.text.startswith("Six outlets, identical phrasing")
        assert record.raw["content"].startswith("<p>")

    def test_truncated_link_text_does_not_lose_the_destination(self, mastodon, mastodon_payloads):
        record = mastodon.to_record(mastodon_payloads[0])
        assert record.urls == ["https://www.example-news.com/story-2024"]
        assert record.domains == ["example-news.com"]

    def test_engagement_mapping(self, mastodon, mastodon_payloads):
        engagement = mastodon.to_record(mastodon_payloads[0]).engagement
        assert engagement.likes == 41  # favourites_count
        assert engagement.shares == 17  # reblogs_count
        assert engagement.replies == 3
        assert engagement.views is None  # Mastodon does not measure views

    def test_acct_keeps_the_instance_and_source_detail_is_the_instance(
        self, mastodon, mastodon_payloads
    ):
        record = mastodon.to_record(mastodon_payloads[0])
        assert record.author_handle == "researcher@scholar.social"
        assert record.source_detail == "scholar.social"

    def test_structured_tags_and_mentions_used(self, mastodon, mastodon_payloads):
        record = mastodon.to_record(mastodon_payloads[0])
        assert record.hashtags == ["election"]
        assert record.mentions == ["colleague@instance.tld"]

    def test_media_urls_captured(self, mastodon, mastodon_payloads):
        record = mastodon.to_record(mastodon_payloads[0])
        assert record.media_urls == ["https://files.mastodon.social/media/abc.png"]

    def test_root_is_its_own_conversation(self, mastodon, mastodon_payloads):
        record = mastodon.to_record(mastodon_payloads[0])
        assert record.parent_id is None
        assert record.conversation_id == "mastodon:110451234567890123"

    def test_boost_becomes_an_edge_not_a_collapsed_duplicate(self, mastodon, mastodon_payloads):
        record = mastodon.to_record(mastodon_payloads[1])
        # The boost is its own record: booster as author, boosted status as parent.
        assert record.id == "mastodon:110459999999999999"
        assert record.author_handle == "amplifier@botsin.space"
        assert record.parent_id == "mastodon:110451234567890123"
        assert record.raw["is_boost"] is True
        # Text comes from the boosted status, which has no text of its own.
        assert record.text.startswith("Six outlets")
        assert mastodon.flags["boost"] == 1

    def test_boost_engagement_comes_from_the_boosted_status(self, mastodon, mastodon_payloads):
        # The boost wrapper reports 0 favourites; that is an artefact, not a
        # measurement of the content.
        assert mastodon.to_record(mastodon_payloads[1]).engagement.likes == 41

    def test_reply_threads_but_root_is_not_fabricated(self, mastodon, mastodon_payloads):
        record = mastodon.to_record(mastodon_payloads[2])
        assert record.parent_id == "mastodon:110451234567890123"
        assert record.conversation_id is None  # resolving it costs an API call

    def test_instance_language_label_beats_detection(self, mastodon, mastodon_payloads):
        assert mastodon.to_record(mastodon_payloads[2]).lang == "de"

    def test_local_account_falls_back_to_profile_host(self, mastodon, mastodon_payloads):
        # acct has no @instance for a local user; the profile URL supplies it.
        assert mastodon.to_record(mastodon_payloads[2]).source_detail == "troet.cafe"

    def test_empty_content_dropped(self, mastodon, mastodon_payloads):
        assert mastodon.to_record(mastodon_payloads[3]) is None
        assert mastodon.dropped[DropReason.EMPTY_TEXT.value] == 1

    def test_author_rollup_captures_coordination_priors(self, mastodon, mastodon_payloads):
        record = mastodon.to_record(mastodon_payloads[1])
        author = mastodon.to_author(mastodon_payloads[1], record)
        assert author.followers == 4 and author.following == 900
        assert author.created_at.isoformat() == "2024-04-28T00:00:00+00:00"
        assert author.raw["bot"] is True
        assert author.raw["statuses_count"] == 6120

    def test_datetimes_are_flattened_before_mapping(self, mastodon):
        from datetime import datetime, timezone

        flattened = mastodon._jsonable({"created_at": datetime(2024, 5, 1, tzinfo=timezone.utc)})
        assert flattened == {"created_at": "2024-05-01T00:00:00+00:00"}


# --- gdelt ----------------------------------------------------------------


@pytest.fixture
def gdelt(settings, store):
    from ingest.sources.gdelt import GdeltSource

    return GdeltSource(settings=settings, store=store)


@pytest.fixture
def gdelt_doc(fixtures_dir):
    return load_fixture(fixtures_dir, "gdelt_doc_articles.json")


@pytest.fixture
def gdelt_gkg(fixtures_dir):
    return load_fixture(fixtures_dir, "gdelt_gkg_rows.json")


class TestGdeltDocMapping:
    def test_article_shape(self, gdelt, gdelt_doc):
        record = gdelt.to_record(gdelt_doc[0])
        assert record.content_type == "article"
        assert record.source_detail == "example-news.com"
        assert record.text.startswith("Officials review claim")

    def test_publisher_domain_drives_domain_risk_baseline(self, gdelt, gdelt_doc):
        record = gdelt.to_record(gdelt_doc[0])
        assert record.domains == ["example-news.com"]
        assert record.author_id == "gdelt:example-news.com"

    def test_seendate_parsed_as_utc(self, gdelt, gdelt_doc):
        record = gdelt.to_record(gdelt_doc[0])
        assert record.timestamp.isoformat() == "2024-05-01T12:00:00+00:00"

    def test_language_name_becomes_iso_code(self, gdelt, gdelt_doc):
        assert gdelt.to_record(gdelt_doc[1]).lang == "es"

    def test_engagement_is_entirely_unmeasured(self, gdelt, gdelt_doc):
        engagement = gdelt.to_record(gdelt_doc[0]).engagement
        assert engagement.model_dump() == {
            "likes": None,
            "shares": None,
            "replies": None,
            "views": None,
        }

    def test_untitled_row_dropped(self, gdelt, gdelt_doc):
        assert gdelt.to_record(gdelt_doc[2]) is None
        assert gdelt.dropped[DropReason.EMPTY_TEXT.value] == 1

    def test_socialimage_becomes_media_url(self, gdelt, gdelt_doc):
        assert gdelt.to_record(gdelt_doc[0]).media_urls == [
            "https://www.example-news.com/img/ballots.jpg"
        ]


class TestGdeltGkgMapping:
    def test_page_title_from_extras(self, gdelt, gdelt_gkg):
        record = gdelt.to_record(gdelt_gkg[0])
        assert record.text == "Officials review claim that ballots were counted twice"
        assert gdelt.flags["title_from_url_slug"] == 0

    def test_themes_tone_and_entities_land_in_raw(self, gdelt, gdelt_gkg):
        raw = gdelt.to_record(gdelt_gkg[0]).raw
        assert raw["themes"] == ["ELECTION", "DEMOCRACY", "MEDIA_MSM"]
        assert raw["organizations"] == ["election commission", "associated press"]
        assert round(raw["tone"]["tone"], 2) == -3.45
        assert raw["tone"]["word_count"] == 247

    def test_gkg_timestamp_format(self, gdelt, gdelt_gkg):
        assert gdelt.to_record(gdelt_gkg[0]).timestamp.isoformat() == "2024-05-01T12:00:00+00:00"

    def test_slug_reconstructed_title_is_flagged_as_such(self, gdelt, gdelt_gkg):
        record = gdelt.to_record(gdelt_gkg[1])
        assert record.text == "agency retracts vaccine guidance after review 2024"
        assert gdelt.flags["title_from_url_slug"] == 1

    def test_non_web_row_dropped(self, gdelt, gdelt_gkg):
        assert gdelt.to_record(gdelt_gkg[2]) is None
        assert gdelt.dropped[DropReason.UNSUPPORTED_TYPE.value] == 1

    def test_lastupdate_parsing(self):
        from ingest.sources.gdelt import _parse_lastupdate

        text = (
            "205896 e0f1 http://data.gdeltproject.org/gdeltv2/20240501120000.export.CSV.zip\n"
            "1204993 a1b2 http://data.gdeltproject.org/gdeltv2/20240501120000.mentions.CSV.zip\n"
            "6829418 c3d4 http://data.gdeltproject.org/gdeltv2/20240501120000.gkg.csv.zip\n"
        )
        urls = _parse_lastupdate(text)
        assert len(urls) == 3 and urls[2].endswith(".gkg.csv.zip")

    def test_file_kind_detection(self):
        from ingest.sources.gdelt import _file_kind

        assert _file_kind("http://x/20240501.gkg.csv.zip") == "gkg"
        assert _file_kind("http://x/20240501.mentions.CSV.zip") == "mentions"
        assert _file_kind("http://x/20240501.export.CSV.zip") == "export"


# --- news / rss -----------------------------------------------------------


@pytest.fixture
def news(settings, store):
    from ingest.sources.news_rss import NewsRssSource

    return NewsRssSource(settings=settings, store=store)


@pytest.fixture
def news_payloads(fixtures_dir):
    return load_fixture(fixtures_dir, "news_rss_entries.json")


class _RobotsResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


class _RobotsSession:
    def __init__(self, status_code: int, body: bytes = b""):
        self.status_code = status_code
        self.body = body
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return _RobotsResponse(self.status_code, self.body)


class TestNewsRssRobots:
    URL_A = "https://news.example.com/a"
    URL_B = "https://news.example.com/b"

    def _use(self, news, monkeypatch, status, body=b""):
        session = _RobotsSession(status, body)
        monkeypatch.setattr(news, "http", lambda: session)
        return session

    @pytest.mark.parametrize("status", [404, 410])
    def test_no_published_rules_allows(self, news, monkeypatch, status):
        self._use(news, monkeypatch, status)
        assert news._robots_allows(self.URL_A) is True
        assert "robots_unreachable" not in news.flags

    @pytest.mark.parametrize("status", [403, 429, 503])
    def test_any_other_status_is_unreachable(self, news, monkeypatch, status):
        self._use(news, monkeypatch, status)
        assert news._robots_allows(self.URL_A) is False
        assert news.flags["robots_unreachable"] == 1
        assert "robots_disallowed" not in news.flags

    @pytest.mark.parametrize(
        "body", [b"User-agent: *\nDisallow: /", b"\xef\xbb\xbfUser-agent: *\nDisallow: /"]
    )
    def test_a_real_disallow_is_counted_as_one(self, news, monkeypatch, body):
        """A 200 with a Disallow blocks the fetch and is counted, with or without a BOM."""
        self._use(news, monkeypatch, 200, body)
        assert news._robots_allows(self.URL_A) is False
        assert news.flags["robots_disallowed"] == 1
        assert "robots_unreachable" not in news.flags

    def test_a_200_with_no_matching_rule_allows(self, news, monkeypatch):
        self._use(news, monkeypatch, 200, b"User-agent: *\nDisallow: /private/\n")
        assert news._robots_allows(self.URL_A) is True
        assert "robots_disallowed" not in news.flags

    def test_one_robots_request_per_host(self, news, monkeypatch):
        session = self._use(news, monkeypatch, 404)
        news._robots_allows(self.URL_A)
        news._robots_allows(self.URL_B)
        assert session.calls == ["https://news.example.com/robots.txt"]


class TestNewsRssMapping:
    def test_title_and_summary_combined(self, news, news_payloads):
        record = news.to_record(news_payloads[0])
        assert record.text.startswith("Officials review claim")
        assert "no evidence of double counting" in record.text

    def test_escaped_entities_unescaped(self, news, news_payloads):
        assert '"The process is audited,"' in news.to_record(news_payloads[0]).text

    def test_struct_time_becomes_utc_aware(self, news, news_payloads):
        record = news.to_record(news_payloads[0])
        assert record.timestamp.isoformat() == "2024-05-01T12:00:00+00:00"

    def test_outlet_domain_is_source_detail_and_pseudo_author(self, news, news_payloads):
        record = news.to_record(news_payloads[0])
        assert record.source_detail == "example-news.com"
        assert record.author_id == "news:example-news.com"
        assert record.author_handle == "Staff Reporter"

    def test_tracking_params_stripped_from_id_and_urls(self, news, news_payloads):
        record = news.to_record(news_payloads[0])
        assert "utm_medium" not in record.id
        assert record.urls == ["https://www.example-news.com/2024/05/01/ballot-claim-review"]

    def test_engagement_unmeasured(self, news, news_payloads):
        assert news.to_record(news_payloads[0]).engagement.likes is None

    def test_syndicated_copy_is_kept_and_near_duplicate_by_simhash(self, news, news_payloads):
        from ingest.normalize import hamming

        original = news.to_record(news_payloads[0])
        syndicated = news.to_record(news_payloads[1])
        assert original.id != syndicated.id  # both kept: breadth is signal
        assert hamming(original.simhash, syndicated.simhash) <= 12

    def test_atom_content_list_and_updated_parsed(self, news, news_payloads):
        record = news.to_record(news_payloads[2])
        assert "vaccine ingredients" in record.text
        assert record.timestamp.isoformat() == "2024-05-01T15:30:00+00:00"

    def test_undated_entry_dropped(self, news, news_payloads):
        assert news.to_record(news_payloads[3]) is None
        assert news.dropped[DropReason.MISSING_TIMESTAMP.value] == 1

    def test_media_extracted(self, news, news_payloads):
        assert news.to_record(news_payloads[0]).media_urls == [
            "https://www.example-news.com/img/ballots.jpg"
        ]

    def test_newsapi_article_keeps_description_and_truncated_content(self, news, news_payloads):
        record = news.to_record(news_payloads[4])
        assert record.raw["api"] == "newsapi"
        assert "overstated the certainty" in record.text
        assert record.raw["content_truncated"].endswith("[+1423 chars]")
        assert record.source_detail == "example-health.org"

    def test_newsapi_absent_key_degrades_to_rss(self, news, caplog):
        # Acceptance criterion: the pipeline runs to completion without the key.
        assert news.settings.newsapi_key is None
        assert list(news._fetch_newsapi({"enabled": True})) == []


# --- youtube --------------------------------------------------------------


@pytest.fixture
def youtube(settings, store):
    from ingest.sources.youtube import YouTubeSource

    return YouTubeSource(settings=settings, store=store)


@pytest.fixture
def ytdlp_recorded(fixtures_dir):
    """Two real yt-dlp info dicts and the caption bodies their tracks point at."""
    return load_fixture(fixtures_dir, "youtube_ytdlp_info.json")


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: answers from recorded data, never the network."""

    def __init__(self, searches=None, infos=None, captions=None, fail=()):
        self.searches = searches or {}
        self.infos = infos or {}
        self.captions = captions or {}
        self.fail = set(fail)
        self.requests: list[str] = []

    def extract_info(self, url, download=False):
        self.requests.append(url)
        if url in self.fail:
            raise RuntimeError("Sign in to confirm you are not a bot")
        if url.startswith(("ytsearch", "ytsearchdate")):
            return {"entries": [{"id": vid} for vid in self.searches.get(url, [])]}
        return self.infos[url.rsplit("=", 1)[1]]

    def urlopen(self, url):
        body = self.captions[url]
        if isinstance(body, Exception):
            raise body

        class _Response:
            def read(self_inner):
                return body.encode("utf-8")

        return _Response()


def _recorded(ytdlp_recorded, index):
    return ytdlp_recorded[index]["info"], ytdlp_recorded[index]["captions"]


def _payload(youtube, ytdlp_recorded, index):
    info, captions = _recorded(ytdlp_recorded, index)
    return youtube._payload(FakeYDL(captions=captions), info, {})


class TestYouTubeSubtitleSelection:
    def _info(self, ytdlp_recorded, **changes):
        info, _ = _recorded(ytdlp_recorded, 0)
        return {**info, **changes}

    def test_manual_track_in_the_video_language_wins_over_auto(self, ytdlp_recorded):
        from ingest.sources.youtube import select_subtitle_track

        lang, kind, fmt = select_subtitle_track(self._info(ytdlp_recorded))
        assert (lang, kind) == ("en", "manual")
        assert fmt["ext"] == "json3"  # structured format preferred over vtt

    def test_auto_original_is_used_when_there_is_no_manual_track(self, ytdlp_recorded):
        from ingest.sources.youtube import select_subtitle_track

        lang, kind, _ = select_subtitle_track(self._info(ytdlp_recorded, subtitles={}))
        assert (lang, kind) == ("en-orig", "auto")

    def test_auto_translation_into_another_language_is_never_used(self, ytdlp_recorded):
        from ingest.sources.youtube import select_subtitle_track

        info = self._info(ytdlp_recorded, subtitles={}, language="fr")
        assert select_subtitle_track(info) is None

    def test_regional_language_code_matches_its_base_track(self, ytdlp_recorded):
        from ingest.sources.youtube import select_subtitle_track

        info = self._info(ytdlp_recorded, language="en-US")
        assert select_subtitle_track(info)[0] == "en"

    def test_unknown_language_with_many_orig_tracks_is_not_guessed(self, ytdlp_recorded):
        from ingest.sources.youtube import select_subtitle_track

        info = self._info(ytdlp_recorded, subtitles={}, language=None)
        assert select_subtitle_track(info) is None

    def test_unknown_language_with_a_single_manual_track_uses_it(self, ytdlp_recorded):
        from ingest.sources.youtube import select_subtitle_track

        info = self._info(ytdlp_recorded, language=None)
        assert select_subtitle_track(info)[:2] == ("en", "manual")

    def test_no_tracks_at_all(self):
        from ingest.sources.youtube import select_subtitle_track

        assert select_subtitle_track({"language": "en"}) is None


class TestYouTubeSubtitleParsing:
    def test_rolling_auto_captions_are_deduplicated(self, ytdlp_recorded):
        from ingest.sources.youtube import parse_vtt

        _, captions = _recorded(ytdlp_recorded, 0)
        text = parse_vtt(captions["https://caption.invalid/NPyRRJodYw4/auto/en-orig.vtt"])
        lines = text.splitlines()
        assert lines[0] == "Subscribe to Factly now so you don't"
        assert len(lines) == len(set(lines))
        assert "<" not in text and "00:00" not in text

    def test_vtt_entities_and_nbsp_are_cleaned(self, ytdlp_recorded):
        from ingest.sources.youtube import parse_vtt

        _, captions = _recorded(ytdlp_recorded, 0)
        text = parse_vtt(captions["https://caption.invalid/NPyRRJodYw4/manual/en.vtt"])
        assert "&nbsp;" not in text and "\xa0" not in text
        assert text.startswith("On 1 November 2026, the INDIA bloc organised a")

    def test_json3_and_vtt_agree_on_a_manual_track(self, ytdlp_recorded):
        from ingest.sources.youtube import parse_json3, parse_vtt

        _, captions = _recorded(ytdlp_recorded, 0)
        from_json = parse_json3(captions["https://caption.invalid/NPyRRJodYw4/manual/en.json3"])
        from_vtt = parse_vtt(captions["https://caption.invalid/NPyRRJodYw4/manual/en.vtt"])
        assert from_json.splitlines()[0] == from_vtt.splitlines()[0]

    def test_hindi_text_survives(self, ytdlp_recorded):
        from ingest.sources.youtube import parse_vtt

        _, captions = _recorded(ytdlp_recorded, 0)
        text = parse_vtt(captions["https://caption.invalid/NPyRRJodYw4/auto/hi-orig.vtt"])
        assert "सब्सक्राइब" in text

    def test_json3_with_no_segments_is_empty(self):
        from ingest.sources.youtube import parse_json3

        assert parse_json3('{"events": [{"tStartMs": 0}]}') == ""


class TestYouTubeMapping:
    def test_video_record_text_carries_title_description_and_transcript(
        self, youtube, ytdlp_recorded
    ):
        record = youtube.to_record(_payload(youtube, ytdlp_recorded, 0))
        assert record.content_type == "video"
        assert record.source_detail == "UCpi2S8wW4xLlUCVryhyBtsA"
        assert record.author_handle == "Factly"
        assert record.text.startswith("FACT CHECK:")
        assert "INDIA bloc organised a" in record.text  # transcript
        assert record.lang == "en"

    def test_manual_subtitle_provenance_is_recorded_in_raw(self, youtube, ytdlp_recorded):
        raw = youtube.to_record(_payload(youtube, ytdlp_recorded, 0)).raw
        assert raw["subtitle_kind"] == "manual"
        assert raw["subtitle_lang"] == "en"

    def test_auto_subtitle_provenance_is_recorded_in_raw(self, youtube, ytdlp_recorded):
        record = youtube.to_record(_payload(youtube, ytdlp_recorded, 1))
        assert record.raw["subtitle_kind"] == "auto"
        assert record.raw["subtitle_lang"] == "en-orig"
        assert "sorry guys" in record.text

    def test_video_without_subtitles_is_kept_and_counted(self, youtube, ytdlp_recorded):
        info, _ = _recorded(ytdlp_recorded, 1)
        info = {**info, "automatic_captions": {}, "subtitles": {}}
        record = youtube.to_record(youtube._payload(FakeYDL(), info, {}))
        assert record is not None
        assert record.raw["subtitle_kind"] == "none"
        assert record.raw["subtitle_lang"] is None
        assert youtube.flags["no_subtitles"] == 1
        assert youtube.flags["with_subtitles"] == 0

    def test_records_with_subtitles_are_counted_by_kind(self, youtube, ytdlp_recorded):
        youtube.to_record(_payload(youtube, ytdlp_recorded, 0))
        youtube.to_record(_payload(youtube, ytdlp_recorded, 1))
        assert youtube.flags["with_subtitles"] == 2
        assert youtube.flags["subtitles_manual"] == 1
        assert youtube.flags["subtitles_auto"] == 1

    def test_caption_download_failure_is_counted_not_fatal(self, youtube, ytdlp_recorded):
        info, captions = _recorded(ytdlp_recorded, 0)
        broken = {url: OSError("timeout") for url in captions}
        payload = youtube._payload(FakeYDL(captions=broken), info, {})
        record = youtube.to_record(payload)
        assert record.raw["subtitle_kind"] == "none"
        assert youtube.flags["subtitle_fetch_failed"] == 1
        assert youtube.flags["no_subtitles"] == 1

    def test_engagement_mapping_including_views(self, youtube, ytdlp_recorded):
        engagement = youtube.to_record(_payload(youtube, ytdlp_recorded, 0)).engagement
        assert engagement.views == 59155
        assert engagement.likes == 308
        assert engagement.replies == 96
        assert engagement.shares is None  # YouTube exposes no share count

    def test_none_statistics_stay_none_not_zero(self, youtube, ytdlp_recorded):
        info, _ = _recorded(ytdlp_recorded, 0)
        info = {**info, "like_count": None, "comment_count": None, "view_count": 5120}
        engagement = youtube.to_record(youtube._payload(FakeYDL(), info, {})).engagement
        assert engagement.views == 5120
        assert engagement.likes is None and engagement.replies is None

    def test_measured_zero_stays_zero(self, youtube, ytdlp_recorded):
        info, _ = _recorded(ytdlp_recorded, 0)
        info = {**info, "like_count": 0, "comment_count": 0, "view_count": 0}
        engagement = youtube.to_record(youtube._payload(FakeYDL(), info, {})).engagement
        assert (engagement.views, engagement.likes, engagement.replies) == (0, 0, 0)

    def test_media_url_is_the_watchable_video(self, youtube, ytdlp_recorded):
        record = youtube.to_record(_payload(youtube, ytdlp_recorded, 0))
        assert record.media_urls == ["https://www.youtube.com/watch?v=NPyRRJodYw4"]

    def test_video_is_its_own_conversation_root(self, youtube, ytdlp_recorded):
        record = youtube.to_record(_payload(youtube, ytdlp_recorded, 0))
        assert record.parent_id is None
        assert record.conversation_id == "youtube:NPyRRJodYw4"

    def test_timestamp_is_utc_from_epoch(self, youtube, ytdlp_recorded):
        record = youtube.to_record(_payload(youtube, ytdlp_recorded, 0))
        assert record.timestamp.utcoffset().total_seconds() == 0
        assert record.timestamp.year == 2026

    def test_upload_date_is_the_fallback_timestamp(self, youtube, ytdlp_recorded):
        info, _ = _recorded(ytdlp_recorded, 0)
        payload = youtube._payload(FakeYDL(), {**info, "timestamp": None}, {})
        assert youtube.to_record(payload).timestamp.strftime("%Y-%m-%d") == "2026-11-03"

    def test_missing_timestamp_is_dropped_with_a_reason(self, youtube, ytdlp_recorded):
        info, _ = _recorded(ytdlp_recorded, 0)
        payload = youtube._payload(FakeYDL(), {**info, "timestamp": None, "upload_date": None}, {})
        assert youtube.to_record(payload) is None
        assert youtube.dropped[DropReason.MISSING_TIMESTAMP.value] == 1

    def test_empty_video_is_dropped_with_a_reason(self, youtube):
        payload = {"_kind": "video", "id": "x", "timestamp": 1700000000, "title": ""}
        assert youtube.to_record(payload) is None
        assert youtube.dropped[DropReason.EMPTY_TEXT.value] == 1

    def test_top_level_comment_parents_to_the_video(self, youtube, ytdlp_recorded):
        info, _ = _recorded(ytdlp_recorded, 0)
        payload = next(youtube._comment_payloads({"id": info["id"], "channel_id": "UCx"}, info, 5))
        record = youtube.to_record(payload)
        assert record.content_type == "video_comment"
        assert record.parent_id == "youtube:NPyRRJodYw4"
        assert record.conversation_id == "youtube:NPyRRJodYw4"
        assert record.engagement.views is None and record.engagement.replies is None

    def test_reply_parents_to_the_comment(self, youtube):
        payload = {
            "_kind": "video_comment",
            "_video_id": "vid",
            "id": "c2",
            "parent": "c1",
            "text": "agreed",
            "author_id": "UCa",
            "like_count": 0,
            "timestamp": 1700000000,
        }
        record = youtube.to_record(payload)
        assert record.parent_id == "youtube:c1"
        assert record.engagement.likes == 0

    def test_comment_like_none_stays_none(self, youtube):
        payload = {
            "_kind": "video_comment",
            "_video_id": "vid",
            "id": "c3",
            "parent": "root",
            "text": "hello",
            "timestamp": 1700000000,
            "like_count": None,
        }
        assert youtube.to_record(payload).engagement.likes is None

    def test_empty_comment_dropped(self, youtube):
        payload = {"_kind": "video_comment", "id": "c4", "text": " ", "timestamp": 1700000000}
        assert youtube.to_record(payload) is None
        assert youtube.dropped[DropReason.EMPTY_TEXT.value] == 1

    def test_comments_disabled_is_counted(self, youtube):
        assert list(youtube._comment_payloads({"id": "v"}, {"comments": None}, 10)) == []
        assert youtube.flags["comments_unavailable"] == 1

    def test_author_rollup(self, youtube, ytdlp_recorded):
        payload = _payload(youtube, ytdlp_recorded, 0)
        author = youtube.to_author(payload, youtube.to_record(payload))
        assert author.handle == "Factly"
        assert author.raw["channel_id"] == "UCpi2S8wW4xLlUCVryhyBtsA"


class TestYouTubeDiscoveryAndFetch:
    """Search and fetch against a fake yt-dlp. No network, no key."""

    def _run(self, youtube, monkeypatch, ytdlp_recorded, config, *, fail=(), searches=None):
        infos = {e["info"]["id"]: e["info"] for e in ytdlp_recorded}
        captions = {}
        for e in ytdlp_recorded:
            captions.update(e["captions"])
        fake = FakeYDL(
            searches=searches or {"ytsearch5:q one": ["NPyRRJodYw4", "sPzKg_dPcg8", "gone0000000"]},
            infos=infos,
            captions=captions,
            fail=fail,
        )
        monkeypatch.setattr(youtube, "_client", lambda *a, **k: fake)
        monkeypatch.setattr("ingest.sources.youtube.sources_config", lambda: {"youtube": config})
        youtube.options = {"queries": ["q one"]}
        return fake, list(youtube.fetch())

    def _config(self, **discovery):
        base = {"results_per_search": 5, "published_within_days": 100000}
        return {"discovery": {**base, **discovery}, "comments": {"enabled": False}}

    def test_fetch_yields_videos_with_subtitles(self, youtube, monkeypatch, ytdlp_recorded):
        _, payloads = self._run(youtube, monkeypatch, ytdlp_recorded, self._config())
        assert [p["id"] for p in payloads] == ["NPyRRJodYw4", "sPzKg_dPcg8"]
        assert payloads[0]["_subtitle"]["kind"] == "manual"
        assert payloads[1]["_subtitle"]["kind"] == "auto"

    def test_failed_extraction_is_counted_and_skipped(self, youtube, monkeypatch, ytdlp_recorded):
        fail = {"https://www.youtube.com/watch?v=sPzKg_dPcg8"}
        _, payloads = self._run(youtube, monkeypatch, ytdlp_recorded, self._config(), fail=fail)
        assert [p["id"] for p in payloads] == ["NPyRRJodYw4"]
        # gone0000000 is not in the fake's infos either, which raises KeyError
        assert youtube.flags["extract_failed"] == 2

    def test_old_videos_are_counted_as_outside_the_window(
        self, youtube, monkeypatch, ytdlp_recorded
    ):
        # The second recorded video was published in 2022.
        _, payloads = self._run(
            youtube, monkeypatch, ytdlp_recorded, self._config(published_within_days=30)
        )
        assert youtube.flags["outside_published_window"] >= 1
        assert "sPzKg_dPcg8" not in [p["id"] for p in payloads]

    def test_failed_search_is_counted(self, youtube, monkeypatch, ytdlp_recorded):
        fake = FakeYDL(fail={"ytsearch5:q one"})
        monkeypatch.setattr(youtube, "_client", lambda *a, **k: fake)
        monkeypatch.setattr(
            "ingest.sources.youtube.sources_config", lambda: {"youtube": self._config()}
        )
        youtube.options = {"queries": ["q one"]}
        assert list(youtube.fetch()) == []
        assert youtube.flags["search_failed"] == 1

    def test_date_order_uses_the_date_sorted_search(self, youtube, monkeypatch, ytdlp_recorded):
        fake, _ = self._run(
            youtube,
            monkeypatch,
            ytdlp_recorded,
            self._config(order="date"),
            searches={"ytsearchdate5:q one": []},
        )
        assert fake.requests == ["ytsearchdate5:q one"]

    def test_the_devanagari_query_reaches_search_unmangled(
        self, youtube, monkeypatch, ytdlp_recorded
    ):
        query = "वोट चोरी दावा फैक्ट चेक"
        fake = FakeYDL(searches={f"ytsearch5:{query}": []})
        monkeypatch.setattr(youtube, "_client", lambda *a, **k: fake)
        monkeypatch.setattr(
            "ingest.sources.youtube.sources_config", lambda: {"youtube": self._config()}
        )
        youtube.options = {"queries": [query]}
        list(youtube.fetch())
        assert fake.requests == [f"ytsearch5:{query}"]

    def test_duplicate_ids_across_queries_are_extracted_once(
        self, youtube, monkeypatch, ytdlp_recorded
    ):
        searches = {"ytsearch5:a": ["NPyRRJodYw4"], "ytsearch5:b": ["NPyRRJodYw4"]}
        infos = {e["info"]["id"]: e["info"] for e in ytdlp_recorded}
        captions = {}
        for e in ytdlp_recorded:
            captions.update(e["captions"])
        fake = FakeYDL(searches=searches, infos=infos, captions=captions)
        monkeypatch.setattr(youtube, "_client", lambda *a, **k: fake)
        monkeypatch.setattr(
            "ingest.sources.youtube.sources_config", lambda: {"youtube": self._config()}
        )
        youtube.options = {"queries": ["a", "b"]}
        assert len(list(youtube.fetch())) == 1

    def test_run_counts_subtitled_records_in_the_result(
        self, youtube, monkeypatch, ytdlp_recorded
    ):
        self._patch_for_run(youtube, monkeypatch, ytdlp_recorded)
        result = youtube.run()
        assert result.ok and result.written == 2
        assert result.flags["with_subtitles"] == 2
        assert result.flags["subtitles_manual"] == 1

    def _patch_for_run(self, youtube, monkeypatch, ytdlp_recorded):
        infos = {e["info"]["id"]: e["info"] for e in ytdlp_recorded}
        captions = {}
        for e in ytdlp_recorded:
            captions.update(e["captions"])
        fake = FakeYDL(
            searches={"ytsearch5:q one": ["NPyRRJodYw4", "sPzKg_dPcg8"]},
            infos=infos,
            captions=captions,
        )
        monkeypatch.setattr(youtube, "_client", lambda *a, **k: fake)
        monkeypatch.setattr(
            "ingest.sources.youtube.sources_config", lambda: {"youtube": self._config()}
        )
        youtube.options = {"queries": ["q one"]}


class TestYouTubeClientOptions:
    """What is handed to yt-dlp. Region and relevance live in configs/sources.yaml."""

    def _opts(self, youtube, discovery, **kwargs):
        flat = kwargs.pop("flat", True)
        return youtube._ydl_options({"discovery": discovery}, flat=flat, **kwargs)

    def test_region_becomes_geo_bypass_country(self, youtube):
        assert self._opts(youtube, {"region_code": "in"})["geo_bypass_country"] == "IN"

    def test_absent_region_leaves_the_option_out(self, youtube):
        assert "geo_bypass_country" not in self._opts(youtube, {})

    def test_relevance_language_is_sent_only_when_set(self, youtube):
        opts = self._opts(youtube, {"relevance_language": "hi"})
        assert opts["extractor_args"]["youtube"]["lang"] == ["hi"]
        assert "extractor_args" not in self._opts(youtube, {"relevance_language": None})

    def test_comment_cap_is_a_split_extractor_argument(self, youtube):
        opts = self._opts(youtube, {}, flat=False, comments=40)
        assert opts["getcomments"] is True
        assert opts["extractor_args"]["youtube"]["max_comments"][0] == "40"

    def test_no_media_is_ever_downloaded(self, youtube):
        assert self._opts(youtube, {})["skip_download"] is True

    def test_source_runs_without_any_credential(self, youtube):
        assert youtube.settings.has_credentials("youtube") is True


# --- reddit / kaggle ------------------------------------------------------


@pytest.fixture
def kaggle(settings, store):
    from ingest.sources.reddit_kaggle import RedditKaggleSource

    return RedditKaggleSource(settings=settings, store=store)


@pytest.fixture
def kaggle_rows(fixtures_dir):
    return load_fixture(fixtures_dir, "reddit_kaggle_rows.json")


class TestRedditKaggleMapping:
    def test_column_map_drives_the_mapping(self, kaggle, kaggle_rows):
        record = kaggle.to_record(kaggle_rows[0])
        assert record.id == "reddit:9x1abc"
        assert record.author_id == "reddit:concerned_citizen"
        assert record.source_detail == "conspiracy"
        assert record.engagement.likes == 1284

    def test_title_and_body_combined(self, kaggle, kaggle_rows):
        record = kaggle.to_record(kaggle_rows[0])
        assert record.text.startswith("Six outlets, identical phrasing")
        assert "Screenshots in the comments" in record.text

    def test_epoch_string_becomes_utc(self, kaggle, kaggle_rows):
        assert kaggle.to_record(kaggle_rows[0]).timestamp.isoformat() == "2024-05-01T12:00:00+00:00"

    def test_absent_threading_is_null_not_invented(self, kaggle, kaggle_rows):
        # This is why README says Kaggle Reddit data cannot support the
        # coordination graph: there is nothing to build edges from.
        record = kaggle.to_record(kaggle_rows[0])
        assert record.parent_id is None and record.conversation_id is None
        assert record.raw["threading_available"] is False
        assert kaggle.flags["no_threading_in_dataset"] == 1

    def test_deleted_author_sentinel(self, kaggle, kaggle_rows):
        record = kaggle.to_record(kaggle_rows[1])
        assert record.author_is_deleted and record.author_handle is None
        assert record.text == "Archive of the six versions"

    def test_second_dataset_with_different_columns(self, kaggle, kaggle_rows):
        record = kaggle.to_record(kaggle_rows[2])
        assert record.content_type == "comment"
        assert record.engagement.likes == -12
        assert record.timestamp.isoformat() == "2024-05-01T13:00:00+00:00"

    def test_threading_used_when_the_dataset_has_it(self, kaggle, kaggle_rows):
        record = kaggle.to_record(kaggle_rows[2])
        assert record.parent_id == "reddit:t3_9x1abc"
        assert record.raw["threading_available"] is True

    def test_removed_body_without_title_dropped(self, kaggle, kaggle_rows):
        assert kaggle.to_record(kaggle_rows[3]) is None
        assert kaggle.dropped[DropReason.DELETED_TEXT.value] == 1

    def test_unparseable_timestamp_dropped(self, kaggle, kaggle_rows):
        assert kaggle.to_record(kaggle_rows[4]) is None
        assert kaggle.dropped[DropReason.MISSING_TIMESTAMP.value] == 1

    def test_original_columns_preserved_in_raw(self, kaggle, kaggle_rows):
        raw = kaggle.to_record(kaggle_rows[0]).raw
        assert raw["dataset"] == "example-owner/reddit-posts"
        assert raw["columns"]["subreddit"] == "conspiracy"

    def test_local_path_does_not_require_kaggle_credentials(self, settings, store, tmp_path):
        # The Academic Torrents workflow is "point at an already-downloaded
        # directory". Gating it behind a Kaggle key makes that impossible on a
        # machine with no Kaggle account.
        from ingest.sources.reddit_kaggle import RedditKaggleSource

        dump = tmp_path / "dump"
        dump.mkdir()
        (dump / "posts.csv").write_text(
            "id,title,author,created_utc\nk1,A local post about a claim,someone,1714564800\n",
            encoding="utf-8",
        )
        spec = {
            "slug": "local-dump",
            "timestamp_format": "epoch",
            "column_map": {
                "native_id": "id",
                "title": "title",
                "author": "author",
                "timestamp": "created_utc",
            },
        }
        source = RedditKaggleSource(settings=settings, store=store, path=str(dump), spec=spec)
        source.preflight()  # must not raise
        result = source.run()
        assert result.ok and result.written == 1

    def test_download_path_still_requires_credentials(self, kaggle):
        from ingest.sources.base import SourceUnavailable

        with pytest.raises(SourceUnavailable, match="credentials absent"):
            kaggle.preflight()

    def test_no_configured_datasets_warns_and_yields_nothing(self, kaggle):
        assert list(kaggle.fetch()) == []

    @pytest.mark.parametrize(
        "value,fmt,expected",
        [
            ("1714564800", "epoch", "2024-05-01T12:00:00+00:00"),
            (1714564800000, "epoch", "2024-05-01T12:00:00+00:00"),
            ("2024-05-01T12:00:00Z", "iso", "2024-05-01T12:00:00+00:00"),
            ("2024-05-01 12:00:00", "%Y-%m-%d %H:%M:%S", "2024-05-01T12:00:00+00:00"),
            ("", "epoch", None),
            ("garbage", "iso", None),
        ],
    )
    def test_timestamp_formats(self, value, fmt, expected):
        from ingest.sources.reddit_kaggle import _parse_timestamp

        parsed = _parse_timestamp(value, fmt)
        assert (parsed.isoformat() if parsed else None) == expected


# --- registry -------------------------------------------------------------


class TestRegistry:
    def test_every_registered_source_resolves(self):
        from ingest.sources import REGISTRY, get_source_class
        from ingest.sources.base import BaseSource

        for name in REGISTRY:
            cls = get_source_class(name)
            assert issubclass(cls, BaseSource)
            assert cls.name == name

    def test_unknown_source_names_are_helpful(self):
        from ingest.sources import get_source_class

        with pytest.raises(KeyError, match="unknown source"):
            get_source_class("twitter")


class TestGdeltQueryConstruction:
    """The DOC query form, pinned offline after the live API rejected the docs' version."""

    def test_single_language_is_passed_as_a_string_not_a_list(self):
        from ingest.sources.gdelt import _language_filter

        # A one-element list makes gdeltdoc emit "(sourcelang:English)", and
        # GDELT rejects parentheses around a non-OR'd term.
        assert _language_filter(["English"]) == "English"
        assert _language_filter("English") == "English"

    def test_multiple_languages_stay_a_list(self):
        from ingest.sources.gdelt import _language_filter

        assert _language_filter(["English", "Spanish"]) == ["English", "Spanish"]

    def test_absent_language_is_none(self):
        from ingest.sources.gdelt import _language_filter

        assert _language_filter([]) is None
        assert _language_filter(None) is None

    def test_emitted_query_string_is_valid_for_each_form(self):
        gdeltdoc = pytest.importorskip("gdeltdoc")
        from ingest.sources.gdelt import _language_filter

        for languages in (["English"], ["English", "Spanish"], []):
            query = gdeltdoc.Filters(
                keyword=["ballot fraud"],
                start_date="2024-05-01",
                end_date="2024-05-08",
                num_records=10,
                language=_language_filter(languages),
            ).query_string
            # Never a parenthesized single term: that is the rejected form.
            assert "(sourcelang:English)" not in query

    def test_country_takes_the_same_arity_treatment_as_language(self):
        from ingest.sources.gdelt import _country_filter

        # Same gdeltdoc rendering rule, same trap: a one-element list becomes
        # "(sourcecountry:IN)" and GDELT refuses the parentheses.
        assert _country_filter(["IN"]) == "IN"
        assert _country_filter("IN") == "IN"
        assert _country_filter(["IN", "PK"]) == ["IN", "PK"]
        assert _country_filter([]) is None
        assert _country_filter(None) is None

    def test_country_scoping_reaches_the_emitted_query(self):
        gdeltdoc = pytest.importorskip("gdeltdoc")
        from ingest.sources.gdelt import _country_filter, _language_filter

        query = gdeltdoc.Filters(
            keyword=["voter roll claim"],
            start_date="2026-10-01",
            end_date="2026-10-08",
            num_records=10,
            language=_language_filter(["English", "Hindi"]),
            country=_country_filter(["IN"]),
        ).query_string
        assert "sourcecountry:IN" in query
        assert "(sourcecountry:IN)" not in query

    def test_an_absent_country_leaves_the_query_global(self):
        """A missing `countries` key must not silently become a filter.

        The config default is the global corpus, so a typo in the key name
        degrades to "too much data" rather than to "zero rows", which is the
        failure direction that gets noticed.
        """
        gdeltdoc = pytest.importorskip("gdeltdoc")
        from ingest.sources.gdelt import _country_filter

        query = gdeltdoc.Filters(
            keyword=["voter roll claim"],
            start_date="2026-10-01",
            end_date="2026-10-08",
            num_records=10,
            country=_country_filter({}.get("countries")),
        ).query_string
        assert "sourcecountry" not in query

    def test_a_bad_country_code_warns_instead_of_silently_emptying_the_corpus(self, caplog):
        """gdeltdoc validates nothing here, and GDELT answers junk with zero rows.

        An empty corpus that looks like a quiet news week is the failure mode
        this project cannot afford, so the two typos anyone actually makes --
        the 3-letter ISO code and the country's name -- have to be loud.
        """
        import logging

        from ingest.sources.gdelt import _country_filter

        for bad in ("IND", "India", "in"):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="ingest.gdelt"):
                # Still passed through: the adapter's job is to warn, not to
                # overrule a config it cannot fully validate.
                assert _country_filter([bad]) == bad
            assert "not a 2-letter FIPS code" in caplog.text, bad

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="ingest.gdelt"):
            assert _country_filter(["IN"]) == "IN"
        assert caplog.text == ""

    def test_indian_language_names_resolve_to_iso_codes(self):
        """GDELT reports language *names*; an unmapped name means a null `lang`."""
        from ingest.sources.gdelt import _language_code

        for name, code in [
            ("Hindi", "hi"),
            ("Bengali", "bn"),
            ("Tamil", "ta"),
            ("Telugu", "te"),
            ("Marathi", "mr"),
            ("Gujarati", "gu"),
            ("Kannada", "kn"),
            ("Malayalam", "ml"),
            ("Punjabi", "pa"),
            ("Urdu", "ur"),
            ("Assamese", "as"),
            ("Nepali", "ne"),
        ]:
            assert _language_code(name) == code, name
        # GDELT spells Odia both ways depending on the feed.
        assert _language_code("Oriya") == _language_code("Odia") == "or"


class TestGdeltArchiveReading:
    """GKG fields are enormous and GDELT drops are occasionally malformed."""

    def _zip(self, tmp_path, rows: list[list[str]]):
        import zipfile

        path = tmp_path / "20240501120000.gkg.csv.zip"
        body = "\n".join("\t".join(cell for cell in row) for row in rows)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("20240501120000.gkg.csv", body)
        return path

    def test_huge_field_does_not_blow_the_csv_limit(self, tmp_path):
        # GCAM/V2Themes run to hundreds of KB; the csv default is 128KB.
        from ingest.sources.gdelt import GKG_COLUMNS, _read_zipped_csv

        row = [""] * len(GKG_COLUMNS)
        row[0] = "20240501120000-1"
        row[1] = "20240501120000"
        row[4] = "https://example-news.com/story"
        row[17] = "x" * 300_000  # GCAM
        rows = list(_read_zipped_csv(self._zip(tmp_path, [row]), GKG_COLUMNS))
        assert len(rows) == 1
        assert len(rows[0]["GCAM"]) == 300_000

    def test_short_rows_are_padded_not_dropped(self, tmp_path):
        from ingest.sources.gdelt import GKG_COLUMNS, _read_zipped_csv

        rows = list(_read_zipped_csv(self._zip(tmp_path, [["id1", "20240501120000"]]), GKG_COLUMNS))
        assert rows[0]["GKGRECORDID"] == "id1"
        assert rows[0]["Extras"] == ""

    def test_corrupt_archive_raises_source_unavailable_not_a_raw_error(self, tmp_path):
        from ingest.sources.base import SourceUnavailable
        from ingest.sources.gdelt import GKG_COLUMNS, _read_zipped_csv

        bad = tmp_path / "broken.zip"
        bad.write_bytes(b"not a zip at all")
        with pytest.raises(SourceUnavailable, match="could not read GDELT archive"):
            list(_read_zipped_csv(bad, GKG_COLUMNS))
