"""Normalization is pure, so it is cheap to test exhaustively. Do so."""

from __future__ import annotations

import pytest

from ingest.normalize import (
    build_text_fields,
    canonicalize_url,
    clean_text,
    detect_lang,
    extract_hashtags,
    extract_html_links,
    extract_mentions,
    extract_urls,
    hamming,
    is_deleted_text,
    is_shortlink,
    resolve_domain,
    resolve_domains,
    simhash,
    strip_html,
)


class TestStripHtml:
    def test_mastodon_status_markup(self):
        html = '<p>Hello <a href="https://example.com/a">link</a> world</p>'
        assert strip_html(html) == "Hello link world\n"

    def test_block_boundaries_do_not_glue_words(self):
        assert "onetwo" not in strip_html("<p>one</p><p>two</p>")
        assert "one\ntwo" in strip_html("<p>one</p><p>two</p>")

    def test_br_becomes_newline(self):
        assert strip_html("a<br>b") == "a\nb"

    def test_escaped_entities(self):
        assert strip_html("Tom &amp; Jerry &#39;s") == "Tom & Jerry 's"

    def test_double_escaped_rss(self):
        assert strip_html("&amp;lt;b&amp;gt;bold&amp;lt;/b&amp;gt;") == "<b>bold</b>"

    def test_plain_text_passes_through_untouched(self):
        assert strip_html("no markup here") == "no markup here"

    @pytest.mark.parametrize("value", [None, ""])
    def test_empty(self, value):
        assert strip_html(value) == ""


class TestCleanText:
    def test_collapses_horizontal_whitespace_only(self):
        assert clean_text("a   \t b\n\nc") == "a b\n\nc"

    def test_caps_blank_line_runs(self):
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_zero_width_characters(self):
        assert clean_text("br​eak﻿ing") == "breaking"

    def test_nfkc_normalizes_fullwidth(self):
        assert clean_text("ＢＲＥＡＫＩＮＧ") == "BREAKING"

    def test_preserves_case_and_punctuation(self):
        # Phase 2 embeds this with transformers: surface form must survive.
        assert clean_text("BREAKING!!! They don't want you to know.") == (
            "BREAKING!!! They don't want you to know."
        )

    def test_deleted_markers(self):
        assert is_deleted_text("[deleted]")
        assert is_deleted_text("  [REMOVED] ")
        assert not is_deleted_text("the post was deleted by a mod")
        assert not is_deleted_text(None)


class TestUrls:
    def test_extracts_from_prose(self):
        urls = extract_urls("see https://example.com/a and http://b.org/x?y=1")
        assert urls == ["https://example.com/a", "http://b.org/x?y=1"]

    def test_trailing_sentence_punctuation_trimmed(self):
        assert extract_urls("read https://example.com/a.") == ["https://example.com/a"]

    def test_structured_entities_preferred_and_merged(self):
        raw = {"urls": [{"url": "https://structured.example/1"}]}
        urls = extract_urls("also https://inline.example/2", raw)
        assert urls[0] == "https://structured.example/1"
        assert "https://inline.example/2" in urls

    def test_deduped(self):
        assert extract_urls("https://a.com/x https://a.com/x") == ["https://a.com/x"]

    def test_tracking_params_and_fragment_dropped(self):
        assert (
            canonicalize_url("https://Ex.com/a?utm_source=x&id=7#frag") == "https://ex.com/a?id=7"
        )

    def test_canonicalization_makes_campaign_links_collapse(self):
        a = canonicalize_url("https://news.example/story?utm_campaign=a&fbclid=1")
        b = canonicalize_url("https://news.example/story?utm_campaign=b")
        assert a == b == "https://news.example/story"

    @pytest.mark.parametrize(
        "url,domain",
        [
            ("https://www.bbc.co.uk/news/x", "bbc.co.uk"),
            ("http://News.BBC.co.uk/x", "bbc.co.uk"),
            ("https://sub.domain.example.com", "example.com"),
            ("example.org/path", "example.org"),
            ("mailto:a@b.com", None),
            ("https://192.168.0.1/x", None),
            (None, None),
        ],
    )
    def test_resolve_domain(self, url, domain):
        assert resolve_domain(url) == domain

    def test_resolve_domains_dedupes_preserving_order(self):
        urls = ["https://a.com/1", "https://www.a.com/2", "https://b.com"]
        assert resolve_domains(urls) == ["a.com", "b.com"]

    def test_shortlink_detection(self):
        assert is_shortlink("https://bit.ly/abc")
        assert not is_shortlink("https://nytimes.com/abc")


class TestHtmlLinks:
    def test_href_recovered_when_display_text_is_truncated(self):
        # This is exactly how Mastodon renders links: the visible text is
        # elided, so only the href holds the real destination.
        html = (
            '<p><a href="https://example.com/very/long/path/story-2024" rel="nofollow">'
            '<span class="invisible">https://</span><span class="ellipsis">example.com/very</span>'
            '<span class="invisible">/long/path/story-2024</span></a></p>'
        )
        assert extract_html_links(html) == ["https://example.com/very/long/path/story-2024"]
        assert build_text_fields(html, is_html=True)["domains"] == ["example.com"]

    def test_hashtag_and_mention_anchors_are_not_outbound_links(self):
        html = (
            '<p><a href="https://mastodon.social/tags/election" class="mention hashtag" rel="tag">'
            "#<span>election</span></a> "
            '<a href="https://instance.tld/@user" class="u-url mention">@<span>user</span></a> '
            '<a href="https://news.example/story">source</a></p>'
        )
        assert extract_html_links(html) == ["https://news.example/story"]

    def test_no_anchors(self):
        assert extract_html_links("<p>plain</p>") == []
        assert extract_html_links(None) == []


class TestEntities:
    def test_hashtags_lowercased_without_hash(self):
        assert extract_hashtags("#Election #FRAUD now") == ["election", "fraud"]

    def test_hashtag_ignores_html_entities_and_anchors(self):
        assert extract_hashtags("color &#35; and https://x.com/a#section") == []

    def test_structured_hashtags_merge(self):
        tags = extract_hashtags("#b", [{"name": "A"}])
        assert tags == ["a", "b"]

    def test_mentions_keep_fediverse_instance(self):
        assert extract_mentions("hi @user@mastodon.social and @local") == [
            "user@mastodon.social",
            "local",
        ]

    def test_structured_mentions_use_acct(self):
        assert extract_mentions("", [{"acct": "someone@instance.tld"}]) == ["someone@instance.tld"]


class TestLangDetect:
    def test_returns_none_for_short_text(self):
        assert detect_lang("hi") is None
        assert detect_lang("") is None
        assert detect_lang(None) is None

    def test_detects_english(self):
        assert detect_lang("The quick brown fox jumps over the lazy dog every morning.") == "en"

    def test_detects_non_english(self):
        assert (
            detect_lang("Der schnelle braune Fuchs springt jeden Morgen über den faulen Hund.")
            == "de"
        )

    def test_deterministic_across_calls(self):
        text = "Este es un texto en español que debería detectarse de forma consistente."
        assert detect_lang(text) == detect_lang(text) == "es"

    def test_primary_subtag_only(self):
        code = detect_lang("这是一段足够长的中文文本，用于测试语言检测功能是否正常工作。")
        assert code == "zh"


class TestSimhash:
    def test_identical_text_identical_hash(self):
        text = "the vaccine microchip claim resurfaced again this week in three languages"
        assert simhash(text) == simhash(text)

    def test_fits_in_uint64(self):
        assert 0 <= simhash("some reasonably long piece of text here") < 2**64

    def test_empty_text_is_zero(self):
        assert simhash("") == 0
        assert simhash(None) == 0

    def test_near_duplicates_are_close(self):
        a = "Officials confirmed the ballots were counted twice in the county on Tuesday night"
        b = "Officials confirmed the ballots were counted twice in the county on Tuesday evening"
        assert hamming(simhash(a), simhash(b)) <= 8

    def test_unrelated_texts_are_far(self):
        a = "Officials confirmed the ballots were counted twice in the county on Tuesday"
        b = "A new species of deep sea jellyfish was described by marine biologists"
        assert hamming(simhash(a), simhash(b)) > 12

    def test_short_text_below_ngram_size_still_hashes(self):
        assert simhash("two words") != 0


class TestBuildTextFields:
    def test_html_pipeline_end_to_end(self):
        fields = build_text_fields(
            '<p>BREAKING: <a href="https://www.example.com/story?utm_source=x">proof</a> '
            "of the thing #Election @user@instance.tld</p>",
            is_html=True,
        )
        assert fields["text"].startswith("BREAKING: proof")
        assert fields["urls"] == ["https://www.example.com/story"]
        assert fields["domains"] == ["example.com"]
        assert fields["hashtags"] == ["election"]
        assert fields["mentions"] == ["user@instance.tld"]
        assert isinstance(fields["simhash"], int)

    def test_keys_match_schema_fields(self):
        from ingest.schema import Record

        assert set(build_text_fields("hello world")) <= set(Record.model_fields)
