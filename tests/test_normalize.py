"""Normalization is pure, so it is cheap to test exhaustively. Do so."""

from __future__ import annotations

import logging
import sys

import pytest

from ingest.normalize import (
    ROMANIZED_HINDI_MARKERS,
    SCRIPT_TO_LANG,
    build_text_fields,
    canonicalize_url,
    clean_text,
    detect_lang,
    dominant_script,
    extract_hashtags,
    extract_html_links,
    extract_mentions,
    extract_urls,
    hamming,
    html_parser_class,
    is_code_mixed,
    is_deleted_text,
    is_shortlink,
    looks_romanized_hindi,
    resolve_domain,
    resolve_domains,
    romanized_hindi_markers,
    script_profile,
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


class TestHtmlParserBackend:
    """The backend resolution itself, because its silent failure cost real data.

    Both call sites used to import selectolax's Modest backend inside a bare
    ``except Exception``. selectolax 1.0 removed that backend, so the import
    began raising ImportError and the project silently switched to regex
    handling -- no error, no log, and hashtag and mention anchors flowing into
    ``urls`` and ``domains`` for an unknown length of time.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """The resolved class is cached per process; don't leak it between tests."""
        from ingest import normalize

        saved = normalize._html_parser_cls
        normalize._html_parser_cls = False
        yield
        normalize._html_parser_cls = saved

    def test_a_real_parser_is_available_in_this_environment(self):
        """Guards the install. If this fails, every HTML number is degraded."""
        assert html_parser_class() is not None

    def test_lexbor_is_preferred(self):
        assert html_parser_class().__name__ == "LexborHTMLParser"

    def test_resolution_is_cached_not_repeated_per_record(self):
        assert html_parser_class() is html_parser_class()

    def _break_both_backends(self, monkeypatch):
        # A None entry in sys.modules makes `import x` raise ImportError.
        monkeypatch.setitem(sys.modules, "selectolax.lexbor", None)
        monkeypatch.setitem(sys.modules, "selectolax.parser", None)

    def test_losing_the_parser_warns_rather_than_degrading_in_silence(
        self, monkeypatch, caplog
    ):
        self._break_both_backends(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="ingest.normalize"):
            assert html_parser_class() is None
        assert "selectolax is unusable" in caplog.text
        # The message has to say what it costs, not just that it happened.
        assert "urls" in caplog.text and "domains" in caplog.text

    def test_the_warning_fires_once_not_once_per_record(self, monkeypatch, caplog):
        self._break_both_backends(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="ingest.normalize"):
            for _ in range(50):
                html_parser_class()
        assert caplog.text.count("selectolax is unusable") == 1

    def test_the_regex_fallback_still_returns_something_usable(self, monkeypatch):
        """Degraded, not broken: a wrong domain list beats a crashed run."""
        self._break_both_backends(monkeypatch)
        html = (
            '<p><a href="https://mastodon.social/tags/election" class="mention hashtag" '
            'rel="tag">#election</a> <a href="https://news.example/story">source</a></p>'
        )
        links = extract_html_links(html)
        assert "https://news.example/story" in links
        # And this is precisely the pollution the fallback cannot avoid. Asserted
        # so the cost of the degraded path is recorded rather than assumed.
        assert "https://mastodon.social/tags/election" in links
        # strip_html degrades gracefully: the block boundary survives because
        # it is turned into a newline before the parser is reached at all.
        assert strip_html("<p>a</p><p>b</p>") == "a\nb\n"


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

    def test_bare_mention_collapses_into_its_qualified_form(self):
        # Mastodon shows "@colleague" but the entity carries the full handle;
        # counting both would split one account into two graph nodes.
        assert extract_mentions("hi @colleague", [{"acct": "colleague@instance.tld"}]) == [
            "colleague@instance.tld"
        ]

    def test_distinct_local_mentions_are_not_collapsed(self):
        assert extract_mentions("@someone_else", [{"acct": "colleague@instance.tld"}]) == [
            "colleague@instance.tld",
            "someone_else",
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

    def test_devanagari_articles_are_detected_as_hindi(self):
        text = (
            "बंगाल में मतदाता सूची के पुनरीक्षण के बाद सिर्फ सात लाख वोटरों ने "
            "फिर से शामिल होने के लिए आवेदन किया है, यह दावा भ्रामक है।"
        )
        assert detect_lang(text) == "hi"


class TestScriptProfile:
    """Writing-system detection. Cheap, deterministic, and no model involved."""

    def test_shares_sum_to_one_over_attributable_letters(self):
        profile = script_profile("यह video है")
        assert pytest.approx(sum(profile.values()), abs=1e-9) == 1.0
        assert set(profile) == {"Devanagari", "Latin"}

    def test_empty_text_is_no_evidence_not_zero_evidence(self):
        # {} rather than {"Latin": 0.0}: a caller must be able to tell
        # "nothing to read" from "read it, found no Latin".
        assert script_profile("") == {}
        assert script_profile(None) == {}
        assert script_profile("12345 !?@#") == {}
        assert dominant_script("") is None

    def test_each_indian_script_is_recognised(self):
        for text, expected in [
            ("यह झूठ है", "Devanagari"),
            ("এটা মিথ্যা", "Bengali"),
            ("இது பொய்", "Tamil"),
            ("ఇది అబద్ధం", "Telugu"),
            ("ಇದು ಸುಳ್ಳು", "Kannada"),
            ("ഇത് നുണയാണ്", "Malayalam"),
            ("આ ખોટું છે", "Gujarati"),
            ("ਇਹ ਝੂਠ ਹੈ", "Gurmukhi"),
            ("یہ جھوٹ ہے", "Arabic"),
        ]:
            assert dominant_script(text) == expected, text

    def test_code_mixing_needs_both_scripts_to_be_substantial(self):
        assert is_code_mixed("यह video बिलकुल fake है")
        assert not is_code_mixed("plain english only")
        assert not is_code_mixed("पूरी तरह हिंदी में लिखा गया वाक्य")
        # One stray Latin brand name in a Hindi paragraph is not code-mixing.
        hindi = "यह दावा पूरी तरह से गलत है और इसकी पुष्टि नहीं हुई है। " * 3
        assert not is_code_mixed(hindi + "WhatsApp")

    def test_devanagari_is_not_mapped_to_a_language_by_script_alone(self):
        """Hindi, Marathi and Nepali share it, so the script cannot decide."""
        assert "Devanagari" not in SCRIPT_TO_LANG
        # Short Devanagari therefore stays an honest null rather than a guess.
        assert detect_lang("यह झूठ है") is None

    def test_short_text_in_a_single_language_script_resolves_anyway(self):
        """langdetect abstains under 20 chars; the alphabet does not have to."""
        assert detect_lang("இது பொய்") == "ta"
        assert detect_lang("ఇది అబద్ధం") == "te"
        assert detect_lang("આ ખોટું છે") == "gu"
        assert detect_lang("ਇਹ ਝੂਠ ਹੈ") == "pa"
        # Latin is used by hundreds of languages, so it resolves nothing.
        assert detect_lang("short en") is None


class TestRomanizedHindi:
    """Hindi typed in Latin letters: the dominant register of Indian social media.

    langdetect has no class for it, so it cannot abstain -- it returns the
    nearest of its 55 trained languages with confidence. Measured on ten
    hand-written sentences: Swahili x5, Estonian x2, Somali x2, Turkish x1,
    English x0. A confident `sw` is not a near miss, it routes the record to
    the "non-English, do not score" path and out of the analysis.
    """

    POSITIVES = [
        "yeh video bilkul fake hai maine pehle bhi dekha hai",
        "bhai ye khabar sach hai ya jhooth batao",
        "is video ko sabke paas share karo jaldi",
        "mujhe lagta hai ye galat jankari failai ja rahi hai",
        "aapne jo bataya wo bilkul sahi hai sir",
        "sab log is message ko forward kar rahe hain",
        "agar ye sach hai to proof dikhao",
        "unka kehna hai ki video edited hai",
    ]

    #: Each of the first three lands exactly ONE marker. They are the reason
    #: the threshold is two markers and not one.
    ADVERSARIAL_ENGLISH = [
        "jo biden said that on tuesday in washington",
        "the ki is a japanese concept of life energy",
        "se habla espanol here at our office today",
        "this is so sad bro i cant believe it happened",
        "can someone fact check this video please",
        "ka ching that is a lot of money right there",
    ]

    def test_romanized_hindi_is_labelled_hindi(self):
        for text in self.POSITIVES:
            assert looks_romanized_hindi(text), text
            assert detect_lang(text) == "hi", text

    def test_english_is_never_relabelled(self):
        for text in self.ADVERSARIAL_ENGLISH:
            assert not looks_romanized_hindi(text), text
            assert detect_lang(text) == "en", text

    def test_one_marker_is_not_enough(self):
        """The single-marker setting is what admits 'jo biden said that'."""
        for text in self.ADVERSARIAL_ENGLISH[:3]:
            hits, _ = romanized_hindi_markers(text)
            assert hits == 1, text
            assert looks_romanized_hindi(text, min_markers=1, min_share=0.0), text
            assert not looks_romanized_hindi(text), text

    def test_english_colliding_tokens_are_absent_from_the_lexicon(self):
        """Including any of these buys recall and sells a wrong language label."""
        for token in ("to", "is", "par", "ye", "log", "the", "hum", "main", "ab", "do", "me"):
            assert token not in ROMANIZED_HINDI_MARKERS, token

    def test_other_romanized_indian_languages_are_out_of_scope_not_handled(self):
        """A per-language lexicon each time. Documented limit, not a bug."""
        for text in [
            "indha video poi illa naan paatha irukken",  # Tamil
            "ee video thappu aanu njan kandittund",  # Malayalam
            "ee video tappu nenu chusanu",  # Telugu
        ]:
            assert romanized_hindi_markers(text)[0] == 0, text

    def test_no_latin_tokens_is_zero_not_a_crash(self):
        assert romanized_hindi_markers("यह पूरी तरह हिंदी है") == (0, 0.0)
        assert romanized_hindi_markers("") == (0, 0.0)
        assert romanized_hindi_markers(None) == (0, 0.0)
        assert not looks_romanized_hindi(None)

    def test_the_check_runs_before_langdetect_not_after(self):
        """Order matters: langdetect would have already committed to Swahili."""
        text = "sab log is message ko forward kar rahe hain"
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        assert detect(text) != "hi"  # langdetect alone gets this wrong
        assert detect_lang(text) == "hi"  # the lexicon catches it first


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
