"""Fact-check verdict extraction.

Pure parsing is tested against markup copied from the real publishers, because
every bug worth catching here is a shape surprise in someone's CMS output
rather than a logic error.
"""

from __future__ import annotations

import logging

import pytest

from ingest.factcheck import (
    DropReason,
    FactCheck,
    FactCheckExtractor,
    Tier,
    claim_and_rating,
    find_claim_reviews,
    label_distribution,
    normalize_rating,
    write_factchecks,
)

# --- markup shapes, as the publishers actually emit them -------------------

#: BOOM: ClaimReview at the top level of its own script block.
BOOM_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"ClaimReview",
 "datePublished":"2026-10-08",
 "itemReviewed":{"@type":"Claim","name":"Video shows Sheikh Hasina calling for unification"},
 "reviewRating":{"@type":"Rating","alternateName":"False","ratingValue":1,
                 "bestRating":5,"worstRating":1},
 "author":{"@type":"Organization","name":"BOOM"}}
</script>
</head><body>article</body></html>
"""

#: The Quint: `itemReviewed.name` is a LIST. schema.org allows it; reading it as
#: a string yields a stringified Python list, which looks like success.
QUINT_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"ClaimReview",
 "itemReviewed":{"@type":"Claim","name":["The video shows the hijacked Flydubai flight"]},
 "reviewRating":{"@type":"Rating","alternateName":"False"}}
</script>
</head></html>
"""

#: Vishvas News: nested inside @graph, claim in Devanagari.
VISHVAS_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@graph":[
  {"@type":"WebPage","name":"page"},
  {"@type":"ClaimReview",
   "itemReviewed":{"@type":"Claim","name":"स्मृति ईरानी मदीना के दौरे पर हैं"},
   "reviewRating":{"@type":"Rating","alternateName":"Half true"}}
]}
</script>
</head></html>
"""

#: Alt News / Factly: Article markup only, no ClaimReview anywhere.
NO_REVIEW_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Article","headline":"Something happened"}
</script>
</head></html>
"""


@pytest.fixture
def extractor(tmp_path, monkeypatch):
    from ingest.config import Settings

    settings = Settings(data_dir=tmp_path)
    config = {
        "publishers": {
            "boomlive.in": {"name": "BOOM", "tier": "claimreview"},
            "factcrescendo.com": {"name": "Fact Crescendo", "tier": "rss_tag"},
        },
        "ratings": {
            "false": "false",
            "fake": "false",
            "true": "true",
            "half true": "drop",
            "misleading": "drop",
        },
        "fetch": {"max_articles": 5, "request_delay_seconds": 0, "cache": True},
    }
    return FactCheckExtractor(settings=settings, config=config)


class TestJsonLdDiscovery:
    def test_top_level_claimreview(self):
        reviews = find_claim_reviews(BOOM_HTML)
        assert len(reviews) == 1
        claim, rating = claim_and_rating(reviews[0])
        assert claim == "Video shows Sheikh Hasina calling for unification"
        assert rating == "False"

    def test_claimreview_nested_in_a_graph(self):
        """Publishers bury it in @graph, in lists, under the article node."""
        reviews = find_claim_reviews(VISHVAS_HTML)
        assert len(reviews) == 1
        claim, rating = claim_and_rating(reviews[0])
        assert claim == "स्मृति ईरानी मदीना के दौरे पर हैं"
        assert rating == "Half true"

    def test_a_list_valued_claim_name_is_not_stringified(self):
        """The Quint returns a list. `str(["x"])` is '[\\'x\\']' and looks fine."""
        claim, rating = claim_and_rating(find_claim_reviews(QUINT_HTML)[0])
        assert claim == "The video shows the hijacked Flydubai flight"
        assert "[" not in claim and "'" not in claim

    def test_article_markup_yields_no_reviews(self):
        assert find_claim_reviews(NO_REVIEW_HTML) == []

    def test_malformed_block_does_not_lose_a_good_one(self):
        """Publishers ship trailing commas. One bad block must cost nothing."""
        html = (
            '<script type="application/ld+json">{"broken": ,}</script>' + BOOM_HTML
        )
        assert len(find_claim_reviews(html)) == 1

    def test_no_markup_at_all(self):
        assert find_claim_reviews("") == []
        assert find_claim_reviews("<html><body>nothing</body></html>") == []

    def test_numeric_rating_value_is_never_used_as_the_verdict(self):
        """`ratingValue` is a number on an UNDECLARED scale.

        BOOM's 1-of-5 means "worst"; another publisher's 1 could mean "best".
        Only the human-readable `alternateName` is comparable across sites.
        """
        _, rating = claim_and_rating(find_claim_reviews(BOOM_HTML)[0])
        assert rating == "False"
        assert rating != "1"


class TestRatingNormalization:
    def test_case_and_punctuation_are_folded(self):
        assert normalize_rating("False") == "false"
        assert normalize_rating("  HALF-TRUE  ") == "half true"
        assert normalize_rating("Missing Context!") == "missing context"
        assert normalize_rating("AI-Generated") == "ai generated"

    def test_empty_input(self):
        assert normalize_rating("") == ""
        assert normalize_rating(None) == ""


class TestLabelMapping:
    def test_false_and_true_map_through(self, extractor):
        assert extractor.label_for("False") == "false"
        assert extractor.label_for("fake") == "false"
        assert extractor.label_for("TRUE") == "true"

    def test_the_middle_is_dropped_not_forced(self, extractor):
        assert extractor.label_for("Half true") is None
        assert extractor.label_for("Misleading") is None
        assert extractor.dropped[DropReason.RATING_IS_MIDDLE.value] == 2

    def test_a_middle_rating_and_an_unknown_one_are_counted_separately(self, extractor):
        """Different problems: one is our decision, one is a publisher change.

        Only the second is a bug to go and fix, so a single "dropped" count
        would hide the one that needs action.
        """
        extractor.label_for("Half true")
        extractor.label_for("Mostly Fabricated Nonsense")
        assert extractor.dropped[DropReason.RATING_IS_MIDDLE.value] == 1
        assert extractor.dropped[DropReason.RATING_UNKNOWN.value] == 1
        assert extractor.unknown_ratings == {"mostly fabricated nonsense": 1}

    def test_every_rating_seen_is_counted_including_dropped_ones(self, extractor):
        """The drop count is the cost of the binary mapping; it must be visible."""
        for rating in ("False", "False", "Half true", "Misleading"):
            extractor.label_for(rating)
        assert extractor.ratings_seen == {"false": 2, "half true": 1, "misleading": 1}


class TestClaimReviewTier:
    def test_extracts_a_verdict_with_its_tier_recorded(self, extractor):
        record = {
            "id": "news:boomlive.in/fact-check/x",
            "url": "https://www.boomlive.in/fact-check/x",
            "source_detail": "boomlive.in",
            "title": "headline",
        }
        verdict = extractor.from_html(record, BOOM_HTML)
        assert verdict is not None
        assert verdict.label == "false"
        assert verdict.rating_raw == "False"
        assert verdict.tier == Tier.CLAIMREVIEW.value
        assert verdict.publisher == "BOOM"
        assert verdict.record_id == record["id"]
        assert verdict.claim_lang == "en"
        assert verdict.reviewed_at == "2026-10-08"

    def test_the_publishers_claim_is_kept_not_the_headline(self, extractor):
        """They differ, and the difference is the point.

        A headline is written to be read; a claim is written to be checked.
        Training on headlines teaches the model a publisher's house style.
        """
        record = {
            "id": "news:x",
            "url": "https://www.boomlive.in/fact-check/x",
            "source_detail": "boomlive.in",
            "title": "Video Of Sheikh Hasina Calling For Unified Bengal Is AI-Generated",
        }
        verdict = extractor.from_html(record, BOOM_HTML)
        assert verdict.claim == "Video shows Sheikh Hasina calling for unification"
        assert verdict.claim != record["title"]

    def test_absent_markup_is_distinguished_from_an_unusable_review(self, extractor):
        record = {
            "id": "news:x",
            "url": "https://www.boomlive.in/x",
            "source_detail": "boomlive.in",
        }
        assert extractor.from_html(record, NO_REVIEW_HTML) is None
        assert extractor.dropped[DropReason.NO_CLAIMREVIEW.value] == 1
        # A review that exists but carries no rating is a different bug.
        empty = '<script type="application/ld+json">{"@type":"ClaimReview"}</script>'
        assert extractor.from_html(record, empty) is None
        assert extractor.dropped[DropReason.NO_RATING.value] == 1


class TestRssTagTier:
    def test_a_verdict_hiding_among_topic_tags_is_found(self, extractor):
        """Fact Crescendo ships ['False', 'International', 'इजरायल', ...]."""
        record = {
            "id": "news:factcrescendo.com/x",
            "url": "https://factcrescendo.com/x/",
            "source_detail": "factcrescendo.com",
            "title": "इजरायल पर ईरानी मिसाइल हमले का दावा गलत",
        }
        verdict = extractor.from_rss_tags(record, ["False", "International", "इजरायल"])
        assert verdict is not None
        assert verdict.label == "false"
        assert verdict.tier == Tier.RSS_TAG.value
        assert verdict.claim_lang == "hi"

    def test_tags_with_no_verdict_yield_nothing_and_are_counted(self, extractor):
        record = {
            "id": "news:x",
            "url": "https://factcrescendo.com/x/",
            "source_detail": "factcrescendo.com",
        }
        assert extractor.from_rss_tags(record, ["Politics", "यूपी"]) is None
        assert extractor.dropped == {DropReason.NO_VERDICT_TAG.value: 1}

    def test_a_middle_verdict_in_the_tags_still_drops(self, extractor):
        record = {
            "id": "news:x",
            "url": "https://factcrescendo.com/x/",
            "source_detail": "factcrescendo.com",
            "title": "t",
        }
        assert extractor.from_rss_tags(record, ["Misleading", "Political"]) is None
        assert extractor.dropped[DropReason.RATING_IS_MIDDLE.value] == 1


class TestRobotsHandling:
    def test_unreachable_robots_is_not_reported_as_disallowed(self, extractor, monkeypatch):
        """A flaky network must not look like a publisher ban.

        The first run of this module reported 53 articles as robots-disallowed
        across four publishers whose robots.txt permits all of them. Every one
        was the fetch failing, which is a local problem, not a policy.
        """
        def boom(*args, **kwargs):
            raise OSError("network down")

        import requests

        monkeypatch.setattr(requests, "get", boom)
        allowed, reason = extractor._robots_allows("https://www.boomlive.in/fact-check/x")
        assert allowed is False
        assert reason == DropReason.ROBOTS_UNREACHABLE

    def test_a_real_disallow_is_reported_as_one(self, extractor, monkeypatch):
        class Response:
            status_code = 200
            text = "User-agent: *\nDisallow: /fact-check/\n"

        import requests

        monkeypatch.setattr(requests, "get", lambda *a, **k: Response())
        allowed, reason = extractor._robots_allows("https://www.boomlive.in/fact-check/x")
        assert allowed is False
        assert reason == DropReason.ROBOTS_DISALLOWED

    def test_a_missing_robots_txt_permits_crawling(self, extractor, monkeypatch):
        """404 means no rules, which the standard reads as allow."""
        class Response:
            status_code = 404
            text = ""

        import requests

        monkeypatch.setattr(requests, "get", lambda *a, **k: Response())
        allowed, reason = extractor._robots_allows("https://www.boomlive.in/fact-check/x")
        assert allowed is True
        assert reason is None


class TestDriver:
    def test_an_unconfigured_domain_is_skipped_and_counted(self, extractor):
        records = [{"id": "news:x", "url": "https://example.com/a", "source_detail": "example.com"}]
        assert list(extractor.run(records)) == []
        assert extractor.dropped[DropReason.NO_PUBLISHER_CONFIG.value] == 1

    def test_a_record_with_no_url_is_skipped_and_counted(self, extractor):
        records = [{"id": "news:x", "url": "", "source_detail": "boomlive.in"}]
        assert list(extractor.run(records)) == []
        assert extractor.dropped[DropReason.NO_URL.value] == 1


BOOM_URL = "https://www.boomlive.in/fact-check/x"


def _no_network(monkeypatch):
    """Make any real HTTP call fail the test instead of reaching the network."""
    import requests

    def refuse(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(requests, "get", refuse)


class TestDriverTiers:
    def test_run_yields_one_verdict_per_tier(self, extractor, monkeypatch):
        _no_network(monkeypatch)
        # A cache hit stands in for the fetch on the claimreview tier.
        cache = extractor._cache_path(BOOM_URL)
        cache.parent.mkdir(parents=True)
        cache.write_text(BOOM_HTML, encoding="utf-8")
        records = [
            {"id": "news:a", "url": BOOM_URL, "source_detail": "boomlive.in"},
            {
                "id": "news:b",
                "url": "https://factcrescendo.com/b/",
                "source_detail": "factcrescendo.com",
                "title": "a headline",
                "tags": ["Fake", "Politics"],
            },
        ]
        verdicts = list(extractor.run(records))
        assert [(v.record_id, v.tier, v.label) for v in verdicts] == [
            ("news:a", Tier.CLAIMREVIEW.value, "false"),
            ("news:b", Tier.RSS_TAG.value, "false"),
        ]
        assert extractor.dropped == {}

    def test_an_empty_page_is_counted_not_lost(self, extractor, monkeypatch):
        _no_network(monkeypatch)
        cache = extractor._cache_path(BOOM_URL)
        cache.parent.mkdir(parents=True)
        cache.write_text("", encoding="utf-8")
        records = [{"id": "news:a", "url": BOOM_URL, "source_detail": "boomlive.in"}]
        assert list(extractor.run(records)) == []
        assert extractor.dropped == {DropReason.NO_CLAIMREVIEW.value: 1}


class TestFetchHtml:
    def test_a_cache_hit_needs_no_network_and_no_budget(self, extractor, monkeypatch):
        _no_network(monkeypatch)
        cache = extractor._cache_path(BOOM_URL)
        cache.parent.mkdir(parents=True)
        cache.write_text("<html>cached</html>", encoding="utf-8")
        before = extractor._budget
        assert extractor.fetch_html(BOOM_URL) == "<html>cached</html>"
        assert extractor._budget == before
        assert extractor.dropped == {}

    def test_a_spent_budget_is_counted_per_record(self, extractor, monkeypatch):
        _no_network(monkeypatch)
        extractor._budget = 0
        records = [
            {"id": f"news:{i}", "url": f"{BOOM_URL}{i}", "source_detail": "boomlive.in"}
            for i in range(2)
        ]
        assert list(extractor.run(records)) == []
        assert extractor.dropped == {DropReason.BUDGET_EXHAUSTED.value: 2}

    def test_a_domain_at_its_failure_limit_is_counted(self, extractor, monkeypatch):
        _no_network(monkeypatch)
        extractor._fetch_failures["boomlive.in"] = 3
        assert extractor.fetch_html(BOOM_URL) is None
        assert extractor.dropped == {DropReason.DOMAIN_GIVEN_UP.value: 1}

    def test_an_ordinary_failure_is_counted_once_under_its_own_code(
        self, extractor, monkeypatch
    ):
        import requests

        def boom(*args, **kwargs):
            raise OSError("down")

        monkeypatch.setattr(extractor, "_robots_allows", lambda url: (True, None))
        monkeypatch.setattr(requests, "get", boom)
        records = [{"id": "news:a", "url": BOOM_URL, "source_detail": "boomlive.in"}]
        assert list(extractor.run(records)) == []
        assert extractor.dropped == {DropReason.FETCH_FAILED.value: 1}


class TestSideArtifact:
    def _verdict(self, simhash: int) -> FactCheck:
        return FactCheck(
            record_id="news:x",
            url="https://www.boomlive.in/fact-check/x",
            domain="boomlive.in",
            publisher="BOOM",
            tier=Tier.CLAIMREVIEW.value,
            claim="a claim",
            claim_lang="en",
            rating_raw="False",
            label="false",
            claim_simhash=simhash,
        )

    def test_a_high_simhash_survives_the_round_trip(self, tmp_path):
        """simhash is UNSIGNED 64-bit; pyarrow infers int64 and overflows.

        Found by running it: anything above 2**63 raised "Python int too large
        to convert to C long", i.e. on roughly half of all simhashes, which
        presents as an intermittent crash with no obvious cause.
        """
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        from ingest.config import Settings

        big = 2**64 - 1
        path = write_factchecks([self._verdict(big)], Settings(data_dir=tmp_path))
        assert path is not None
        table = pq.read_table(path)
        assert table.column("claim_simhash")[0].as_py() == big
        assert table.schema.field("claim_simhash").type == "uint64"

    def test_nothing_to_write_writes_nothing(self, tmp_path):
        from ingest.config import Settings

        assert write_factchecks([], Settings(data_dir=tmp_path)) is None


class TestSingleClassWarning:
    def test_published_fact_checks_are_usually_one_class(self, extractor, caplog):
        """Not a quirk of one run: debunks are news, confirmations are not.

        The first full run over the real corpus produced 96 `false` and 0
        `true`, which makes this artifact positive-class supervision rather
        than a trainable binary dataset.
        """
        verdicts = [
            FactCheck(
                record_id=f"news:{i}",
                url="u",
                domain="boomlive.in",
                publisher="BOOM",
                tier=Tier.CLAIMREVIEW.value,
                claim="c",
                claim_lang="en",
                rating_raw="False",
                label="false",
                claim_simhash=1,
            )
            for i in range(5)
        ]
        with caplog.at_level(logging.WARNING):
            assert label_distribution(verdicts) == {"false": 5}
        assert len(label_distribution(verdicts)) == 1
