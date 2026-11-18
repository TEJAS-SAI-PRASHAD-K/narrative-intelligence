"""Archive backfill through paginated listing pages (BOOM Hindi).

The sitemaps of this publisher list only the newest articles, so `backfill.listing`
reads the category pages instead. Payloads are small hand-written files under
fixtures/factcheck_backfill/. `requests.get` is replaced by a table of canned
responses, so no test opens a socket.
"""

from __future__ import annotations

import pytest

from ingest.config import Settings
from ingest.factcheck import DropReason, FactCheckExtractor
from ingest.factcheck_backfill import SitemapBackfill
from tests.test_factcheck_backfill import FakeWeb, Response, fixture_bytes

DOMAIN = "boomlive.in"
HOST = "https://hindi.boomlive.in"
ROBOTS = f"{HOST}/robots.txt"
PAGE_1 = f"{HOST}/fact-check"
PAGE_2 = f"{HOST}/fact-check/2"
PAGE_3 = f"{HOST}/fact-check/3"
DAILY = f"{HOST}/sitemap-daily.xml"
ARTICLES = {
    "32481": f"{HOST}/fact-check/old-video-viral-with-false-claim-32481",
    "32479": f"{HOST}/fact-check/edited-image-viral-32479",
    "32477": f"{HOST}/fact-check/-leading-hyphen-slug-32477",
    "32410": f"{HOST}/fact-check/older-claim-32410",
    "32400": f"{HOST}/fact-check/oldest-claim-32400",
}
LINK_PATTERN = r'href="(/fact-check/[^"/?#]+-[0-9]{3,6})"'


def listing_config(**listing_overrides) -> dict:
    listing = {
        "first_url": PAGE_1,
        "page_url": f"{HOST}/fact-check/{{page}}",
        "link_pattern": LINK_PATTERN,
        "max_pages": 10,
    }
    listing.update(listing_overrides)
    return {
        "publishers": {
            DOMAIN: {
                "name": "BOOM",
                "tier": "claimreview",
                "languages": ["en", "hi"],
                "backfill": {
                    "listing": listing,
                    "url_pattern": r"^https://hindi\.boomlive\.in/fact-check/",
                    "claim_languages": ["hi"],
                },
            }
        },
        "ratings": {"false": "false", "true": "true", "misleading": "drop"},
        "fetch": {"max_articles": 5, "request_delay_seconds": 0, "cache": True},
        "backfill": {"max_urls_per_publisher": 50, "min_lastmod": "2022-01-01"},
    }


def routes() -> dict[str, Response | Exception]:
    article = Response(200, fixture_bytes("hindi_claimreview.html"))
    table: dict[str, Response | Exception] = {
        ROBOTS: Response(200, b"User-agent: *\nAllow: /\nDisallow: /search\n"),
        PAGE_1: Response(200, fixture_bytes("boom_hindi_listing_page1.html")),
        PAGE_2: Response(200, fixture_bytes("boom_hindi_listing_page2.html")),
        PAGE_3: Response(200, fixture_bytes("boom_hindi_listing_empty.html")),
    }
    table.update({url: article for url in ARTICLES.values()})
    return table


@pytest.fixture
def web(monkeypatch):
    import requests

    fake = FakeWeb(routes())
    monkeypatch.setattr(requests, "get", fake)
    return fake


def make(tmp_path, config=None) -> SitemapBackfill:
    extractor = FactCheckExtractor(
        settings=Settings(data_dir=tmp_path), config=config or listing_config()
    )
    return SitemapBackfill(extractor, set())


def pages_read(web) -> list[str]:
    articles = set(ARTICLES.values())
    return [u for u in web.calls if u.startswith(f"{HOST}/fact-check") and u not in articles]


def test_pages_are_read_until_one_holds_no_new_link(tmp_path, web):
    backfill = make(tmp_path)
    entries = backfill.candidates(DOMAIN)
    assert pages_read(web) == [PAGE_1, PAGE_2, PAGE_3]
    assert {e.loc for e in entries} == set(ARTICLES.values())
    stats = backfill.stats[DOMAIN]
    assert stats.listing_pages_read == 3
    assert stats.listing_links_found == 5
    assert stats.dropped[DropReason.LISTING_END.value] == 1


def test_relative_links_are_joined_and_other_sections_are_not_matched(tmp_path, web):
    entries = {e.loc for e in make(tmp_path).candidates(DOMAIN)}
    assert f"{HOST}/explainers/not-a-fact-check-32478" not in entries
    assert f"{HOST}/fact-check/absolute-link-is-not-matched-32476" not in entries
    assert ARTICLES["32477"] in entries


def test_a_listing_only_block_yields_verdicts_in_the_configured_language(tmp_path, web):
    backfill = make(tmp_path)
    verdicts = list(backfill.run([DOMAIN]))
    assert {v.url for v in verdicts} == set(ARTICLES.values())
    assert {v.claim_lang for v in verdicts} == {"hi"}
    assert {v.label for v in verdicts} == {"false"}
    assert backfill.stats[DOMAIN].verdicts == 5


def test_the_url_bound_stops_paging_before_the_end(tmp_path, web):
    config = listing_config()
    config["backfill"]["max_urls_per_publisher"] = 3
    entries = make(tmp_path, config).candidates(DOMAIN)
    assert len(entries) == 3
    assert pages_read(web) == [PAGE_1]


def test_max_pages_stops_paging_and_is_counted(tmp_path, web):
    backfill = make(tmp_path, listing_config(max_pages=2))
    backfill.candidates(DOMAIN)
    assert pages_read(web) == [PAGE_1, PAGE_2]
    assert backfill.stats[DOMAIN].dropped[DropReason.LISTING_PAGES_EXHAUSTED.value] == 1


def test_a_page_repeating_earlier_links_ends_the_archive(tmp_path, web):
    web.routes[PAGE_2] = web.routes[PAGE_1]
    backfill = make(tmp_path)
    entries = backfill.candidates(DOMAIN)
    assert pages_read(web) == [PAGE_1, PAGE_2]
    assert len(entries) == 3
    assert backfill.stats[DOMAIN].dropped[DropReason.LISTING_END.value] == 1


def test_an_unreachable_page_keeps_what_was_read_and_is_counted(tmp_path, web):
    web.routes[PAGE_2] = Response(503, b"")
    backfill = make(tmp_path)
    entries = backfill.candidates(DOMAIN)
    assert len(entries) == 3
    assert PAGE_3 not in web.calls
    assert backfill.stats[DOMAIN].dropped[DropReason.LISTING_UNREACHABLE.value] == 1


def test_an_oversized_page_is_counted_and_stops_paging(tmp_path, web):
    backfill = make(tmp_path, listing_config(max_page_bytes=50))
    assert backfill.candidates(DOMAIN) == []
    assert backfill.stats[DOMAIN].dropped[DropReason.LISTING_TOO_LARGE.value] == 1
    assert pages_read(web) == [PAGE_1]


def test_a_link_on_another_domain_is_refused_and_counted(tmp_path, web):
    web.routes[PAGE_1] = Response(
        200, b'<a href="https://elsewhere.example/fact-check/x-32481">x</a>'
    )
    config = listing_config(link_pattern=r'href="(https://[^"]+/fact-check/[^"/]+-[0-9]{3,6})"')
    config["publishers"][DOMAIN]["backfill"].pop("url_pattern")
    backfill = make(tmp_path, config)
    assert backfill.candidates(DOMAIN) == []
    assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_OFF_DOMAIN.value] == 1


def test_an_incomplete_listing_block_is_counted_not_run(tmp_path, web):
    backfill = make(tmp_path, listing_config(link_pattern=""))
    assert backfill.candidates(DOMAIN) == []
    assert backfill.stats[DOMAIN].dropped[DropReason.NO_BACKFILL_CONFIG.value] == 1
    assert web.calls == []


def test_robots_disallow_on_a_listing_page_is_counted(tmp_path, web):
    web.routes[ROBOTS] = Response(200, b"User-agent: *\nDisallow: /fact-check\n")
    backfill = make(tmp_path)
    assert backfill.candidates(DOMAIN) == []
    assert backfill.stats[DOMAIN].dropped[DropReason.ROBOTS_DISALLOWED.value] == 1


def test_sitemap_and_listing_share_one_fetch_per_article(tmp_path, web):
    sitemap = (
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>" + ARTICLES["32481"].encode() + b"</loc>"
        b"<lastmod>2026-11-17T16:12:07+05:30</lastmod></url></urlset>"
    )
    web.routes[DAILY] = Response(200, sitemap)
    config = listing_config()
    config["publishers"][DOMAIN]["backfill"]["sitemaps"] = [DAILY]
    backfill = make(tmp_path, config)
    entries = backfill.candidates(DOMAIN)
    assert len(entries) == 5
    stats = backfill.stats[DOMAIN]
    assert stats.duplicate_loc == 1
    list(backfill.run([DOMAIN]))
    assert web.calls.count(ARTICLES["32481"]) == 1
