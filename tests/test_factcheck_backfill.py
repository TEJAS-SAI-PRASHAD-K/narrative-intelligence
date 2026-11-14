"""Sitemap backfill of fact-check verdicts.

Every payload is a small hand-written file under fixtures/factcheck_backfill/.
`requests.get` is replaced by a table of canned responses, so no test opens a
socket.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ingest import cli, factcheck
from ingest.config import Settings
from ingest.factcheck import DropReason, FactCheckExtractor, write_factchecks
from ingest.factcheck_backfill import (
    LEDGER_FILENAME,
    SitemapBackfill,
    existing_record_ids,
    ledger_path,
    looks_like_article,
    looks_like_challenge,
    parse_lastmod,
    parse_sitemap,
    read_ledger,
    record_for_url,
    update_ledger,
)
from ingest.sources.news_rss import _article_id

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "factcheck_backfill"
DOMAIN = "vishvasnews.com"
INDEX_URL = "https://www.vishvasnews.com/sitemap_index.xml"
SITEMAP_1 = "https://www.vishvasnews.com/post-sitemap.xml"
SITEMAP_2 = "https://www.vishvasnews.com/post-sitemap2.xml"
ROBOTS_URL = "https://www.vishvasnews.com/robots.txt"
NEW_URL = "https://www.vishvasnews.com/viral/fact-check-new-one/"
NEWER_URL = "https://www.vishvasnews.com/politics/fact-check-newer-one/"
NO_LASTMOD_URL = "https://www.vishvasnews.com/politics/fact-check-no-lastmod/"
STORED_URL = "https://www.vishvasnews.com/viral/fact-check-already-stored/"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class Response:
    def __init__(self, status: int = 200, content: bytes = b"", final_url: str | None = None):
        self.status_code = status
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        self.encoding = "utf-8"
        self.final_url = final_url
        self.url = ""
        self.closed = False

    def iter_content(self, chunk_size: int = 1):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeWeb:
    """A table of canned responses. Any URL not in it is a test failure."""

    def __init__(self, routes: dict[str, Response | Exception]):
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, url, *args, **kwargs):
        self.calls.append(url)
        if url not in self.routes:
            raise AssertionError(f"unexpected request: {url}")
        outcome = self.routes[url]
        if isinstance(outcome, Exception):
            raise outcome
        outcome.url = outcome.final_url or url
        return outcome

    def articles(self) -> list[str]:
        return [u for u in self.calls if not u.endswith((".xml", "robots.txt"))]


def base_routes() -> dict[str, Response | Exception]:
    page = Response(200, fixture_bytes("hindi_claimreview.html"))
    return {
        ROBOTS_URL: Response(200, b"User-agent: *\nDisallow: /wp-admin/\n"),
        INDEX_URL: Response(200, fixture_bytes("sitemap_index.xml")),
        SITEMAP_1: Response(200, fixture_bytes("sitemap_urlset.xml")),
        SITEMAP_2: Response(200, fixture_bytes("sitemap_urlset.xml")),
        NEW_URL: page,
        NEWER_URL: page,
        NO_LASTMOD_URL: page,
        STORED_URL: page,
    }


def make_config(**overrides) -> dict:
    backfill_block = {
        "sitemaps": [INDEX_URL],
        "sitemap_pattern": r"/post-sitemap[0-9]*\.xml$",
        "url_pattern": "/fact-check",
        "exclude_pattern": r"^https://www\.vishvasnews\.com/english/",
    }
    backfill_block.update(overrides.pop("block", {}))
    config = {
        "publishers": {
            DOMAIN: {
                "name": "Vishvas News",
                "tier": "claimreview",
                "languages": ["hi", "en"],
                "backfill": backfill_block,
            },
            "boomlive.in": {"name": "BOOM", "tier": "claimreview"},
        },
        "ratings": {"false": "false", "true": "true", "misleading": "drop"},
        "fetch": {"max_articles": 5, "request_delay_seconds": 0, "cache": True},
        "backfill": {
            "max_urls_per_publisher": 50,
            "max_sitemaps_per_publisher": 10,
            "min_lastmod": "2022-01-01",
        },
    }
    return config


@pytest.fixture
def web(monkeypatch):
    import requests

    fake = FakeWeb(base_routes())
    monkeypatch.setattr(requests, "get", fake)
    return fake


def make_backfill(
    tmp_path, config=None, known=None, ledger=False, recheck=False
) -> SitemapBackfill:
    extractor = FactCheckExtractor(
        settings=Settings(data_dir=tmp_path), config=config or make_config()
    )
    return SitemapBackfill(
        extractor,
        known if known is not None else set(),
        ledger_file=ledger_path(Settings(data_dir=tmp_path)) if ledger else None,
        recheck_dropped=recheck,
    )


class TestParseSitemap:
    def test_index_lists_children_with_lastmod(self):
        kind, entries = parse_sitemap(fixture_bytes("sitemap_index.xml"))
        assert kind == "index"
        assert [e.loc for e in entries][:2] == [SITEMAP_1, SITEMAP_2]
        assert entries[1].lastmod.tzinfo == timezone.utc
        assert entries[3].lastmod is None

    def test_plain_urlset_with_extra_namespace_and_absent_lastmod(self):
        kind, entries = parse_sitemap(fixture_bytes("sitemap_urlset.xml"))
        assert kind == "urlset"
        assert len(entries) == 8
        by_loc = {e.loc: e for e in entries}
        assert by_loc[NO_LASTMOD_URL].lastmod is None
        assert by_loc[NEW_URL].lastmod is not None

    def test_truncated_xml_is_rejected(self):
        with pytest.raises(ValueError, match="well-formed"):
            parse_sitemap(fixture_bytes("sitemap_malformed.xml"))

    def test_a_doctype_with_entities_is_refused_before_parsing(self):
        with pytest.raises(ValueError, match="DOCTYPE"):
            parse_sitemap(fixture_bytes("sitemap_entity.xml"))

    def test_an_unknown_root_is_rejected(self):
        with pytest.raises(ValueError, match="root"):
            parse_sitemap(b"<html><body/></html>")

    def test_lastmod_formats(self):
        assert parse_lastmod("2026-02-19").tzinfo == timezone.utc
        assert parse_lastmod("2026-02-19T11:04:22+05:30").hour == 5
        assert parse_lastmod("not a date") is None
        assert parse_lastmod(None) is None

    @pytest.mark.parametrize("value", ["5", "2026-13-01", "19/02/2026", "yesterday", "2026-02-30"])
    def test_lastmod_has_no_lenient_fallback(self, value):
        assert parse_lastmod(value) is None

    @pytest.mark.parametrize(
        "value", ["2026", "2026-02", "2026-02-19T11:04", "2026-02-19T11:04:22.5Z"]
    )
    def test_lastmod_accepts_the_w3c_forms(self, value):
        assert parse_lastmod(value).year == 2026

    def test_a_utf8_byte_order_mark_is_accepted(self):
        body = b"\xef\xbb\xbf" + fixture_bytes("sitemap_urlset.xml")
        assert parse_sitemap(body)[0] == "urlset"

    def test_a_declared_encoding_other_than_utf8_or_ascii_is_refused(self):
        body = b'<?xml version="1.0" encoding="UTF-16"?><urlset/>'
        with pytest.raises(ValueError, match="encoding"):
            parse_sitemap(body)

    def test_empty_loc_is_counted(self):
        counters: dict[str, int] = {}
        parse_sitemap(fixture_bytes("sitemap_oddities.xml"), counters)
        assert counters == {"empty_loc": 1}


ENTITY_DOC = (
    '<?xml version="1.0"?>'
    '<!DOCTYPE urlset [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;">]>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<url><loc>https://www.vishvasnews.com/viral/&b;/</loc></url></urlset>"
)


class TestEncodingBypass:
    """A byte search for DOCTYPE cannot see UTF-16, so the encoding is checked first."""

    @pytest.mark.parametrize(
        "body",
        [
            ENTITY_DOC.encode("utf-16-le"),
            ENTITY_DOC.encode("utf-16-be"),
            ENTITY_DOC.encode("utf-16"),
            b"\xff\xfe" + ENTITY_DOC.encode("utf-16-le"),
            b"\xfe\xff" + ENTITY_DOC.encode("utf-16-be"),
            ENTITY_DOC.encode("utf-32"),
        ],
        ids=["utf16le", "utf16be", "utf16-bom", "utf16le-bom", "utf16be-bom", "utf32"],
    )
    def test_nothing_parses(self, body, monkeypatch):
        import xml.etree.ElementTree as ET

        def never(*args, **kwargs):
            raise AssertionError("the parser was reached")

        monkeypatch.setattr(ET, "fromstring", never)
        with pytest.raises(ValueError):
            parse_sitemap(body)

    def test_the_run_counts_it_as_malformed(self, tmp_path, web):
        web.routes[SITEMAP_2] = Response(200, ENTITY_DOC.encode("utf-16"))
        backfill = make_backfill(tmp_path)
        assert backfill.candidates(DOMAIN) == []
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_MALFORMED.value] == 1


class TestDiscovery:
    def test_index_is_followed_and_filters_apply(self, tmp_path, web):
        backfill = make_backfill(tmp_path)
        found = backfill.candidates(DOMAIN)
        urls = [e.loc for e in found]
        # Newest first. The page with no lastmod sorts last.
        assert urls == [NEWER_URL, NEW_URL, STORED_URL, NO_LASTMOD_URL]
        stats = backfill.stats[DOMAIN]
        assert stats.discovered == 4
        # page-sitemap.xml fails the sitemap pattern. The 2021 child fails the
        # lastmod cutoff. The off-domain child is refused.
        assert stats.sitemaps_filtered_pattern == 1
        assert stats.sitemaps_skipped_old == 1
        assert stats.dropped[DropReason.SITEMAP_OFF_DOMAIN.value] >= 1
        assert stats.filtered_lastmod == 1
        assert stats.filtered_pattern == 2
        assert stats.lastmod_absent_kept == 1
        assert SITEMAP_1 not in web.calls
        assert web.articles() == []

    def test_urls_already_stored_are_not_candidates(self, tmp_path, web):
        known = {record_for_url(STORED_URL, DOMAIN)["id"]}
        backfill = make_backfill(tmp_path, known=known)
        urls = [e.loc for e in backfill.candidates(DOMAIN)]
        assert STORED_URL not in urls
        assert backfill.stats[DOMAIN].already_present == 1

    def test_limit_bound_keeps_the_newest_and_counts_the_rest(self, tmp_path, web):
        config = make_config()
        config["backfill"]["max_urls_per_publisher"] = 2
        backfill = make_backfill(tmp_path, config)
        urls = [e.loc for e in backfill.candidates(DOMAIN)]
        assert urls == [NEWER_URL, NEW_URL]
        assert backfill.stats[DOMAIN].dropped[DropReason.BACKFILL_BOUND_REACHED.value] == 2

    def test_publisher_bound_can_be_tighter_than_the_global_one(self, tmp_path, web):
        config = make_config(block={"max_urls": 1})
        backfill = make_backfill(tmp_path, config)
        assert [e.loc for e in backfill.candidates(DOMAIN)] == [NEWER_URL]

    def test_min_lastmod_null_reads_the_old_child_too(self, tmp_path, web):
        config = make_config()
        config["backfill"]["min_lastmod"] = None
        backfill = make_backfill(tmp_path, config)
        urls = [e.loc for e in backfill.candidates(DOMAIN)]
        assert "https://www.vishvasnews.com/viral/fact-check-too-old/" in urls
        assert SITEMAP_1 in web.calls

    def test_sitemap_budget_is_counted(self, tmp_path, web):
        config = make_config()
        config["backfill"]["max_sitemaps_per_publisher"] = 1
        backfill = make_backfill(tmp_path, config)
        backfill.candidates(DOMAIN)
        assert SITEMAP_2 not in web.calls
        stats = backfill.stats[DOMAIN]
        assert stats.sitemaps_read == 1
        assert stats.dropped[DropReason.SITEMAP_BUDGET_EXHAUSTED.value] >= 1

    def test_a_sitemap_index_that_lists_itself_terminates(self, tmp_path, monkeypatch):
        import requests

        loop = (
            b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            b"<sitemap><loc>" + INDEX_URL.encode() + b"</loc></sitemap></sitemapindex>"
        )
        routes = base_routes()
        routes[INDEX_URL] = Response(200, loop)
        monkeypatch.setattr(requests, "get", FakeWeb(routes))
        backfill = make_backfill(tmp_path)
        assert backfill.candidates(DOMAIN) == []

    def test_a_malformed_sitemap_is_counted_and_the_run_continues(self, tmp_path, web):
        web.routes[SITEMAP_2] = Response(200, fixture_bytes("sitemap_malformed.xml"))
        backfill = make_backfill(tmp_path)
        assert backfill.candidates(DOMAIN) == []
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_MALFORMED.value] == 1
        assert backfill.extractor.dropped[DropReason.SITEMAP_MALFORMED.value] == 1

    def test_an_entity_bomb_is_counted_as_malformed(self, tmp_path, web):
        web.routes[SITEMAP_2] = Response(200, fixture_bytes("sitemap_entity.xml"))
        backfill = make_backfill(tmp_path)
        backfill.candidates(DOMAIN)
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_MALFORMED.value] == 1

    def test_an_unreachable_sitemap_is_counted_not_raised(self, tmp_path, web):
        web.routes[SITEMAP_2] = ConnectionError("down")
        backfill = make_backfill(tmp_path)
        assert backfill.candidates(DOMAIN) == []
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_UNREACHABLE.value] == 1

    def test_an_error_status_is_unreachable(self, tmp_path, web):
        web.routes[SITEMAP_2] = Response(503)
        backfill = make_backfill(tmp_path)
        backfill.candidates(DOMAIN)
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_UNREACHABLE.value] == 1

    def test_an_oversize_sitemap_is_counted(self, tmp_path, web):
        config = make_config()
        config["backfill"]["max_sitemap_bytes"] = 100
        web.routes[INDEX_URL] = Response(200, fixture_bytes("sitemap_index.xml"))
        backfill = make_backfill(tmp_path, config)
        assert backfill.candidates(DOMAIN) == []
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_TOO_LARGE.value] == 1

    def test_robots_disallow_stops_before_any_sitemap_is_fetched(self, tmp_path, web):
        web.routes[ROBOTS_URL] = Response(200, b"User-agent: *\nDisallow: /\n")
        backfill = make_backfill(tmp_path)
        assert backfill.candidates(DOMAIN) == []
        assert backfill.stats[DOMAIN].dropped[DropReason.ROBOTS_DISALLOWED.value] == 1
        assert web.calls == [ROBOTS_URL]

    def test_unreachable_robots_is_not_called_a_disallow(self, tmp_path, web):
        web.routes[ROBOTS_URL] = Response(500)
        backfill = make_backfill(tmp_path)
        backfill.candidates(DOMAIN)
        assert backfill.stats[DOMAIN].dropped[DropReason.ROBOTS_UNREACHABLE.value] == 1

    def test_missing_sitemaps_and_wrong_tier_are_counted(self, tmp_path, web):
        no_sitemaps = make_backfill(tmp_path, make_config(block={"sitemaps": []}))
        assert no_sitemaps.candidates(DOMAIN) == []
        assert no_sitemaps.stats[DOMAIN].dropped[DropReason.NO_BACKFILL_CONFIG.value] == 1

        config = make_config()
        config["publishers"][DOMAIN]["tier"] = "rss_tag"
        wrong_tier = make_backfill(tmp_path, config)
        assert wrong_tier.candidates(DOMAIN) == []
        assert wrong_tier.stats[DOMAIN].dropped[DropReason.NO_BACKFILL_CONFIG.value] == 1


class BoundedStream(Response):
    """A body of unlimited length that fails the test when read past a limit."""

    def __init__(self, fail_after: int):
        super().__init__(200, b"")
        self.fail_after = fail_after
        self.consumed = 0

    def iter_content(self, chunk_size: int = 1):
        while True:
            self.consumed += chunk_size
            if self.consumed > self.fail_after:
                raise AssertionError(f"read {self.consumed} bytes past the bound")
            yield b"x" * chunk_size


class TestSizeBound:
    def test_reading_stops_at_the_bound_not_after_the_full_body(self, tmp_path, web):
        config = make_config()
        bound = 200_000
        config["backfill"]["max_sitemap_bytes"] = bound
        stream = BoundedStream(fail_after=bound + 65536)
        web.routes[INDEX_URL] = stream
        backfill = make_backfill(tmp_path, config)
        assert backfill.candidates(DOMAIN) == []
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_TOO_LARGE.value] == 1
        assert bound < stream.consumed <= bound + 65536
        assert stream.closed


class TestRedirects:
    def test_a_sitemap_that_ends_off_domain_is_discarded(self, tmp_path, web):
        web.routes[SITEMAP_2] = Response(
            200, fixture_bytes("sitemap_urlset.xml"), final_url="https://evil.example/s.xml"
        )
        backfill = make_backfill(tmp_path)
        assert backfill.candidates(DOMAIN) == []
        dropped = backfill.stats[DOMAIN].dropped
        assert dropped[DropReason.BACKFILL_OFF_DOMAIN_REDIRECT.value] == 1

    def test_a_sitemap_redirect_from_apex_to_www_succeeds(self, tmp_path, web):
        web.routes[SITEMAP_2] = Response(
            200, fixture_bytes("sitemap_urlset.xml"), final_url="https://vishvasnews.com/s.xml"
        )
        backfill = make_backfill(tmp_path)
        assert len(backfill.candidates(DOMAIN)) == 4

    def test_an_article_that_ends_off_domain_is_discarded_and_not_cached(self, tmp_path, web):
        web.routes[NEW_URL] = Response(
            200, fixture_bytes("hindi_claimreview.html"), final_url="https://evil.example/x"
        )
        backfill = make_backfill(tmp_path)
        verdicts = list(backfill.run())
        assert len(verdicts) == 3
        dropped = backfill.stats[DOMAIN].dropped
        assert dropped[DropReason.BACKFILL_OFF_DOMAIN_REDIRECT.value] == 1
        assert not backfill.extractor._cache_path(NEW_URL).exists()

    def test_an_article_redirect_from_apex_to_www_succeeds(self, tmp_path, web):
        web.routes[NEW_URL] = Response(
            200,
            fixture_bytes("hindi_claimreview.html"),
            final_url="https://vishvasnews.com/viral/fact-check-new-one/",
        )
        backfill = make_backfill(tmp_path)
        assert len(list(backfill.run())) == 4

    def test_the_daily_path_does_not_check_the_final_domain(self, tmp_path, web):
        web.routes[NEW_URL] = Response(
            200, fixture_bytes("hindi_claimreview.html"), final_url="https://other.example/x"
        )
        extractor = make_backfill(tmp_path).extractor
        assert extractor.fetch_html(NEW_URL) is not None


class TestClaimLanguages:
    def test_a_marathi_row_is_excluded_and_counted_by_language(self, tmp_path, web):
        web.routes[NEWER_URL] = Response(200, fixture_bytes("marathi_claimreview.html"))
        web.routes[NEW_URL] = Response(200, fixture_bytes("english_claimreview.html"))
        backfill = make_backfill(tmp_path, make_config(block={"claim_languages": ["hi"]}))
        verdicts = list(backfill.run())
        assert {v.claim_lang for v in verdicts} == {"hi"}
        assert len(verdicts) == 2
        stats = backfill.stats[DOMAIN]
        assert stats.language_excluded == {"mr": 1, "en": 1}
        assert stats.dropped[DropReason.BACKFILL_LANGUAGE_EXCLUDED.value] == 2
        assert stats.verdicts == 2
        assert backfill.extractor.dropped[DropReason.BACKFILL_LANGUAGE_EXCLUDED.value] == 2

    def test_nothing_is_excluded_without_the_setting(self, tmp_path, web):
        web.routes[NEWER_URL] = Response(200, fixture_bytes("marathi_claimreview.html"))
        backfill = make_backfill(tmp_path)
        assert len(list(backfill.run())) == 4

    def test_the_cli_reports_excluded_languages(self, tmp_path, web, monkeypatch):
        web.routes[NEWER_URL] = Response(200, fixture_bytes("marathi_claimreview.html"))
        settings = Settings(data_dir=tmp_path)
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        config = make_config(block={"claim_languages": ["hi"]})
        monkeypatch.setattr(factcheck, "load_yaml", lambda name: config)
        result = CliRunner().invoke(cli.app, ["factchecks", "--backfill", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "'mr': 1" in result.output


class TestReportedCounts:
    def test_unreadable_lastmod_empty_and_duplicate_loc(self, tmp_path, web):
        web.routes[SITEMAP_2] = Response(200, fixture_bytes("sitemap_oddities.xml"))
        backfill = make_backfill(tmp_path)
        urls = [e.loc for e in backfill.candidates(DOMAIN)]
        stats = backfill.stats[DOMAIN]
        # The unreadable date keeps the page, as an absent date does.
        assert "https://www.vishvasnews.com/viral/fact-check-bad-date/" in urls
        assert stats.lastmod_unparseable == 1
        assert stats.empty_loc == 1
        assert stats.duplicate_loc == 1
        assert urls.count("https://www.vishvasnews.com/viral/fact-check-twice/") == 1

    def test_sitemaps_left_unread_when_the_wanted_count_is_reached(self, tmp_path, web):
        config = make_config()
        config["backfill"]["max_urls_per_publisher"] = 1
        backfill = make_backfill(tmp_path, config)
        backfill.candidates(DOMAIN)
        assert backfill.stats[DOMAIN].sitemaps_unread >= 1


class TestRun:
    def test_a_hindi_page_gives_the_same_row_as_the_rss_path(self, tmp_path, web):
        backfill = make_backfill(tmp_path)
        verdicts = list(backfill.run())
        assert len(verdicts) == 4
        row = verdicts[0].as_row()
        assert row["claim_lang"] == "hi"
        assert row["tier"] == "claimreview"
        assert row["label"] == "false"
        assert row["publisher"] == "Vishvas News"
        assert row["domain"] == DOMAIN
        assert row["record_id"] == f"news:{_article_id(NEWER_URL)}"
        assert row["reviewed_at"] == "2026-02-19"

        rss_path = FactCheckExtractor(
            settings=Settings(data_dir=tmp_path / "rss"), config=make_config()
        )
        html = fixture_bytes("hindi_claimreview.html").decode()
        reference = rss_path.from_html(
            {"id": row["record_id"], "url": NEWER_URL, "source_detail": DOMAIN}, html
        )
        assert set(row) == set(reference.as_row())
        assert {k: v for k, v in row.items() if k != "extracted_at"} == {
            k: v for k, v in reference.as_row().items() if k != "extracted_at"
        }

    def test_per_publisher_counts(self, tmp_path, web):
        backfill = make_backfill(tmp_path, known={record_for_url(STORED_URL, DOMAIN)["id"]})
        list(backfill.run())
        stats = backfill.stats[DOMAIN]
        assert (stats.discovered, stats.already_present) == (4, 1)
        assert (stats.fetched, stats.verdicts) == (3, 3)
        assert STORED_URL not in web.articles()

    def test_a_publisher_without_a_block_is_counted(self, tmp_path, web):
        backfill = make_backfill(tmp_path)
        list(backfill.run())
        assert backfill.extractor.dropped[DropReason.NO_BACKFILL_CONFIG.value] == 1

    def test_page_drops_are_counted_per_publisher(self, tmp_path, web):
        web.routes[NEW_URL] = Response(200, b"<html></html>")
        web.routes[NEWER_URL] = ConnectionError("down")
        backfill = make_backfill(tmp_path)
        verdicts = list(backfill.run())
        assert len(verdicts) == 2
        dropped = backfill.stats[DOMAIN].dropped
        assert dropped[DropReason.NO_CLAIMREVIEW.value] == 1
        assert dropped[DropReason.FETCH_FAILED.value] == 1

    def test_a_page_robots_disallow_is_counted(self, tmp_path, web):
        web.routes[ROBOTS_URL] = Response(200, b"User-agent: *\nDisallow: /viral/\n")
        backfill = make_backfill(tmp_path)
        list(backfill.run())
        # Two of the four candidate pages are under /viral/.
        assert backfill.stats[DOMAIN].dropped[DropReason.ROBOTS_DISALLOWED.value] == 2
        assert NEW_URL not in web.calls

    def test_a_rerun_creates_no_duplicates_and_fetches_nothing(self, tmp_path, web):
        settings = Settings(data_dir=tmp_path)
        first = make_backfill(tmp_path)
        write_factchecks(list(first.run()), settings)
        assert len(existing_record_ids(settings)) == 4

        web.calls.clear()
        second = make_backfill(tmp_path, known=existing_record_ids(settings))
        assert list(second.run()) == []
        assert web.articles() == []
        assert len(existing_record_ids(settings)) == 4

    def test_a_rerun_after_a_cache_hit_needs_no_page_request(self, tmp_path, web):
        list(make_backfill(tmp_path).run())
        web.calls.clear()
        again = list(make_backfill(tmp_path).run())
        assert len(again) == 4
        assert web.articles() == []


def middle_page() -> Response:
    html = fixture_bytes("hindi_claimreview.html").replace(b'"False"', b'"Misleading"')
    assert b"Misleading" in html
    return Response(200, html)


ARTICLE_NO_CLAIMREVIEW = b"<html><body><article><p>text</p></article></body></html>"
CHALLENGE_PAGE = b"<html><head><title>Just a moment...</title></head><body>cf-chl</body></html>"


def seed_ledger(
    tmp_path, backfill, urls, reason="rating_is_middle", fingerprint=None, now=None
) -> None:
    """Write ledger rows for ``urls`` as an earlier run would have."""
    rows = {
        record_for_url(u, DOMAIN)["id"]: {
            "url": u,
            "domain": DOMAIN,
            "reason": reason,
            "fingerprint": fingerprint or backfill.fingerprint(DOMAIN),
            "detail": "",
        }
        for u in urls
    }
    update_ledger(ledger_path(Settings(data_dir=tmp_path)), rows, set(), now=now)


def urls_of(verdicts) -> set[str]:
    return {v.url for v in verdicts}


class TestDroppedLedger:
    @pytest.fixture
    def small_budget(self):
        config = make_config()
        config["backfill"]["max_urls_per_publisher"] = 2
        return config

    def test_dropped_pages_are_skipped_and_the_budget_reaches_later_pages(
        self, tmp_path, web, small_budget
    ):
        web.routes[NEWER_URL] = middle_page()
        web.routes[NEW_URL] = middle_page()
        first = make_backfill(tmp_path, small_budget, ledger=True)
        assert list(first.run()) == []
        rows = read_ledger(ledger_path(Settings(data_dir=tmp_path)))
        assert {r["url"] for r in rows.values()} == {NEWER_URL, NEW_URL}
        row = rows[record_for_url(NEWER_URL, DOMAIN)["id"]]
        assert row["reason"] == DropReason.RATING_IS_MIDDLE.value
        assert row["domain"] == DOMAIN
        assert row["first_seen"].tzinfo is not None and row["last_seen"].tzinfo is not None

        second = make_backfill(tmp_path, small_budget, ledger=True)
        verdicts = list(second.run())
        stats = second.stats[DOMAIN]
        assert stats.previously_dropped == 2
        assert stats.dropped[DropReason.PREVIOUSLY_DROPPED.value] == 2
        assert second.extractor.dropped[DropReason.PREVIOUSLY_DROPPED.value] == 2
        assert urls_of(verdicts) == {STORED_URL, NO_LASTMOD_URL}
        assert stats.fetched == 2

    def test_a_ledger_hit_makes_no_request_for_the_page(self, tmp_path, web, small_budget):
        web.routes[NEWER_URL] = middle_page()
        web.routes[NEW_URL] = middle_page()
        list(make_backfill(tmp_path, small_budget, ledger=True).run())
        # The page cache would hide a request, so remove it.
        for cached in (tmp_path).rglob("*.html"):
            cached.unlink()
        web.calls.clear()
        list(make_backfill(tmp_path, small_budget, ledger=True).run())
        assert NEWER_URL not in web.calls and NEW_URL not in web.calls

    def test_language_and_no_claimreview_drops_are_recorded(self, tmp_path, web):
        web.routes[NEWER_URL] = Response(200, fixture_bytes("marathi_claimreview.html"))
        web.routes[NEW_URL] = Response(200, ARTICLE_NO_CLAIMREVIEW)
        config = make_config(block={"claim_languages": ["hi"]})
        list(make_backfill(tmp_path, config, ledger=True).run())
        rows = read_ledger(ledger_path(Settings(data_dir=tmp_path)))
        reasons = {r["url"]: r["reason"] for r in rows.values()}
        assert reasons == {
            NEWER_URL: DropReason.BACKFILL_LANGUAGE_EXCLUDED.value,
            NEW_URL: DropReason.NO_CLAIMREVIEW.value,
        }

    def test_transient_failures_are_not_recorded(self, tmp_path, web):
        web.routes[NEWER_URL] = ConnectionError("down")
        web.routes[NEW_URL] = Response(500)
        backfill = make_backfill(tmp_path, ledger=True)
        list(backfill.run())
        dropped = backfill.stats[DOMAIN].dropped
        assert dropped[DropReason.FETCH_FAILED.value] == 2
        assert not ledger_path(Settings(data_dir=tmp_path)).exists()

    def test_recheck_dropped_fetches_ledger_urls_again(self, tmp_path, web, small_budget):
        web.routes[NEWER_URL] = middle_page()
        web.routes[NEW_URL] = middle_page()
        list(make_backfill(tmp_path, small_budget, ledger=True).run())
        path = ledger_path(Settings(data_dir=tmp_path))
        first_rows = read_ledger(path)
        web.calls.clear()

        again = make_backfill(tmp_path, small_budget, ledger=True, recheck=True)
        assert list(again.run()) == []
        assert {NEWER_URL, NEW_URL} <= set(web.calls)
        assert again.stats[DOMAIN].previously_dropped == 0
        rewritten = read_ledger(path)
        for record_id, row in first_rows.items():
            assert rewritten[record_id]["first_seen"] == row["first_seen"]
            assert rewritten[record_id]["last_seen"] >= row["last_seen"]

    def test_an_unreadable_ledger_warns_and_the_run_goes_on(self, tmp_path, web, caplog):
        path = ledger_path(Settings(data_dir=tmp_path))
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a parquet file")
        with caplog.at_level("WARNING", logger="ingest.factcheck.backfill"):
            backfill = make_backfill(tmp_path, ledger=True)
            verdicts = list(backfill.run())
        assert len(verdicts) == 4
        assert DropReason.LEDGER_UNREADABLE.value in caplog.text
        assert backfill.extractor.dropped[DropReason.LEDGER_UNREADABLE.value] == 1

    def test_a_url_that_later_gives_a_verdict_leaves_the_ledger(
        self, tmp_path, web, small_budget
    ):
        web.routes[NEWER_URL] = middle_page()
        web.routes[NEW_URL] = middle_page()
        list(make_backfill(tmp_path, small_budget, ledger=True).run())
        # The publisher changes its rating.
        web.routes[NEWER_URL] = Response(200, fixture_bytes("hindi_claimreview.html"))
        again = make_backfill(tmp_path, small_budget, ledger=True, recheck=True)
        verdicts = list(again.run())
        assert urls_of(verdicts) == {NEWER_URL}
        remaining = read_ledger(ledger_path(Settings(data_dir=tmp_path)))
        assert {r["url"] for r in remaining.values()} == {NEW_URL}


SITEMAP_3 = "https://www.vishvasnews.com/post-sitemap3.xml"
SITEMAP_4 = "https://www.vishvasnews.com/post-sitemap4.xml"


def urlset(*urls: str) -> Response:
    body = "".join(
        f"<url><loc>{u}</loc><lastmod>2026-02-19T00:00:00+00:00</lastmod></url>" for u in urls
    )
    return Response(
        200,
        f'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{body}</urlset>'.encode(),
    )


def index_of(*children: str) -> Response:
    """An index that lists ``children`` newest first."""
    body = "".join(
        f"<sitemap><loc>{c}</loc><lastmod>2026-02-{20 - i:02d}T00:00:00+00:00</lastmod></sitemap>"
        for i, c in enumerate(children)
    )
    return Response(
        200,
        f'<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{body}</sitemapindex>'.encode(),
    )


class TestSitemapBudget:
    L1 = "https://www.vishvasnews.com/viral/fact-check-ledgered/"
    S1 = "https://www.vishvasnews.com/viral/fact-check-stored/"
    N1 = "https://www.vishvasnews.com/viral/fact-check-fresh/"

    def routes(self, web, children=(SITEMAP_1, SITEMAP_2, SITEMAP_3)):
        page = Response(200, fixture_bytes("hindi_claimreview.html"))
        web.routes[INDEX_URL] = index_of(*children)
        web.routes[SITEMAP_1] = urlset(self.L1)
        web.routes[SITEMAP_2] = urlset(self.S1)
        web.routes[SITEMAP_3] = urlset(self.N1)
        web.routes[SITEMAP_4] = urlset(self.N1)
        for url in (self.L1, self.S1, self.N1):
            web.routes[url] = page

    def config(self, **bounds):
        config = make_config()
        config["backfill"].update({"max_sitemaps_per_publisher": 2, "min_lastmod": None})
        config["backfill"].update(bounds)
        return config

    def test_sitemaps_with_nothing_new_do_not_use_the_sitemap_budget(self, tmp_path, web):
        self.routes(web)
        seeder = make_backfill(tmp_path, self.config())
        seed_ledger(tmp_path, seeder, [self.L1])
        known = {record_for_url(self.S1, DOMAIN)["id"]}
        backfill = make_backfill(tmp_path, self.config(), known=known, ledger=True)
        verdicts = list(backfill.run())
        assert urls_of(verdicts) == {self.N1}
        stats = backfill.stats[DOMAIN]
        assert stats.sitemaps_exhausted == 2
        assert stats.sitemaps_unread == 0
        assert DropReason.SITEMAP_BUDGET_EXHAUSTED.value not in stats.dropped

    def test_a_sitemap_with_new_urls_still_uses_the_budget(self, tmp_path, web):
        self.routes(web)
        # Every child holds a new URL, so the budget of 2 (the index and one child) ends it.
        web.routes[SITEMAP_1] = urlset(self.N1)
        backfill = make_backfill(tmp_path, self.config())
        list(backfill.run())
        assert backfill.stats[DOMAIN].sitemaps_read == 2
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_BUDGET_EXHAUSTED.value] == 2
        assert backfill.stats[DOMAIN].sitemaps_unread == 2

    def test_the_hard_fetch_cap_stops_free_sitemaps_too(self, tmp_path, web):
        self.routes(web)
        seeder = make_backfill(tmp_path, self.config())
        seed_ledger(tmp_path, seeder, [self.L1])
        config = self.config(max_sitemap_fetches_per_publisher=2)
        known = {record_for_url(self.S1, DOMAIN)["id"]}
        backfill = make_backfill(tmp_path, config, known=known, ledger=True)
        assert list(backfill.run()) == []
        assert SITEMAP_3 not in web.calls
        assert backfill.stats[DOMAIN].dropped[DropReason.SITEMAP_BUDGET_EXHAUSTED.value] == 2
        assert backfill.stats[DOMAIN].sitemaps_unread == 2


class TestLedgerSafety:
    def middle_pages(self, web):
        web.routes[NEWER_URL] = middle_page()
        web.routes[NEW_URL] = middle_page()

    @pytest.fixture
    def small_budget(self):
        config = make_config()
        config["backfill"]["max_urls_per_publisher"] = 2
        return config

    def test_a_page_that_is_not_an_article_is_suspect_and_not_recorded(self, tmp_path, web):
        web.routes[NEWER_URL] = Response(200, CHALLENGE_PAGE)
        web.routes[NEW_URL] = Response(200, b"<html><body>cookie wall</body></html>")
        backfill = make_backfill(tmp_path, ledger=True)
        list(backfill.run())
        dropped = backfill.stats[DOMAIN].dropped
        assert dropped[DropReason.NO_CLAIMREVIEW_SUSPECT.value] == 2
        assert not ledger_path(Settings(data_dir=tmp_path)).exists()
        # The suspect copies are removed, so the next run fetches them again.
        web.calls.clear()
        list(make_backfill(tmp_path, ledger=True).run())
        assert {NEWER_URL, NEW_URL} <= set(web.calls)

    def test_article_detection(self):
        assert looks_like_article("<html><article>x</article></html>")
        assert not looks_like_article("<html><body>x</body></html>")

    def test_a_real_article_with_cloudflare_scripts_or_challenge_words_is_article_like(self):
        jsd = '<script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script>'
        assert looks_like_article(f"<html>{jsd}<article>x</article></html>")
        assert looks_like_article("<article>Wait, just a moment. Attention required.</article>")

    def test_a_challenge_title_without_an_article_is_suspect(self):
        page = "<html><head><title>Just a moment...</title></head><body></body></html>"
        assert not looks_like_article(page)
        assert looks_like_challenge(page)
        assert not looks_like_challenge("<article>just a moment</article>")

    @pytest.mark.parametrize(
        "meta",
        [
            '<meta property="og:type" content="article">',
            '<meta content="article" property="og:type">',
            '<meta name="og:type" content="article" />',
        ],
    )
    def test_og_type_is_found_in_either_attribute_order(self, meta):
        assert looks_like_article(f"<html><head>{meta}</head></html>")

    def test_an_og_type_that_is_not_article_does_not_count(self):
        assert not looks_like_article('<meta property="og:type" content="website">')

    def test_a_real_article_without_claimreview_is_recorded_despite_cloudflare_scripts(
        self, tmp_path, web
    ):
        page = (
            b'<html><script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script>'
            b"<article>Just a moment of silence.</article></html>"
        )
        web.routes[NEWER_URL] = Response(200, page)
        backfill = make_backfill(tmp_path, ledger=True)
        list(backfill.run())
        rows = read_ledger(ledger_path(Settings(data_dir=tmp_path)))
        assert {r["url"]: r["reason"] for r in rows.values()} == {
            NEWER_URL: DropReason.NO_CLAIMREVIEW.value
        }

    def test_a_stale_fingerprint_is_fetched_again_and_counted(self, tmp_path, web, small_budget):
        self.middle_pages(web)
        list(make_backfill(tmp_path, small_budget, ledger=True).run())
        # The publisher rating map now counts "misleading" as false.
        changed = make_config()
        changed["backfill"]["max_urls_per_publisher"] = 2
        changed["ratings"]["misleading"] = "false"
        again = make_backfill(tmp_path, changed, ledger=True)
        verdicts = list(again.run())
        stats = again.stats[DOMAIN]
        assert urls_of(verdicts) == {NEWER_URL, NEW_URL}
        assert stats.ledger_stale == 2
        assert stats.previously_dropped == 0
        assert again.extractor.dropped[DropReason.LEDGER_STALE.value] == 2
        assert read_ledger(ledger_path(Settings(data_dir=tmp_path))) == {}

    def test_a_ledger_without_the_fingerprint_column_is_stale_not_a_crash(self, tmp_path, web):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = ledger_path(Settings(data_dir=tmp_path))
        path.parent.mkdir(parents=True)
        now = __import__("datetime").datetime.now(timezone.utc)
        old = pa.table(
            {
                "record_id": [record_for_url(NEWER_URL, DOMAIN)["id"]],
                "url": [NEWER_URL],
                "domain": [DOMAIN],
                "reason": ["rating_is_middle"],
                "first_seen": pa.array([now], pa.timestamp("us", tz="UTC")),
                "last_seen": pa.array([now], pa.timestamp("us", tz="UTC")),
            }
        )
        pq.write_table(old, path)
        backfill = make_backfill(tmp_path, ledger=True)
        verdicts = list(backfill.run())
        assert NEWER_URL in urls_of(verdicts)
        assert backfill.stats[DOMAIN].ledger_stale == 1

    def test_recheck_covers_the_least_recently_seen_first(self, tmp_path, web, small_budget):
        from datetime import datetime

        seeder = make_backfill(tmp_path, small_budget)
        seed_ledger(tmp_path, seeder, [NEWER_URL, NEW_URL])
        path = ledger_path(Settings(data_dir=tmp_path))
        # NEW_URL is the older row, although NEWER_URL is the newer page.
        update_ledger(path, {record_for_url(NEWER_URL, DOMAIN)["id"]: {
            "url": NEWER_URL, "domain": DOMAIN, "reason": "rating_is_middle",
            "fingerprint": seeder.fingerprint(DOMAIN), "detail": ""}}, set(),
            now=datetime(2026, 3, 1, tzinfo=timezone.utc))
        update_ledger(path, {record_for_url(NEW_URL, DOMAIN)["id"]: {
            "url": NEW_URL, "domain": DOMAIN, "reason": "rating_is_middle",
            "fingerprint": seeder.fingerprint(DOMAIN), "detail": ""}}, set(),
            now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        small_budget["backfill"]["max_urls_per_publisher"] = 1
        backfill = make_backfill(tmp_path, small_budget, ledger=True, recheck=True)
        list(backfill.run())
        assert web.articles()[0] == NEW_URL

    def test_a_ledger_write_failure_is_counted_and_the_run_goes_on(
        self, tmp_path, web, monkeypatch, caplog
    ):
        from ingest import factcheck_backfill

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(factcheck_backfill, "update_ledger", boom)
        self.middle_pages(web)
        backfill = make_backfill(tmp_path, ledger=True)
        with caplog.at_level("WARNING", logger="ingest.factcheck.backfill"):
            verdicts = list(backfill.run())
        assert len(verdicts) == 2
        assert backfill.extractor.dropped[DropReason.LEDGER_WRITE_FAILED.value] >= 1
        assert DropReason.LEDGER_WRITE_FAILED.value in caplog.text

    def test_the_cli_still_writes_the_verdicts_when_the_ledger_cannot_be_written(
        self, tmp_path, web, monkeypatch
    ):
        from ingest import factcheck_backfill

        settings = Settings(data_dir=tmp_path)
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(factcheck, "load_yaml", lambda name: make_config())
        monkeypatch.setattr(
            factcheck_backfill, "update_ledger", lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
        )
        web.routes[NEWER_URL] = middle_page()
        result = CliRunner().invoke(cli.app, ["factchecks", "--backfill"])
        assert result.exit_code == 0, result.output
        assert len(existing_record_ids(settings)) == 3
        assert "ledger_write_failed" in result.output

    def test_the_ledger_is_flushed_every_n_drops(self, tmp_path, web, monkeypatch):
        from ingest import factcheck_backfill

        monkeypatch.setattr(factcheck_backfill, "LEDGER_FLUSH_EVERY", 1)
        self.middle_pages(web)
        backfill = make_backfill(tmp_path, ledger=True)
        stream = backfill.run()
        next(stream)  # the first verdict arrives after both middle pages were noted
        rows = read_ledger(ledger_path(Settings(data_dir=tmp_path)))
        assert {r["url"] for r in rows.values()} == {NEWER_URL, NEW_URL}
        stream.close()

    def test_robots_disallow_and_no_rating_are_recorded(self, tmp_path, web):
        web.routes[ROBOTS_URL] = Response(200, b"User-agent: *\nDisallow: /viral/\n")
        norating = (
            b'<html><article></article><script type="application/ld+json">'
            b'{"@type":"ClaimReview","claimReviewed":"x"}</script></html>'
        )
        web.routes[NEWER_URL] = Response(200, norating)
        backfill = make_backfill(tmp_path, ledger=True)
        list(backfill.run())
        rows = read_ledger(ledger_path(Settings(data_dir=tmp_path)))
        reasons = {r["url"]: r["reason"] for r in rows.values()}
        assert reasons[NEW_URL] == DropReason.ROBOTS_DISALLOWED.value
        assert reasons[NEWER_URL] == DropReason.NO_RATING.value

    def test_a_404_is_page_gone_recorded_and_not_a_domain_failure(self, tmp_path, web):
        class Gone(Exception):
            response = type("R", (), {"status_code": 404})()

        web.routes[NEWER_URL] = Gone("HTTP 404")
        backfill = make_backfill(tmp_path, ledger=True)
        list(backfill.run())
        assert backfill.stats[DOMAIN].dropped[DropReason.PAGE_GONE.value] == 1
        assert DropReason.FETCH_FAILED.value not in backfill.stats[DOMAIN].dropped
        assert backfill.extractor._fetch_failures[DOMAIN] == 0
        rows = read_ledger(ledger_path(Settings(data_dir=tmp_path)))
        assert [r["reason"] for r in rows.values()] == [DropReason.PAGE_GONE.value]

    def gone(self):
        class Gone(Exception):
            response = type("R", (), {"status_code": 404})()

        return Gone("HTTP 404")

    def test_a_run_of_404_answers_gives_the_domain_up_and_ledgers_no_page_gone(
        self, tmp_path, web
    ):
        for url in (NEWER_URL, NEW_URL, STORED_URL, NO_LASTMOD_URL):
            web.routes[url] = self.gone()
        config = make_config()
        config["fetch"]["max_consecutive_gone_per_domain"] = 2
        backfill = make_backfill(tmp_path, config, ledger=True)
        assert list(backfill.run()) == []
        dropped = backfill.stats[DOMAIN].dropped
        assert dropped[DropReason.DOMAIN_GIVEN_UP.value] >= 1
        assert DOMAIN in backfill.extractor.gone_domains
        assert not ledger_path(Settings(data_dir=tmp_path)).exists()

    def test_a_200_resets_the_404_streak(self, tmp_path, web):
        web.routes[NEWER_URL] = self.gone()
        web.routes[STORED_URL] = self.gone()
        config = make_config()
        config["fetch"]["max_consecutive_gone_per_domain"] = 2
        backfill = make_backfill(tmp_path, config, ledger=True)
        list(backfill.run())
        assert DropReason.DOMAIN_GIVEN_UP.value not in backfill.stats[DOMAIN].dropped
        assert backfill.stats[DOMAIN].dropped[DropReason.PAGE_GONE.value] == 2

    def test_a_404_does_not_reset_the_failure_count(self, tmp_path, web):
        web.routes[NEWER_URL] = self.gone()
        extractor = make_backfill(tmp_path).extractor
        extractor.set_budget(5)
        extractor._fetch_failures[DOMAIN] = 2
        assert extractor.fetch_html(NEWER_URL, expected_domain=DOMAIN) is None
        assert extractor._fetch_failures[DOMAIN] == 2
        assert extractor._gone_streak[DOMAIN] == 1
        web.routes[NEW_URL] = Response(200, ARTICLE_NO_CLAIMREVIEW)
        assert extractor.fetch_html(NEW_URL, expected_domain=DOMAIN) is not None
        assert (extractor._fetch_failures[DOMAIN], extractor._gone_streak[DOMAIN]) == (0, 0)

    @pytest.mark.parametrize("reason", ["page_gone", "robots_disallowed"])
    def test_expiring_rows_skip_a_page_only_inside_the_ttl(self, tmp_path, web, reason):
        from datetime import datetime, timedelta

        seeder = make_backfill(tmp_path)
        seed_ledger(tmp_path, seeder, [NEWER_URL], reason=reason)
        seed_ledger(
            tmp_path,
            seeder,
            [NEW_URL],
            reason=reason,
            now=datetime.now(timezone.utc) - timedelta(days=40),
        )
        backfill = make_backfill(tmp_path, ledger=True)
        verdicts = list(backfill.run())
        stats = backfill.stats[DOMAIN]
        assert stats.previously_dropped == 1
        assert stats.ledger_stale == 1
        assert NEW_URL in urls_of(verdicts) and NEWER_URL not in urls_of(verdicts)

    def test_page_stable_rows_ignore_a_changed_config(self, tmp_path, web):
        seeder = make_backfill(tmp_path)
        seed_ledger(tmp_path, seeder, [NEWER_URL], reason="no_claimreview", fingerprint="old")
        seed_ledger(tmp_path, seeder, [NEW_URL], reason="rating_is_middle", fingerprint="old")
        backfill = make_backfill(tmp_path, ledger=True)
        verdicts = list(backfill.run())
        stats = backfill.stats[DOMAIN]
        assert (stats.previously_dropped, stats.ledger_stale) == (1, 1)
        assert NEWER_URL not in urls_of(verdicts) and NEW_URL in urls_of(verdicts)

    def test_the_fingerprint_is_computed_once_per_domain(self, tmp_path):
        backfill = make_backfill(tmp_path)
        assert backfill.fingerprint(DOMAIN) is backfill.fingerprint(DOMAIN)

    def test_an_unreadable_ledger_is_renamed_aside_before_it_is_replaced(self, tmp_path, web):
        path = ledger_path(Settings(data_dir=tmp_path))
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a parquet file")
        self.middle_pages(web)
        backfill = make_backfill(tmp_path, ledger=True)
        list(backfill.run())
        aside = list(path.parent.glob(f"{LEDGER_FILENAME}.corrupt-*"))
        assert len(aside) == 1 and aside[0].read_bytes() == b"not a parquet file"
        assert backfill.extractor.dropped[DropReason.LEDGER_QUARANTINED.value] == 1
        assert len(read_ledger(path)) == 2


class TestLedgerCli:
    runner = CliRunner()

    @pytest.fixture
    def env(self, tmp_path, web, monkeypatch):
        settings = Settings(data_dir=tmp_path)
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        config = make_config()
        config["backfill"]["max_urls_per_publisher"] = 2
        monkeypatch.setattr(factcheck, "load_yaml", lambda name: config)
        web.routes[NEWER_URL] = middle_page()
        web.routes[NEW_URL] = middle_page()
        return settings

    def test_the_second_run_reports_previously_dropped_and_adds_rows(self, env):
        first = self.runner.invoke(cli.app, ["factchecks", "--backfill"])
        assert first.exit_code == 0, first.output
        assert (env.data_dir / "factchecks" / LEDGER_FILENAME).exists()
        second = self.runner.invoke(cli.app, ["factchecks", "--backfill"])
        assert second.exit_code == 0, second.output
        assert "'previously_dropped': 2" in second.output
        assert len(existing_record_ids(env)) == 2

    def test_the_dry_run_writes_no_ledger(self, env):
        result = self.runner.invoke(cli.app, ["factchecks", "--backfill", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert not (env.data_dir / "factchecks").exists()

    def test_the_flag_needs_backfill(self, env):
        result = self.runner.invoke(cli.app, ["factchecks", "--recheck-dropped"])
        assert result.exit_code == 1

    def test_the_flag_is_in_the_help(self, env):
        result = self.runner.invoke(cli.app, ["factchecks", "--help"])
        assert "--recheck-dropped" in result.output


class TestCli:
    runner = CliRunner()

    @pytest.fixture
    def env(self, tmp_path, web, monkeypatch):
        settings = Settings(data_dir=tmp_path)
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(factcheck, "load_yaml", lambda name: make_config())
        return settings

    def test_dry_run_writes_nothing(self, env):
        result = self.runner.invoke(cli.app, ["factchecks", "--backfill", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "nothing written" in result.output
        assert not (env.data_dir / "factchecks").exists()

    def test_backfill_writes_rows_and_reports_counts(self, env):
        result = self.runner.invoke(cli.app, ["factchecks", "--backfill"])
        assert result.exit_code == 0, result.output
        assert "backfill per publisher" in result.output
        assert "Vishvas News" in result.output
        assert len(existing_record_ids(env)) == 4

    def test_limit_stops_after_n_verdicts(self, env):
        result = self.runner.invoke(cli.app, ["factchecks", "--backfill", "--limit", "2"])
        assert result.exit_code == 0, result.output
        assert len(existing_record_ids(env)) == 2

    def test_a_second_backfill_adds_nothing(self, env, web):
        self.runner.invoke(cli.app, ["factchecks", "--backfill"])
        web.calls.clear()
        result = self.runner.invoke(cli.app, ["factchecks", "--backfill"])
        assert result.exit_code == 0, result.output
        assert web.articles() == []
        assert len(existing_record_ids(env)) == 4

    def test_no_backfill_block_is_an_error(self, env, monkeypatch):
        config = make_config()
        del config["publishers"][DOMAIN]["backfill"]
        monkeypatch.setattr(factcheck, "load_yaml", lambda name: config)
        result = self.runner.invoke(cli.app, ["factchecks", "--backfill"])
        assert result.exit_code == 1
