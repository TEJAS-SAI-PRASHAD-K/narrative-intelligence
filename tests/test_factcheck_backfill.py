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
    SitemapBackfill,
    existing_record_ids,
    parse_lastmod,
    parse_sitemap,
    record_for_url,
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
    def __init__(self, status: int = 200, content: bytes = b""):
        self.status_code = status
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

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


def make_backfill(tmp_path, config=None, known=None) -> SitemapBackfill:
    extractor = FactCheckExtractor(
        settings=Settings(data_dir=tmp_path), config=config or make_config()
    )
    return SitemapBackfill(extractor, known if known is not None else set())


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
