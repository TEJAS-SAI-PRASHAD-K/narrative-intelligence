"""CLI behaviour, especially graceful degradation.

``fetch-all`` must run to completion with no credentials at all, skipping what
it cannot do. That is an acceptance criterion, so it is a test.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from ingest import cli
from ingest.schema import Record, make_id
from ingest.sources.base import BaseSource, SourceUnavailable
from ingest.store import ParquetStore

TS = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
runner = CliRunner()


@pytest.fixture
def cli_env(settings, monkeypatch):
    """Point the CLI at a throwaway data dir with no credentials."""
    settings.ensure_dirs()
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


def seed(settings, n: int = 3, source: str = "reddit") -> ParquetStore:
    store = ParquetStore(settings)
    store.write_records(
        [
            Record(
                native_id=f"id{i}",
                source=source,
                source_detail="news",
                content_type="post",
                text=f"seeded record {i} with enough text to be interesting",
                lang="en",
                author_id=make_id(source, f"a{i}"),
                timestamp=TS,
                parent_id=make_id(source, "root") if i else None,
                simhash=i,
                raw={"i": i},
            )
            for i in range(n)
        ]
    )
    return store


class FakeSource(BaseSource):
    name = "reddit_convokit"
    source = "reddit"

    def fetch(self) -> Iterator[dict]:
        yield {"id": "x1"}

    def to_record(self, raw: dict) -> Record:
        return Record(
            native_id=raw["id"],
            source="reddit",
            source_detail="news",
            content_type="post",
            text="a record produced without touching the network",
            author_id=make_id("reddit", "someone"),
            timestamp=TS,
        )


class UnavailableSource(FakeSource):
    name = "youtube"
    source = "youtube"

    def preflight(self) -> None:
        raise SourceUnavailable("youtube: YOUTUBE_API_KEY absent; skipping")


class TestStats:
    def test_empty_corpus_message(self, cli_env):
        result = runner.invoke(cli.app, ["stats"])
        assert result.exit_code == 0
        assert "no corpus on disk" in result.stdout

    def test_summary_table(self, cli_env):
        seed(cli_env, n=4)
        result = runner.invoke(cli.app, ["stats"])
        assert result.exit_code == 0
        assert "reddit" in result.stdout
        assert "total records: 4" in result.stdout

    def test_scoped_to_one_source(self, cli_env):
        seed(cli_env, n=2, source="reddit")
        seed(cli_env, n=3, source="news")
        assert "total records: 3" in runner.invoke(cli.app, ["stats", "news"]).stdout


class TestValidate:
    def test_valid_corpus_passes(self, cli_env):
        seed(cli_env, n=3)
        result = runner.invoke(cli.app, ["validate"])
        assert result.exit_code == 0
        assert "schema-valid" in result.stdout

    def test_partition_columns_are_not_treated_as_schema_fields(self, cli_env):
        seed(cli_env, n=1)
        assert runner.invoke(cli.app, ["validate"]).exit_code == 0

    def test_empty_corpus_is_not_a_failure(self, cli_env):
        assert runner.invoke(cli.app, ["validate"]).exit_code == 0

    def test_sample_flag(self, cli_env):
        seed(cli_env, n=5)
        result = runner.invoke(cli.app, ["validate", "--sample", "2"])
        assert "2 rows checked" in result.stdout


class TestManifest:
    def test_absent_manifest(self, cli_env):
        assert "no manifest yet" in runner.invoke(cli.app, ["manifest"]).stdout

    def test_lists_artifacts(self, cli_env, tmp_path):
        artifact = tmp_path / "corpus.jsonl"
        artifact.write_text('{"a": 1}\n', encoding="utf-8")
        ParquetStore(cli_env).manifest.record_artifact(
            "reddit_convokit:reddit-corpus-small",
            path=artifact,
            url="https://zissou.infosci.cornell.edu/convokit/datasets/reddit-corpus-small/",
            rows=1,
        )
        # Wide terminal so rich does not wrap the artifact key mid-word.
        result = runner.invoke(cli.app, ["manifest"], env={"COLUMNS": "200"})
        assert "reddit_convokit:reddit-corpus-small" in result.stdout
        assert "zissou" not in result.stdout  # url is in the manifest file, not the table


class TestShowConfig:
    def test_reports_presence_never_values(self, cli_env, monkeypatch):
        monkeypatch.setattr(cli_env, "youtube_api_key", "SECRET-VALUE-DO-NOT-PRINT")
        result = runner.invoke(cli.app, ["show-config"])
        assert result.exit_code == 0
        assert "SECRET-VALUE" not in result.stdout
        assert "set" in result.stdout
        assert "absent (source will skip)" in result.stdout


class TestFetch:
    def test_unknown_source_is_rejected(self, cli_env):
        result = runner.invoke(cli.app, ["fetch", "twitter"])
        assert result.exit_code != 0

    def test_single_source(self, cli_env, monkeypatch):
        monkeypatch.setattr(cli, "get_source_class", lambda name: FakeSource)
        result = runner.invoke(cli.app, ["fetch", "reddit_convokit"])
        assert result.exit_code == 0
        assert len(ParquetStore(cli_env).read_all("reddit")) == 1


class TestFetchAll:
    def test_runs_to_completion_when_every_source_is_unavailable(self, cli_env, monkeypatch):
        # The acceptance criterion: no credentials anywhere, still exit 0.
        monkeypatch.setattr(cli, "get_source_class", lambda name: UnavailableSource)
        result = runner.invoke(cli.app, ["fetch-all"])
        assert result.exit_code == 0
        assert "skipped" in result.stdout
        assert "YOUTUBE_API_KEY absent" in result.stdout

    def test_mixed_availability_still_writes_what_it_can(self, cli_env, monkeypatch):
        classes = {"reddit_convokit": FakeSource, "youtube": UnavailableSource}
        monkeypatch.setattr(cli, "get_source_class", lambda name: classes[name])
        result = runner.invoke(cli.app, ["fetch-all", "--only", "reddit_convokit,youtube"])
        assert result.exit_code == 0
        assert len(ParquetStore(cli_env).read_all("reddit")) == 1

    def test_rerun_does_not_duplicate(self, cli_env, monkeypatch):
        monkeypatch.setattr(cli, "get_source_class", lambda name: FakeSource)
        runner.invoke(cli.app, ["fetch-all", "--only", "reddit_convokit"])
        runner.invoke(cli.app, ["fetch-all", "--only", "reddit_convokit"])
        assert len(ParquetStore(cli_env).read_all("reddit")) == 1

    def test_a_failing_source_exits_nonzero(self, cli_env, monkeypatch):
        class Broken(FakeSource):
            def fetch(self):
                raise RuntimeError("upstream exploded")
                yield  # pragma: no cover

        monkeypatch.setattr(cli, "get_source_class", lambda name: Broken)
        result = runner.invoke(cli.app, ["fetch-all", "--only", "reddit_convokit"])
        assert result.exit_code == 1
        assert "error" in result.stdout


class TestMastodonStream:
    def test_skips_cleanly_without_a_token(self, cli_env):
        result = runner.invoke(cli.app, ["mastodon-stream", "--minutes", "0.01"])
        assert result.exit_code == 0
        assert "skipping stream" in result.stdout
