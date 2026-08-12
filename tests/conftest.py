"""Shared fixtures. No test in this suite is allowed to touch the network."""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ingest.config import Settings
from ingest.schema import Record, make_id

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test opens a socket.

    The acceptance criterion is "pytest passes with no live network calls". The
    honest way to enforce that is to make a live call impossible, not to promise
    it in a docstring.
    """

    def guard(*args, **kwargs):
        raise RuntimeError(
            "network access attempted in a test; record a fixture under fixtures/ instead"
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at a throwaway data dir, with no credentials."""
    return Settings(
        data_dir=tmp_path / "data",
        mastodon_access_token=None,
        youtube_api_key=None,
        newsapi_key=None,
        _env_file=None,
    )


@pytest.fixture
def sample_record() -> Record:
    return Record(
        native_id="t3_abc",
        source="reddit",
        source_detail="news",
        content_type="post",
        text="a synthetic record used to prove the pipeline round-trips",
        lang="en",
        author_id=make_id("reddit", "alice"),
        author_handle="alice",
        timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
        engagement={"likes": 12, "replies": 3},
        urls=["https://example.com/a"],
        domains=["example.com"],
        hashtags=["election"],
        simhash=12345678901234567890,
        raw={"score": 12, "subreddit": "news", "nested": {"a": [1, 2]}},
    )


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
