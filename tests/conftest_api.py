"""Fixtures for the Phase 4 backend tests.

Imported by tests/conftest.py so the Phase 1/2 suite is untouched.

Two rules this harness enforces, both of them acceptance criteria:

* **No live network and no GPU.** Phase 1's ``no_network`` autouse fixture
  already makes a socket call impossible; the API tests need a database socket,
  so they opt out of it explicitly and only for loopback.
* **Model calls are stubbed at the nlp/interfaces.py boundary.** Nothing in the
  test suite loads a checkpoint. ``stub_capabilities`` fakes the availability
  probe, and the fake scorers satisfy the Protocols.

A test that needs Postgres is marked ``@pytest.mark.postgres`` and skips with a
clear message when ``TEST_DATABASE_URL`` is unset, so `pytest` on a laptop with
no Docker still passes rather than erroring.
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterator

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL")

requires_postgres = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a Postgres with pgvector to run these",
)


@pytest.fixture
def allow_local_sockets(monkeypatch):
    """Undo the global network ban for loopback only.

    Phase 1's guard exists so no test can quietly hit a live API. A test
    database on 127.0.0.1 is not that, and re-allowing exactly loopback keeps
    the guarantee that matters while letting the schema tests run.
    """
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def guard_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError(f"non-loopback network access attempted in a test: {address!r}")
        return real_connect(self, address, *args, **kwargs)

    def guard_create(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError(f"non-loopback network access attempted in a test: {address!r}")
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guard_connect)
    monkeypatch.setattr(socket, "create_connection", guard_create)


@pytest.fixture
def api_env(monkeypatch, tmp_path) -> Iterator[None]:
    """Point app.config at the test datastores and clear every settings cache."""
    from app.config import get_api_settings

    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL or "")
    monkeypatch.setenv("REDIS_URL", TEST_REDIS_URL or "")
    monkeypatch.setenv("API_KEY_PEPPER", "test-pepper-not-a-real-one")
    monkeypatch.setenv("DEMO_MODE", "0")
    monkeypatch.setenv("ENVIRONMENT", "ci")
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))

    _clear_caches()
    get_api_settings.cache_clear()
    yield
    _clear_caches()


def _clear_caches() -> None:
    """Every ``lru_cache``d factory that closes over settings.

    Missing one of these produces the worst kind of test failure: a suite that
    passes alone and fails in a different order, because a cached engine is
    still pointing at the previous test's database.
    """
    from app.config import get_api_settings
    from app.db import get_engine, get_sessionmaker, get_sync_engine, get_sync_sessionmaker
    from app.redis_client import get_redis, get_sync_redis
    from nlp.availability import _probe_cached

    for cached in (
        get_api_settings,
        get_engine,
        get_sessionmaker,
        get_sync_engine,
        get_sync_sessionmaker,
        get_redis,
        get_sync_redis,
        _probe_cached,
    ):
        cached.cache_clear()


@pytest.fixture
def migrated_db(api_env, allow_local_sockets) -> Iterator[str]:
    """A throwaway schema with every migration applied, dropped afterwards.

    A schema rather than a database: creating a database needs a connection to
    another one and cannot run inside a transaction, while ``CREATE SCHEMA`` is
    cheap, transactional and gives the same isolation between test runs.
    """
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is unset")

    from alembic.config import Config
    from sqlalchemy import create_engine, text

    from alembic import command

    schema = f"test_{uuid.uuid4().hex[:12]}"
    sync_url = TEST_DATABASE_URL.replace("+asyncpg", "+psycopg")
    engine = create_engine(sync_url, future=True)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    scoped_url = _with_search_path(sync_url, schema)
    os.environ["DATABASE_URL"] = _with_search_path(TEST_DATABASE_URL, schema)
    os.environ["DATABASE_URL_SYNC"] = scoped_url
    _clear_caches()

    config = Config("alembic.ini")
    config.set_main_option("script_location", "alembic")
    try:
        command.upgrade(config, "head")
        yield schema
    finally:
        _clear_caches()
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        engine.dispose()
        os.environ.pop("DATABASE_URL_SYNC", None)


def _with_search_path(url: str, schema: str) -> str:
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}options=-csearch_path%3D{schema}"


@pytest.fixture
def client(migrated_db):
    """A TestClient against the migrated throwaway schema."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def admin_key(migrated_db) -> str:
    """An admin-scoped key, minted through the same code path the CLI uses."""
    return _mint("test-admin", ["read", "write", "admin"])


@pytest.fixture
def read_key(migrated_db) -> str:
    return _mint("test-read", ["read"])


@pytest.fixture
def write_key(migrated_db) -> str:
    return _mint("test-write", ["read", "write"])


def _mint(name: str, scopes: list[str]) -> str:
    from app.db import sync_session
    from app.models.ops import ApiKey
    from app.security import mint

    minted = mint()
    with sync_session() as session:
        session.add(
            ApiKey(name=name, key_hash=minted.key_hash, prefix=minted.prefix, scopes=scopes)
        )
    return minted.plaintext
