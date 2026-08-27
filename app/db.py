"""Engine, session and declarative base.

Async everywhere on the request path, because the API's work is almost entirely
waiting on Postgres and an async pool serves far more concurrent dashboard
requests per container than a thread pool of the same size.

Two engines, deliberately:

* the **async** engine for FastAPI routes;
* a **sync** engine for Alembic, the Celery workers and the ETL's ``COPY``.
  Celery's prefork model and psycopg's ``copy()`` are both synchronous, and
  bridging them onto the async loop buys nothing but a class of hard-to-debug
  event-loop errors.

There is no ``Base.metadata.create_all`` anywhere in this package. Every schema
change is an Alembic migration; a table that appears because an import ran is a
table nobody reviewed.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import DateTime, MetaData, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import get_api_settings

log = logging.getLogger(__name__)

#: Explicit constraint naming, so Alembic autogenerate produces stable,
#: reversible migrations. Without this, unnamed constraints get server-assigned
#: names and a downgrade cannot find what to drop.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    #: Every timestamp column in this schema is timezone-aware. Phase 1 rejects
    #: naive datetimes at the adapter boundary; this keeps that guarantee all
    #: the way to the wire.
    type_annotation_map = {datetime: DateTime(timezone=True)}

    def as_dict(self) -> dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def utcnow() -> datetime:
    """Timezone-aware now. Never use ``datetime.utcnow()`` in this codebase."""
    return datetime.now(timezone.utc)


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()


def _connect_args(url: str, *settings_options: str) -> dict[str, str]:
    """Merge libpq ``options`` from the URL with the ones we set here.

    SQLAlchemy folds a URL's query string into the driver's connect kwargs, and
    an explicit ``connect_args={"options": ...}`` then *replaces* it wholesale
    rather than merging. Anything that scopes a connection through the URL --
    a search_path for a throwaway test schema, a pooler's session settings --
    is silently dropped, and the connection quietly talks to the default schema
    instead. It fails as "relation does not exist" against a database whose
    migrations demonstrably ran, which is a genuinely bad afternoon.

    Concatenating is the fix: libpq applies the flags left to right.
    """
    from urllib.parse import parse_qs, urlsplit

    from_url = parse_qs(urlsplit(url).query).get("options", [])
    merged = " ".join([*from_url, *settings_options]).strip()
    return {"options": merged} if merged else {}


# ---------------------------------------------------------------------------
# async engine (request path)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    settings = get_api_settings()
    url = settings.require_database_url()
    engine = create_async_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        # A connection that has been idle long enough for a proxy to have
        # dropped it looks alive to the pool and dies on first use.
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=False,
        future=True,
        connect_args=_connect_args(
            url,
            # A runaway dashboard query must not be able to hold a connection
            # open forever. This is the backstop; the real fix is an index.
            f"-c statement_timeout={settings.db_statement_timeout_ms}",
            "-c timezone=UTC",
        ),
    )
    return engine


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,  # response models read attributes after commit
        autoflush=False,
    )


async def get_session() -> AsyncIterator[AsyncSession | None]:
    """FastAPI dependency. One session per request, rolled back on failure.

    In DEMO_MODE this yields ``None`` and opens no connection. That is not a
    shortcut -- it is the point of the mode: the contract build has to run with
    no Postgres anywhere, so a frontend developer can `docker run` one container
    and start work. Every router's demo branch returns before touching the
    session, and the contract test asserts that by running the entire surface
    with DATABASE_URL unset.
    """
    if get_api_settings().demo_mode:
        yield None
        return

    sessionmaker_ = get_sessionmaker()
    async with sessionmaker_() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# sync engine (alembic, celery, etl COPY)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_sync_engine():
    from sqlalchemy import create_engine

    settings = get_api_settings()
    url = (settings.database_url_sync or settings.require_database_url()).replace(
        "+asyncpg", "+psycopg"
    )
    return create_engine(
        url,
        # NullPool in the worker: a prefork child that inherits a pooled
        # connection from its parent shares a socket with a sibling, and the
        # resulting protocol desync is the classic "worker returns another
        # task's rows" bug. Connections are cheap here; correctness is not.
        poolclass=NullPool,
        future=True,
        connect_args=_connect_args(url, "-c timezone=UTC"),
    )


@lru_cache(maxsize=1)
def get_sync_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_sync_engine(), expire_on_commit=False, autoflush=False)


@contextmanager
def sync_session() -> Iterator[Session]:
    """Worker-side session. Commits on clean exit, rolls back on exception."""
    session = get_sync_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def advisory_lock(session: Session, key: str) -> Iterator[bool]:
    """A Postgres session-level advisory lock, keyed by a string.

    This is how a task stays idempotent under a concurrent retry: two workers
    that both pick up ``score.fusion`` for the same project serialise here
    instead of racing to write the same scorecard rows. Returns False without
    blocking if another holder has it, so the caller can skip rather than queue.
    """
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    lock_id = int.from_bytes(digest[:8], "big", signed=True)
    acquired = bool(
        session.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": lock_id}).scalar()
    )
    try:
        yield acquired
    finally:
        if acquired:
            try:
                session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": lock_id})
            except Exception as unlock_error:
                # If the body failed, the transaction is already aborted and
                # this unlock raises InFailedSqlTransaction -- which would then
                # replace the original exception and hide what actually went
                # wrong. Swallow it and let the real error propagate.
                #
                # Not leaking the lock: it is session-scoped, and the session is
                # closed by the caller's context manager immediately after this,
                # which releases it.
                log.debug("could not release advisory lock %s: %s", key, unlock_error)
