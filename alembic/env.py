"""Alembic environment.

Two things here are load-bearing and easy to get wrong:

1. **The URL comes from the application settings, never from alembic.ini.** One
   source of truth, and no credential in a tracked file.
2. **Every model module is imported before ``target_metadata`` is read.** A
   model that is not imported is invisible to autogenerate, and the migration it
   silently omits does not fail -- it produces a schema that is missing a table
   nobody notices until a query 404s in production.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_api_settings  # noqa: E402
from app.db import Base  # noqa: E402
from app.models import import_all_models  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

import_all_models()
target_metadata = Base.metadata


def _database_url() -> str:
    settings = get_api_settings()
    url = settings.database_url_sync or settings.require_database_url()
    # Alembic runs synchronously. If a deployment points the request path at
    # asyncpg, strip it here rather than maintaining a second URL by hand.
    return url.replace("+asyncpg", "+psycopg")


def include_object(obj, name, type_, reflected, compare_to):
    """Keep autogenerate away from things it does not own.

    pgvector's HNSW indexes are created in hand-written migrations with
    operator classes and build parameters that SQLAlchemy cannot round-trip, so
    autogenerate must not try to drop and recreate them on every run.
    """
    if type_ == "index" and name and name.startswith("hnsw_"):
        return False
    if type_ == "table" and name in {"alembic_version", "spatial_ref_sys"}:
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        # Pin the version table to the connection's own schema rather than
        # letting it be resolved through search_path. Without this, a connection
        # scoped to a throwaway schema with `public` behind it finds `public`'s
        # alembic_version, concludes it is already at head, and creates nothing
        # -- which looks exactly like a broken migration and is not.
        current_schema = connection.exec_driver_sql("SELECT current_schema()").scalar()

        # Then end the implicit transaction that probe just opened.
        #
        # This one line is load-bearing and its absence is silent. SQLAlchemy 2.0
        # begins a transaction on first execute; `context.begin_transaction()`
        # sees one already in progress and degrades to a nested no-op, so alembic
        # never commits and `connection.close()` rolls the entire upgrade back.
        # Every migration logs "Running upgrade", exits 0, and creates nothing.
        connection.rollback()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=current_schema,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
            # One transaction for the whole upgrade. A migration that fails
            # halfway must leave no partial schema behind; Postgres has
            # transactional DDL, so there is no reason not to use it.
            transaction_per_migration=False,
        )
        with context.begin_transaction():
            context.run_migrations()

        # Belt and braces. Harmless if begin_transaction already committed, and
        # the difference between a committed migration and a silently discarded
        # one is not something to leave to a framework detail.
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
