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

from alembic import context
from sqlalchemy import engine_from_config, pool

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
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
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


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
