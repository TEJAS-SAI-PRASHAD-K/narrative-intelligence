"""The migration chain must be complete, reversible, and match the models.

Autogenerate drift is the failure mode these tests exist for: a column added to
a model and never migrated produces no error at all, just a query that fails in
production against a table that never grew the column.
"""

from __future__ import annotations

import pytest

from tests.conftest_api import requires_postgres


@requires_postgres
def test_upgrade_head_creates_the_whole_schema(migrated_db):
    from app.models import import_all_models
    from sqlalchemy import inspect

    from app.db import Base, get_sync_engine

    import_all_models()
    tables = set(inspect(get_sync_engine()).get_table_names(schema=migrated_db))
    expected = set(Base.metadata.tables) | {"alembic_version"}
    assert expected <= tables, f"migrations did not create: {sorted(expected - tables)}"


@requires_postgres
def test_models_and_migrations_do_not_drift(migrated_db):
    """Autogenerate against a fully-migrated database must find nothing.

    If this fails, somebody edited a model without generating the migration.
    The diff in the assertion message names exactly what is missing.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from app.models import import_all_models

    from app.db import Base, get_sync_engine

    import_all_models()
    with get_sync_engine().connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={
                "compare_type": True,
                "include_object": _include_object,
                "version_table_schema": migrated_db,
            },
        )
        diff = compare_metadata(context, Base.metadata)

    assert not diff, f"models have drifted from the migrations: {diff}"


def _include_object(obj, name, type_, reflected, compare_to):
    """Mirror the exclusions in alembic/env.py.

    ``alembic/env.py`` is executed as a script by alembic, not importable as a
    module, so the rules are restated rather than imported. They are two lines
    and a test failure here means somebody added an exclusion in one place only,
    which is exactly what should be noticed.
    """
    if type_ == "index" and name and name.startswith("hnsw_"):
        return False
    if type_ == "table" and name in {"alembic_version", "spatial_ref_sys"}:
        return False
    return True


@requires_postgres
def test_downgrade_base_then_upgrade_head_round_trips(migrated_db):
    """Every migration is reversible, proven rather than promised."""
    from alembic.config import Config
    from sqlalchemy import inspect, text

    from alembic import command
    from app.db import get_sync_engine

    config = Config("alembic.ini")
    config.set_main_option("script_location", "alembic")

    engine = get_sync_engine()
    before = set(inspect(engine).get_table_names(schema=migrated_db))

    command.downgrade(config, "base")
    after_downgrade = set(inspect(engine).get_table_names(schema=migrated_db))
    assert after_downgrade <= {"alembic_version"}, (
        f"downgrade left tables behind: {sorted(after_downgrade - {'alembic_version'})}"
    )
    # The sequence is not a table and would not show up above, so check it too.
    with engine.connect() as connection:
        orphan = connection.execute(
            text(
                "SELECT count(*) FROM pg_sequences "
                "WHERE schemaname = :s AND sequencename = 'narrative_display_id_seq'"
            ),
            {"s": migrated_db},
        ).scalar()
    assert orphan == 0, "downgrade left narrative_display_id_seq behind"

    command.upgrade(config, "head")
    assert set(inspect(engine).get_table_names(schema=migrated_db)) == before


@requires_postgres
def test_the_hnsw_index_exists_and_is_cosine(migrated_db):
    """Mixing L2 and cosine anywhere produces subtly wrong neighbours."""
    from sqlalchemy import text

    from app.config import get_api_settings
    from app.db import get_sync_engine

    settings = get_api_settings()
    with get_sync_engine().connect() as connection:
        definition = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
            {"n": f"hnsw_post_embeddings_{settings.embedding_dim}_cosine"},
        ).scalar()

    assert definition, "the HNSW index was not created"
    assert "USING hnsw" in definition
    assert "vector_cosine_ops" in definition
    assert f"m='{settings.hnsw_m}'" in definition.replace('"', "'")


@requires_postgres
def test_reject_accounting_is_enforced_by_the_database(migrated_db):
    """`records_in = records_loaded + records_rejected`, or the insert fails.

    Silent data loss between Parquet and Postgres is the failure mode that ruins
    this project. A constraint catches a loader bug at write time; a comment
    catches it three weeks later, in a metric nobody can explain.
    """
    import uuid

    from app.models.core import Project
    from app.models.ops import IngestRun
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    from app.db import sync_session

    project_id = uuid.uuid4()
    with sync_session() as session:
        session.add(Project(id=project_id, slug=f"p-{project_id.hex[:8]}", name="t"))

    with pytest.raises(IntegrityError):
        with sync_session() as session:
            session.add(
                IngestRun(
                    project_id=project_id,
                    source="reddit",
                    records_in=100,
                    records_loaded=90,
                    records_rejected=5,  # 90 + 5 != 100: five rows unaccounted for
                )
            )

    # The honest version inserts fine.
    with sync_session() as session:
        session.add(
            IngestRun(
                project_id=project_id,
                source="reddit",
                records_in=100,
                records_loaded=95,
                records_rejected=5,
            )
        )
        # Flush explicitly: the sessionmaker has autoflush off, so without this
        # the SELECT would run before the INSERT and the test would pass for the
        # wrong reason on a constraint that never fired.
        session.flush()
        assert session.execute(text("SELECT count(*) FROM ingest_runs")).scalar() == 1


@requires_postgres
def test_engagement_nulls_survive_a_round_trip(migrated_db):
    """NULL is not 0. The single easiest place to corrupt this corpus."""
    import uuid
    from datetime import datetime, timezone

    from app.models.core import Project
    from app.models.corpus import Post

    from app.db import sync_session

    project_id = uuid.uuid4()
    with sync_session() as session:
        session.add(Project(id=project_id, slug=f"p-{project_id.hex[:8]}", name="t"))
        session.add(
            Post(
                id="reddit:abc",
                project_id=project_id,
                native_id="abc",
                source="reddit",
                source_detail="r/news",
                content_type="post",
                text_="body",
                author_id="reddit:alice",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                likes=0,  # measured zero
                shares=None,  # not measured
                replies=3,
                views=None,  # reddit exposes no view count
            )
        )

    with sync_session() as session:
        row = session.get(Post, "reddit:abc")
        assert row.likes == 0
        assert row.shares is None
        assert row.views is None
        assert row.timestamp.tzinfo is not None


@requires_postgres
def test_an_unsigned_simhash_round_trips_through_a_signed_bigint(migrated_db):
    """Postgres has no unsigned integer type.

    Phase 1 emits a uint64. Reinterpreting the same 64 bits as signed keeps the
    column fixed-width (so Hamming work stays fast) and lossless. Widening to
    NUMERIC or, worse, clamping, would break near-duplicate detection quietly.
    """
    import uuid
    from datetime import datetime, timezone

    from app.models.core import Project
    from app.models.corpus import Post

    from app.db import sync_session
    from app.etl.simhash import from_signed, to_signed

    project_id = uuid.uuid4()
    # A value above 2**63 - the half of the uint64 range a naive cast breaks on.
    unsigned = 0xFFFF_FFFF_FFFF_FFF0
    signed = to_signed(unsigned)
    assert signed < 0

    with sync_session() as session:
        session.add(Project(id=project_id, slug=f"p-{project_id.hex[:8]}", name="t"))
        session.add(
            Post(
                id="reddit:sim",
                project_id=project_id,
                native_id="sim",
                source="reddit",
                source_detail="r/news",
                content_type="post",
                text_="body",
                author_id="reddit:alice",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                simhash=signed,
            )
        )

    with sync_session() as session:
        assert from_signed(session.get(Post, "reddit:sim").simhash) == unsigned
