"""The Parquet -> Postgres loader.

The tests that matter here are the ones about *not losing rows*. A loader that
inserts most of the corpus and quietly drops the rest passes every smoke test
and ruins every metric downstream, so the accounting identity gets more
attention than the happy path.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.conftest_api import requires_postgres


# ---------------------------------------------------------------------------
# simhash conversion -- no database needed
# ---------------------------------------------------------------------------
def test_simhash_round_trips_across_the_whole_unsigned_range():
    from app.etl.simhash import from_signed, to_signed

    for value in (0, 1, 2**62, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 1):
        assert from_signed(to_signed(value)) == value

    # The upper half is where a naive cast silently breaks.
    assert to_signed(2**63) < 0
    assert to_signed(2**63 - 1) > 0
    assert to_signed(None) is None
    assert from_signed(None) is None


def test_simhash_rejects_a_value_that_does_not_fit():
    from app.etl.simhash import to_signed

    with pytest.raises(ValueError, match="unsigned 64-bit"):
        to_signed(2**64)


def test_hamming_distance_ignores_the_sign_convention():
    """The property that makes storing these as signed safe at all."""
    from app.etl.simhash import hamming, to_signed

    a, b = 2**63 + 0b1011, 2**63 + 0b1001
    assert hamming(a, b) == 1
    assert hamming(to_signed(a), to_signed(b)) == 1
    assert hamming(to_signed(a), b) == 1


# ---------------------------------------------------------------------------
# partition discovery -- no database needed
# ---------------------------------------------------------------------------
def _write_partition(root: Path, source: str, day: str, rows: list[dict]) -> Path:
    directory = root / f"source={source}" / f"date={day}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "part-0.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=_arrow_schema()), path)
    return path


def _arrow_schema() -> pa.Schema:
    from ingest.store import ARROW_SCHEMA

    return ARROW_SCHEMA


def _row(**overrides):
    """One valid Phase 1 row, in the on-disk shape."""
    base = {
        "id": "reddit:t3_a",
        "native_id": "t3_a",
        "source": "reddit",
        "source_detail": "r/news",
        "content_type": "post",
        "text": "a synthetic record",
        "lang": "en",
        "author_id": "reddit:alice",
        "author_handle": "alice",
        "timestamp": datetime(2026, 5, 1, 12, tzinfo=timezone.utc),
        "parent_id": None,
        "conversation_id": None,
        "engagement": {"likes": 3, "shares": None, "replies": 0, "views": None},
        "urls": ["https://example.com/a"],
        "domains": ["example.com"],
        "media_urls": [],
        "hashtags": ["#x"],
        "mentions": [],
        "simhash": 2**63 + 7,
        "ingested_at": datetime(2026, 5, 2, tzinfo=timezone.utc),
        "raw": json.dumps({"k": "v"}),
    }
    base.update(overrides)
    return base


def test_partition_discovery_filters_by_source_and_date(tmp_path):
    from app.etl.parquet_loader import discover_partitions

    root = tmp_path / "normalized"
    _write_partition(root, "reddit", "2026-05-01", [_row()])
    _write_partition(root, "reddit", "2026-06-01", [_row(id="reddit:t3_b", native_id="t3_b")])
    _write_partition(
        root,
        "mastodon",
        "2026-05-01",
        [_row(id="mastodon:1", native_id="1", source="mastodon", author_id="mastodon:bob")],
    )

    assert len(discover_partitions(root)) == 3
    assert len(discover_partitions(root, sources=["reddit"])) == 2
    assert len(discover_partitions(root, since=datetime(2026, 5, 15).date())) == 1
    assert len(discover_partitions(root, until=datetime(2026, 5, 15).date())) == 2
    assert discover_partitions(tmp_path / "nope") == []


def test_discovery_skips_an_unparseable_partition_directory(tmp_path):
    """A stray directory must not abort a load of the good ones."""
    from app.etl.parquet_loader import discover_partitions

    root = tmp_path / "normalized"
    _write_partition(root, "reddit", "2026-05-01", [_row()])
    (root / "source=reddit" / "date=not-a-date").mkdir(parents=True)
    assert len(discover_partitions(root)) == 1


# ---------------------------------------------------------------------------
# loading -- needs Postgres
# ---------------------------------------------------------------------------
@pytest.fixture
def project(migrated_db):
    from app.models.core import Project

    from app.db import sync_session

    project_id = uuid.uuid4()
    with sync_session() as session:
        session.add(Project(id=project_id, slug="test-project", name="Test"))
    return project_id


def _load(project_id, files, tmp_path, source="reddit"):
    from app.db import sync_session
    from app.etl.parquet_loader import load_source

    with sync_session() as session:
        return load_source(
            session,
            project_id=project_id,
            source=source,
            files=files,
            rejects_dir=tmp_path / "rejects",
        )


@requires_postgres
def test_a_clean_load_accounts_for_every_row(project, tmp_path):
    from app.models.corpus import Post

    from app.db import sync_session

    root = tmp_path / "normalized"
    path = _write_partition(
        root,
        "reddit",
        "2026-05-01",
        [_row(id=f"reddit:t3_{i}", native_id=f"t3_{i}") for i in range(50)],
    )

    result = _load(project, [path], tmp_path)
    assert result.records_in == 50
    assert result.records_loaded == 50
    assert result.records_rejected == 0
    result.assert_accounted()

    with sync_session() as session:
        assert session.query(Post).count() == 50


@requires_postgres
def test_rerunning_the_loader_changes_zero_rows(project, tmp_path):
    """The acceptance criterion, asserted directly.

    Not just "the count is the same" -- the second run must report zero
    insertions, so a loader that deleted and reinserted would still fail here.
    """
    from app.models.corpus import Post

    from app.db import sync_session

    root = tmp_path / "normalized"
    path = _write_partition(
        root,
        "reddit",
        "2026-05-01",
        [_row(id=f"reddit:t3_{i}", native_id=f"t3_{i}") for i in range(30)],
    )

    first = _load(project, [path], tmp_path)
    assert first.records_loaded == 30

    with sync_session() as session:
        before = {p.id: p.ingested_at for p in session.query(Post).all()}

    second = _load(project, [path], tmp_path)
    assert second.records_in == 30
    assert second.records_loaded == 0, "a rerun must insert nothing"
    assert second.records_unchanged == 30
    second.assert_accounted()

    with sync_session() as session:
        after = {p.id: p.ingested_at for p in session.query(Post).all()}
    assert before == after, "a rerun must not rewrite existing rows"


@requires_postgres
def test_null_engagement_is_never_coalesced_to_zero(project, tmp_path):
    """The single easiest place to silently corrupt this corpus."""
    from app.models.corpus import Post

    from app.db import sync_session

    root = tmp_path / "normalized"
    path = _write_partition(
        root,
        "reddit",
        "2026-05-01",
        [
            _row(
                id="reddit:nulls",
                native_id="nulls",
                engagement={"likes": 0, "shares": None, "replies": None, "views": None},
            )
        ],
    )
    _load(project, [path], tmp_path)

    with sync_session() as session:
        row = session.get(Post, "reddit:nulls")
    assert row.likes == 0, "a measured zero must stay zero"
    assert row.shares is None, "an unmeasured metric must stay NULL, not become 0"
    assert row.replies is None
    assert row.views is None


@requires_postgres
def test_an_unsigned_simhash_survives_the_load(project, tmp_path):
    from app.models.corpus import Post

    from app.db import sync_session
    from app.etl.simhash import from_signed

    root = tmp_path / "normalized"
    unsigned = 2**64 - 3
    path = _write_partition(
        root, "reddit", "2026-05-01", [_row(id="reddit:sim", native_id="sim", simhash=unsigned)]
    )
    _load(project, [path], tmp_path)

    with sync_session() as session:
        stored = session.get(Post, "reddit:sim").simhash
    assert stored < 0
    assert from_signed(stored) == unsigned


@requires_postgres
def test_invalid_rows_are_rejected_by_reason_code_and_written_to_disk(project, tmp_path):
    """Silent data loss is the failure mode that ruins this project."""
    from app.models.corpus import Post

    from app.db import sync_session

    root = tmp_path / "normalized"
    rows = [
        _row(id="reddit:ok", native_id="ok"),
        # Phase 1's adapters drop these before a Record is ever built, so the
        # model does not reject them. At this boundary there is no adapter, and
        # the loader has to apply the same rule with the same reason code.
        _row(id="reddit:empty", native_id="empty", text="   "),
        # A content_type this schema version does not know. Realistic: Phase 1
        # gains a new type and Phase 4 has not been redeployed yet. Rejected
        # loudly with a code rather than loaded as an unknown kind of thing.
        _row(id="reddit:newtype", native_id="newtype", content_type="livestream"),
    ]
    path = _write_partition(root, "reddit", "2026-05-01", rows)

    result = _load(project, [path], tmp_path)
    assert result.records_in == 3
    assert result.records_loaded == 1
    assert result.records_rejected == 2
    assert result.rejection_reasons == {"empty_text": 1, "unsupported_type": 1}
    result.assert_accounted()

    # Every rejected row is on disk with its reason, not merely counted.
    assert result.rejects_path
    rejects = pq.read_table(result.rejects_path).to_pylist()
    assert len(rejects) == 2
    by_id = {r["id"]: r for r in rejects}
    assert by_id["reddit:empty"]["reason"] == "empty_text"
    assert by_id["reddit:newtype"]["reason"] == "unsupported_type"
    assert all(r["detail"] and r["row"] for r in rejects)

    with sync_session() as session:
        assert session.query(Post).count() == 1


@requires_postgres
def test_duplicate_ids_within_one_batch_collapse_without_error(project, tmp_path):
    """Two partitions can legitimately carry the same article.

    ON CONFLICT cannot resolve a conflict inside its own INSERT -- it raises
    "cannot affect row a second time" -- so the loader de-duplicates first. An
    RSS feed re-fetched on consecutive days produces exactly this.
    """
    from app.models.corpus import Post

    from app.db import sync_session

    root = tmp_path / "normalized"
    article = _row(
        id="news:a",
        native_id="a",
        source="news",
        source_detail="bbc.co.uk",
        content_type="article",
        author_id="news:bbc.co.uk",
        author_handle="A Reporter",
    )
    first = _write_partition(root, "news", "2026-05-01", [article])
    second = _write_partition(root, "news", "2026-05-02", [dict(article)])

    result = _load(project, [first, second], tmp_path, source="news")
    assert result.records_in == 2
    assert result.records_loaded == 1
    assert result.records_unchanged == 1
    result.assert_accounted()

    with sync_session() as session:
        assert session.query(Post).count() == 1


@requires_postgres
def test_accounting_failure_is_loud(project, tmp_path):
    """The guard itself. If this can be silenced, none of the rest holds."""
    from app.etl.parquet_loader import LoadResult

    broken = LoadResult(source="reddit", records_in=100, records_loaded=90, records_rejected=5)
    with pytest.raises(RuntimeError, match="silent-data-loss"):
        broken.assert_accounted()

    uncoded = LoadResult(
        source="reddit",
        records_in=10,
        records_loaded=8,
        records_rejected=2,
        rejection_reasons={"empty_text": 1},
    )
    with pytest.raises(RuntimeError, match="reason code"):
        uncoded.assert_accounted()


@requires_postgres
def test_author_and_domain_rollups_derive_from_posts(project, tmp_path):
    from app.models.actors import Author
    from app.models.domains import Domain

    from app.db import sync_session
    from app.etl.derive import derive_authors, derive_domains

    root = tmp_path / "normalized"
    rows = [
        _row(
            id="reddit:1",
            native_id="1",
            author_id="reddit:alice",
            author_handle="alice_old",
            timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
            domains=["a.com", "b.com"],
        ),
        _row(
            id="reddit:2",
            native_id="2",
            author_id="reddit:alice",
            author_handle="alice_new",
            timestamp=datetime(2026, 5, 9, tzinfo=timezone.utc),
            domains=["a.com"],
        ),
        _row(
            id="reddit:3",
            native_id="3",
            author_id="reddit:bob",
            author_handle="bob",
            timestamp=datetime(2026, 5, 5, tzinfo=timezone.utc),
            domains=["A.COM"],
        ),
    ]
    path = _write_partition(root, "reddit", "2026-05-01", rows)
    _load(project, [path], tmp_path)

    with sync_session() as session:
        derive_authors(session, project)
        derive_domains(session, project)

    with sync_session() as session:
        alice = session.get(Author, "reddit:alice")
        assert alice.post_count == 2
        assert alice.first_seen == datetime(2026, 5, 1, tzinfo=timezone.utc)
        assert alice.last_seen == datetime(2026, 5, 9, tzinfo=timezone.utc)
        # The most recent handle, not an arbitrary one: it is what an analyst
        # will recognise.
        assert alice.handle == "alice_new"

        # Domains are lowercased on roll-up, so A.COM and a.com are one actor.
        a_com = session.get(Domain, ("a.com", project))
        assert a_com.post_count == 3
        assert a_com.author_count == 2
        assert session.get(Domain, ("A.COM", project)) is None
        assert a_com.enrichment_status == "pending"


@requires_postgres
def test_rederiving_does_not_blank_scores_or_requeue_enrichment(project, tmp_path):
    """A reload must not undo work the scoring and enrichment tasks did.

    Re-running the roll-up with DO UPDATE over every column would wipe bot
    scores and send every already-enriched domain back to the rate-limited
    WHOIS queue on each ingest.
    """
    from app.models.actors import Author
    from app.models.domains import Domain

    from app.db import sync_session, utcnow
    from app.etl.derive import derive_authors, derive_domains

    root = tmp_path / "normalized"
    path = _write_partition(root, "reddit", "2026-05-01", [_row(id="reddit:1", native_id="1")])
    _load(project, [path], tmp_path)
    with sync_session() as session:
        derive_authors(session, project)
        derive_domains(session, project)

    with sync_session() as session:
        session.get(Author, "reddit:alice").bot_score = 0.91
        domain = session.get(Domain, ("example.com", project))
        domain.enrichment_status = "enriched"
        domain.risk_score = 77.0
        domain.enriched_at = utcnow()

    # A second load and re-derivation, exactly as a scheduled ingest would do.
    _load(project, [path], tmp_path)
    with sync_session() as session:
        derive_authors(session, project)
        derive_domains(session, project)

    with sync_session() as session:
        assert session.get(Author, "reddit:alice").bot_score == 0.91
        domain = session.get(Domain, ("example.com", project))
        assert domain.enrichment_status == "enriched"
        assert domain.risk_score == 77.0
