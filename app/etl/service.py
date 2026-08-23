"""Orchestration for the ETL: one transaction, one ingest_runs row per source.

Kept apart from ``parquet_loader`` (which owns COPY) and ``derive`` (which owns
the aggregates) so that the transaction boundary is visible in one place. The
rule it enforces: **a source's data and the row describing that load commit
together or not at all.** An ingest_runs row claiming 4,190 loaded rows next to
an empty posts table is worse than a failed load, because it is a lie the rest
of the system will believe.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import get_api_settings
from app.db import utcnow
from app.etl.derive import derive_authors, derive_domains, merge_author_profiles
from app.etl.parquet_loader import (
    LoadResult,
    count_distinct_parquet_ids,
    count_parquet_rows,
    discover_partitions,
    load_source,
)

log = logging.getLogger(__name__)

SOURCES = ("reddit", "mastodon", "news", "gdelt", "youtube")


@dataclass
class CorpusLoadReport:
    project_id: str
    per_source: dict[str, LoadResult] = field(default_factory=dict)
    authors_touched: int = 0
    domains_touched: int = 0
    profiles_merged: int = 0

    @property
    def records_in(self) -> int:
        return sum(r.records_in for r in self.per_source.values())

    @property
    def records_loaded(self) -> int:
        return sum(r.records_loaded for r in self.per_source.values())

    @property
    def records_rejected(self) -> int:
        return sum(r.records_rejected for r in self.per_source.values())

    @property
    def records_unchanged(self) -> int:
        return sum(r.records_unchanged for r in self.per_source.values())


def resolve_project(session: Session, identifier: str) -> uuid.UUID:
    """Slug or uuid to a project id, or raise with something actionable."""
    from app.models.core import Project

    row = session.execute(select(Project).where(Project.slug == identifier)).scalar_one_or_none()
    if row is not None:
        return row.id
    try:
        parsed = uuid.UUID(identifier)
    except ValueError:
        available = [slug for (slug,) in session.execute(select(Project.slug)).all()]
        raise RuntimeError(
            f"no project with slug or id {identifier!r}. "
            f"Existing projects: {available or '(none -- create one first)'}"
        ) from None
    if session.get(Project, parsed) is None:
        raise RuntimeError(f"no project with id {parsed}")
    return parsed


def load_corpus(
    session: Session,
    *,
    project: str,
    sources: list[str] | None = None,
    since: date | None = None,
    until: date | None = None,
    manifest_sha: str | None = None,
) -> CorpusLoadReport:
    """Load the Phase 1 corpus for one project.

    Idempotent: rerunning over the same partitions inserts nothing and updates
    the roll-ups to the same values. ``tests/test_etl.py`` asserts the zero-row
    rerun rather than trusting this docstring.
    """
    settings = get_api_settings()
    project_id = resolve_project(session, project)
    report = CorpusLoadReport(project_id=str(project_id))

    wanted = sources or list(SOURCES)
    for source in wanted:
        files = discover_partitions(
            settings.normalized_dir, sources=[source], since=since, until=until
        )
        if not files:
            log.info("%s: no partitions match; skipping", source)
            continue

        run_id = uuid.uuid4()
        started = utcnow()
        result = load_source(
            session,
            project_id=project_id,
            source=source,
            files=files,
            rejects_dir=settings.rejects_dir,
            run_id=run_id,
            batch_rows=settings.etl_copy_batch_rows,
        )
        report.per_source[source] = result
        _record_run(
            session,
            run_id=run_id,
            project_id=project_id,
            result=result,
            started_at=started,
            manifest_sha=manifest_sha,
        )

    # Order is load-bearing: authors aggregate from posts, domains unnest from
    # posts, and the author profiles merge on top of the authors that the
    # aggregate just created.
    report.authors_touched = derive_authors(session, project_id)
    report.profiles_merged = merge_author_profiles(session, project_id, settings.authors_dir)
    report.domains_touched = derive_domains(session, project_id)

    _warn_on_partition_threshold(session, project_id)
    return report


def _record_run(
    session: Session,
    *,
    run_id: uuid.UUID,
    project_id: uuid.UUID,
    result: LoadResult,
    started_at,
    manifest_sha: str | None,
) -> None:
    from app.models.ops import IngestRun

    session.add(
        IngestRun(
            id=run_id,
            project_id=project_id,
            source=result.source,
            mode="load",
            status="succeeded" if not result.records_rejected else "partial",
            records_in=result.records_in,
            # The CHECK constraint requires in = loaded + rejected, and rows
            # that were already present are neither newly loaded nor rejected.
            # Counting them as loaded is the honest reading: they are in the
            # table, this run accounted for them, and a reload of unchanged data
            # is not a rejection.
            records_loaded=result.records_loaded + result.records_unchanged,
            records_rejected=result.records_rejected,
            rejection_reasons=result.rejection_reasons or None,
            rejects_path=result.rejects_path,
            manifest_sha=manifest_sha,
            detail=(
                f"{result.files_read} parquet file(s); "
                f"{result.records_loaded} inserted, {result.records_unchanged} already present"
            ),
            started_at=started_at,
            finished_at=utcnow(),
        )
    )


def _warn_on_partition_threshold(session: Session, project_id: uuid.UUID) -> None:
    """Say something once the corpus is big enough for partitioning to pay.

    Monthly RANGE partitioning of ``posts`` costs more than it saves below a few
    million rows -- more planning time, more DDL, a partition key on every
    unique constraint -- so it is deliberately not done here. This makes the
    decision visible instead of leaving it as an unexamined default.
    """
    settings = get_api_settings()
    count = session.execute(
        text("SELECT count(*) FROM posts WHERE project_id = :p"), {"p": project_id}
    ).scalar()
    if count and count > settings.partition_threshold_rows:
        log.warning(
            "posts has %d rows for this project, above the %d threshold where monthly "
            "RANGE partitioning starts to pay for itself. See docs/data-model.md.",
            count,
            settings.partition_threshold_rows,
        )


def verify(session: Session, *, project: str) -> dict:
    """Reconcile manifest <-> Parquet <-> Postgres.

    Three counts that must agree. When they do not, the difference is itemised
    per source rather than reported as one mismatch number, because "we are 47
    rows short somewhere" is not a debuggable statement.
    """
    import json

    settings = get_api_settings()
    project_id = resolve_project(session, project)

    parquet_by_source: dict[str, int] = {}
    distinct_by_source: dict[str, int] = {}
    for source in SOURCES:
        files = discover_partitions(settings.normalized_dir, sources=[source])
        if files:
            parquet_by_source[source] = count_parquet_rows(files)
            distinct_by_source[source] = count_distinct_parquet_ids(files)

    postgres_by_source = {
        source: count
        for source, count in session.execute(
            text("SELECT source, count(*) FROM posts WHERE project_id = :p GROUP BY source"),
            {"p": project_id},
        ).all()
    }

    rejected_by_source = {
        source: count or 0
        for source, count in session.execute(
            text(
                "SELECT source, sum(records_rejected) FROM ingest_runs "
                "WHERE project_id = :p GROUP BY source"
            ),
            {"p": project_id},
        ).all()
    }

    manifest_rows = None
    if settings.manifest_path.exists():
        try:
            manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
            manifest_rows = sum(
                entry.get("rows") or 0 for entry in manifest.values() if isinstance(entry, dict)
            )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read the manifest: %s", exc)

    discrepancies = []
    by_source: dict[str, dict[str, int]] = {}
    for source in sorted(set(parquet_by_source) | set(postgres_by_source)):
        parquet = parquet_by_source.get(source, 0)
        distinct = distinct_by_source.get(source, parquet)
        postgres = postgres_by_source.get(source, 0)
        rejected = rejected_by_source.get(source, 0)
        # The reconciliation identity. Duplicates are separated from losses on
        # purpose: an RSS article re-fetched on two days is one row in Postgres
        # and two in Parquet, and reporting that as a discrepancy forever would
        # train everybody to ignore this report.
        unaccounted = distinct - postgres - rejected
        by_source[source] = {
            "parquet": parquet,
            "distinct_ids": distinct,
            "deduplicated": parquet - distinct,
            "postgres": postgres,
            "rejected": rejected,
            "unaccounted": unaccounted,
        }
        if unaccounted != 0:
            discrepancies.append(
                {
                    "source": source,
                    "parquet_rows": parquet,
                    "distinct_ids": distinct,
                    "deduplicated": parquet - distinct,
                    "postgres_rows": postgres,
                    "rejected_rows": rejected,
                    "unaccounted": unaccounted,
                    "note": (
                        "distinct_ids - postgres - rejected must be zero. A positive "
                        "value is data loss: rows that were read, not rejected, and "
                        "are not in the table. A negative value means Postgres holds "
                        "rows the current Parquet no longer does."
                    ),
                }
            )

    return {
        "project_id": str(project_id),
        "manifest_rows": manifest_rows,
        "parquet_rows": sum(parquet_by_source.values()),
        "parquet_distinct_ids": sum(distinct_by_source.values()),
        "deduplicated_rows": sum(parquet_by_source.values()) - sum(distinct_by_source.values()),
        "postgres_rows": sum(postgres_by_source.values()),
        "rejected_rows": sum(rejected_by_source.values()),
        "reconciled": not discrepancies,
        "discrepancies": discrepancies,
        "by_source": by_source,
        "checked_at": utcnow(),
    }


def truncate_project(session: Session, *, project: str) -> dict[str, int]:
    """Delete a project's corpus, keeping the project row itself.

    Ordered deletes rather than TRUNCATE CASCADE: TRUNCATE cannot be scoped to
    one project, and this schema holds several projects in one database.
    """
    project_id = resolve_project(session, project)
    deleted: dict[str, int] = {}
    # Children before parents. The FKs cascade, but doing it explicitly gives a
    # per-table count, which is what makes the operation reviewable.
    for table in (
        "narrative_posts",
        "narrative_scorecards",
        "compass_citations",
        "compass_feedback",
        "compass_contexts",
        "network_edges",
        "network_layouts",
        "post_scores",
        f"post_embeddings_{get_api_settings().embedding_dim}",
        "author_cohorts",
        "author_group_members",
    ):
        deleted[table] = session.execute(
            text(
                f"DELETE FROM {table} WHERE {_scope_clause(table)}"  # noqa: S608 - fixed list
            ),
            {"p": project_id},
        ).rowcount
    for table in ("narratives", "clustering_runs", "authors", "domains", "posts", "ingest_runs"):
        deleted[table] = session.execute(
            text(f"DELETE FROM {table} WHERE project_id = :p"),  # noqa: S608 - fixed list
            {"p": project_id},
        ).rowcount
    return deleted


def _scope_clause(table: str) -> str:
    """How each child table reaches its project. Explicit, not inferred."""
    return {
        "narrative_posts": ("narrative_id IN (SELECT id FROM narratives WHERE project_id = :p)"),
        "narrative_scorecards": (
            "narrative_id IN (SELECT id FROM narratives WHERE project_id = :p)"
        ),
        "compass_contexts": ("narrative_id IN (SELECT id FROM narratives WHERE project_id = :p)"),
        "compass_citations": (
            "context_id IN (SELECT c.id FROM compass_contexts c "
            "JOIN narratives n ON n.id = c.narrative_id WHERE n.project_id = :p)"
        ),
        "compass_feedback": (
            "context_id IN (SELECT c.id FROM compass_contexts c "
            "JOIN narratives n ON n.id = c.narrative_id WHERE n.project_id = :p)"
        ),
        "network_edges": "project_id = :p",
        "network_layouts": "project_id = :p",
        "post_scores": "post_id IN (SELECT id FROM posts WHERE project_id = :p)",
        "author_cohorts": ("author_id IN (SELECT author_id FROM authors WHERE project_id = :p)"),
        "author_group_members": (
            "author_id IN (SELECT author_id FROM authors WHERE project_id = :p)"
        ),
    }.get(table, "post_id IN (SELECT id FROM posts WHERE project_id = :p)")
