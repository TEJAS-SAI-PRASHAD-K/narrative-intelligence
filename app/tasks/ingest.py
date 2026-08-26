"""Ingestion tasks.

Both are wrappers. ``ingest.fetch`` shells out to Phase 1's own CLI and
``ingest.load`` calls the ETL. Neither knows anything about a platform's API,
and if either ever grows an HTTP client something has gone wrong: Phase 1 owns
ingestion, permanently.
"""

from __future__ import annotations

import logging
from typing import Any

from app.db import advisory_lock, sync_session
from app.tasks.base import TrackedTask, report_progress
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="ingest.fetch", base=TrackedTask, bind=True)
def fetch(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Run Phase 1's adapters to Parquet, then hand off to the loader.

    Invoked in-process rather than as a subprocess so a crash surfaces as a
    Python traceback on the job row instead of an exit code, and so Phase 1's
    logging lands in the same worker log as everything else.
    """
    sources = params.get("sources") or []
    force = bool(params.get("force"))
    mode = params.get("mode", "both")

    from ingest.config import setup_logging

    setup_logging()

    fetched: dict[str, Any] = {}
    errors: dict[str, str] = {}
    targets = sources or ["reddit_convokit", "mastodon", "news_rss", "gdelt", "youtube"]

    for index, source in enumerate(targets):
        report_progress(job_id, 0.05 + 0.7 * index / max(len(targets), 1))
        try:
            fetched[source] = _run_adapter(source, force=force, params=params)
        except Exception as exc:
            # One source failing must not abandon the others. A YouTube quota
            # ceiling is a normal Tuesday and the Reddit pull should still land.
            log.exception("source %s failed", source)
            errors[source] = f"{type(exc).__name__}: {exc}"[:500]

    result: dict[str, Any] = {"fetched": fetched, "errors": errors, "mode": mode}
    if mode == "both" and project_id:
        report_progress(job_id, 0.8)
        result["load"] = load(job_id=None, project_id=project_id, **params)
    report_progress(job_id, 1.0, result=result)
    return result


def _run_adapter(source: str, *, force: bool, params: dict) -> dict[str, Any]:
    """Call one Phase 1 adapter through its own entrypoint."""
    from ingest.sources import get_source

    adapter = get_source(source)
    count = adapter.run(force=force) if hasattr(adapter, "run") else 0
    return {"records": count}


@celery.task(name="ingest.load", base=TrackedTask, bind=True)
def load(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Parquet -> Postgres.

    Serialised per project by an advisory lock. Two concurrent loads of the same
    partitions are individually idempotent but would both stage the whole corpus
    and fight over the same ON CONFLICT, which is wasted work, not corruption --
    still worth avoiding.
    """

    from app.etl.service import load_corpus

    sources = params.get("sources") or None
    since = _as_date(params.get("date_start"))
    until = _as_date(params.get("date_end"))

    with sync_session() as session:
        with advisory_lock(session, f"ingest.load:{project_id}") as acquired:
            if not acquired:
                log.info("another load is already running for %s; skipping", project_id)
                return {"skipped": True, "reason": "another load holds the lock"}

            report_progress(job_id, 0.1)
            report = load_corpus(
                session,
                project=project_id,
                sources=sources,
                since=since,
                until=until,
            )
            report_progress(job_id, 0.9)

    result = {
        "records_in": report.records_in,
        "records_loaded": report.records_loaded,
        "records_unchanged": report.records_unchanged,
        "records_rejected": report.records_rejected,
        "authors_touched": report.authors_touched,
        "domains_touched": report.domains_touched,
        "per_source": {
            source: {
                "read": r.records_in,
                "loaded": r.records_loaded,
                "unchanged": r.records_unchanged,
                "rejected": r.records_rejected,
                "reasons": r.rejection_reasons,
            }
            for source, r in report.per_source.items()
        },
    }

    # Embedding is the next stage and depends only on the posts that just
    # landed. Chained here rather than by the caller so a scheduled ingest
    # produces embeddable posts without anybody remembering to queue it.
    if report.records_loaded:
        celery.send_task("nlp.embed", kwargs={"project_id": project_id}, queue="gpu")

    report_progress(job_id, 1.0, result=result)
    return result


def _as_date(value):
    from datetime import date, datetime

    if not value:
        return None
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()
