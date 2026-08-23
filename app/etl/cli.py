"""``python -m app.etl.cli`` -- load, reload, verify, truncate, seed.

Separate from ``app.cli`` because the surfaces are genuinely different: that one
is two operational commands, this one is the data pipeline's front door and gets
used interactively while debugging a load.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

import typer
from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)
console = Console()
app = typer.Typer(add_completion=False, help="Parquet -> Postgres corpus loader.")


def _setup(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


@app.command()
def load(
    project: str = typer.Option(..., "--project", "-p", help="Project slug or uuid."),
    sources: str = typer.Option("", help="Comma-separated. Empty means every source."),
    since: str = typer.Option("", help="YYYY-MM-DD, inclusive."),
    until: str = typer.Option("", help="YYYY-MM-DD, inclusive."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Load the Phase 1 corpus into Postgres. Safe to rerun."""
    _setup(verbose)
    from app.db import sync_session
    from app.etl.service import load_corpus

    source_list = [s.strip() for s in sources.split(",") if s.strip()] or None
    with sync_session() as session:
        report = load_corpus(
            session,
            project=project,
            sources=source_list,
            since=_parse_date(since),
            until=_parse_date(until),
        )

    table = Table(title="corpus load", header_style="bold")
    for column in ("source", "read", "inserted", "already present", "rejected", "reasons"):
        table.add_column(column)
    for source, result in sorted(report.per_source.items()):
        table.add_row(
            source,
            str(result.records_in),
            str(result.records_loaded),
            str(result.records_unchanged),
            str(result.records_rejected),
            ", ".join(f"{k}={v}" for k, v in result.rejection_reasons.items()) or "-",
        )
    console.print(table)
    console.print(
        f"authors touched: {report.authors_touched}  "
        f"profiles merged: {report.profiles_merged}  "
        f"domains touched: {report.domains_touched}"
    )
    if report.records_rejected:
        console.print(
            f"[yellow]{report.records_rejected} row(s) rejected. "
            "Every one is written to data/rejects/ with its reason code.[/yellow]"
        )


@app.command()
def reload(
    project: str = typer.Option(..., "--project", "-p"),
    force: bool = typer.Option(False, "--force", help="Required. Deletes before loading."),
    sources: str = typer.Option(""),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Delete a project's corpus and load it again.

    ``--force`` is mandatory rather than a convenience: this drops scores,
    narratives and the graph along with the posts, and a reload typed by
    reflex when a plain ``load`` was meant would be expensive to undo.
    """
    _setup(verbose)
    if not force:
        console.print(
            "[red]reload deletes this project's posts, scores, narratives, embeddings "
            "and graph before loading. Pass --force if that is what you want; "
            "`load` alone is idempotent and almost always the command you meant.[/red]"
        )
        raise typer.Exit(2)

    from app.db import sync_session
    from app.etl.service import load_corpus, truncate_project

    with sync_session() as session:
        deleted = truncate_project(session, project=project)
        console.print("deleted: " + ", ".join(f"{k}={v}" for k, v in deleted.items() if v))
        source_list = [s.strip() for s in sources.split(",") if s.strip()] or None
        report = load_corpus(session, project=project, sources=source_list)
    console.print(f"reloaded {report.records_loaded} rows")


@app.command()
def verify(
    project: str = typer.Option(..., "--project", "-p"),
    strict: bool = typer.Option(True, help="Exit non-zero when the counts do not reconcile."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Reconcile manifest <-> Parquet <-> Postgres row counts."""
    _setup(verbose)
    from app.db import sync_session
    from app.etl.service import verify as run_verify

    with sync_session() as session:
        report = run_verify(session, project=project)

    table = Table(title="reconciliation", header_style="bold")
    columns = (
        "source",
        "parquet",
        "distinct ids",
        "deduped",
        "postgres",
        "rejected",
        "unaccounted",
    )
    for column in columns:
        table.add_column(column, justify="right")
    for source, counts in sorted(report["by_source"].items()):
        style = "red" if counts["unaccounted"] else None
        table.add_row(
            source,
            str(counts["parquet"]),
            str(counts["distinct_ids"]),
            str(counts["deduplicated"]),
            str(counts["postgres"]),
            str(counts["rejected"]),
            str(counts["unaccounted"]),
            style=style,
        )
    console.print(table)
    console.print(
        f"manifest rows: {report['manifest_rows']}  "
        f"parquet: {report['parquet_rows']} "
        f"({report['deduplicated_rows']} duplicate id(s) collapsed)  "
        f"postgres: {report['postgres_rows']}  rejected: {report['rejected_rows']}"
    )

    if report["reconciled"]:
        console.print("[green]reconciled: every parquet row is loaded or accounted for[/green]")
        return

    console.print("[red]NOT reconciled[/red]")
    for item in report["discrepancies"]:
        console.print(f"  {json.dumps(item, indent=2, default=str)}")
    if strict:
        raise typer.Exit(1)


@app.command()
def truncate(
    project: str = typer.Option(..., "--project", "-p"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Delete a project's corpus and everything derived from it."""
    _setup()
    if not yes:
        typer.confirm(
            f"This deletes every post, score, narrative and edge for {project!r}. Continue?",
            abort=True,
        )
    from app.db import sync_session
    from app.etl.service import truncate_project

    with sync_session() as session:
        deleted = truncate_project(session, project=project)
    for table_name, count in deleted.items():
        if count:
            console.print(f"{table_name}: {count}")


@app.command("create-project")
def create_project(
    slug: str = typer.Option(..., "--slug"),
    name: str = typer.Option("", "--name"),
    description: str = typer.Option("", "--description"),
) -> None:
    """Create a project row so the loader has somewhere to put the corpus."""
    _setup()
    from sqlalchemy import select

    from app.db import sync_session
    from app.models.core import Project

    with sync_session() as session:
        existing = session.execute(select(Project).where(Project.slug == slug)).scalar_one_or_none()
        if existing:
            console.print(f"[yellow]project {slug!r} already exists ({existing.id})[/yellow]")
            return
        row = Project(slug=slug, name=name or slug, description=description or None)
        session.add(row)
        session.flush()
        console.print(f"[green]created project {slug} ({row.id})[/green]")


@app.command()
def seed(
    slug: str = typer.Option("demo", "--slug", help="Project slug for the demo corpus."),
) -> None:
    """Build the curated demo project.

    Loads whatever Phase 1 corpus is on disk and copies Phase 2/3's committed
    scored Parquet on top, so a live demo never depends on a network call or on
    a model warming up successfully on stage.
    """
    _setup()
    from app.db import sync_session
    from app.etl.seed import build_demo_project

    with sync_session() as session:
        summary = build_demo_project(session, slug=slug)
    for key, value in summary.items():
        console.print(f"{key}: {value}")


def main() -> None:  # pragma: no cover - entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
