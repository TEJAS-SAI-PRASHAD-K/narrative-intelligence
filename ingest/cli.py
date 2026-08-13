"""Command line for the ingestion layer.

    python -m ingest.cli fetch-all          # rebuild the corpus from scratch
    python -m ingest.cli fetch mastodon     # one source
    python -m ingest.cli stats              # per-source summary table
    python -m ingest.cli validate           # re-validate the corpus on disk

``fetch-all`` never fails because a credential is missing. Sources that cannot
run are skipped with a warning and reported as ``skipped`` in the summary --
that is the graceful-degradation acceptance criterion, and it is also what makes
the pipeline reproducible on a machine that has different keys to yours.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from ingest.config import get_settings, setup_logging, sources_config, topics_config
from ingest.schema import Record
from ingest.sources import REGISTRY, get_source_class
from ingest.store import ParquetStore

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Narrative Intelligence Platform - Phase 1 ingestion layer.",
)
console = Console()


@app.callback()
def main(log_level: str = typer.Option("INFO", "--log-level", help="DEBUG/INFO/WARNING")) -> None:
    setup_logging(log_level)


# --- fetching -------------------------------------------------------------


@app.command()
def fetch(
    source: str = typer.Argument(..., help=f"one of: {', '.join(REGISTRY)}"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="stop after N records"),
    force: bool = typer.Option(False, "--force", help="ignore checkpoints and refetch"),
    path: str | None = typer.Option(None, "--path", help="local corpus/dump directory"),
) -> None:
    """Fetch one source into the normalized corpus."""
    settings = get_settings()
    store = ParquetStore(settings)
    options: dict[str, Any] = {"force": force}
    if path:
        options["path"] = path

    source_class = get_source_class(source)
    result = source_class(settings=settings, store=store, limit=limit, **options).run()
    _print_results([result])
    if result.error:
        raise typer.Exit(code=1)


@app.command("fetch-all")
def fetch_all(
    limit: int | None = typer.Option(None, "--limit", "-n", help="per-source record cap"),
    force: bool = typer.Option(False, "--force", help="ignore checkpoints and refetch"),
    only: str | None = typer.Option(None, "--only", help="comma-separated subset of sources"),
) -> None:
    """Rebuild the whole corpus. Missing credentials skip a source, never fail the run."""
    settings = get_settings()
    settings.ensure_dirs()
    store = ParquetStore(settings)
    names = [n.strip() for n in only.split(",")] if only else list(REGISTRY)

    results = []
    for name in names:
        console.rule(f"[bold]{name}")
        try:
            source_class = get_source_class(name)
        except KeyError as exc:
            console.print(f"[red]{exc}")
            continue
        results.append(source_class(settings=settings, store=store, limit=limit, force=force).run())

    _print_results(results)
    console.print(f"\nmanifest: {settings.manifest_path}")
    console.print(f"corpus:   {settings.normalized_dir}")

    # A source that errored is worth a non-zero exit; a *skipped* one is not.
    if any(r.error for r in results):
        raise typer.Exit(code=1)


@app.command("mastodon-stream")
def mastodon_stream(
    minutes: float = typer.Option(2.0, "--minutes", "-m", help="bounded run length"),
) -> None:
    """Tail the public Mastodon stream for a bounded time and write what arrives."""
    from ingest.sources.mastodon import MastodonSource

    settings = get_settings()
    source = MastodonSource(settings=settings, store=ParquetStore(settings))
    try:
        source.preflight()
    except Exception as exc:
        console.print(f"[yellow]skipping stream: {exc}")
        raise typer.Exit(code=0) from None

    records: list[Record] = []
    for raw in source.stream(minutes=minutes):
        record = source.to_record(raw)
        if record is not None:
            records.append(record)
    written = source.store.write_records(records)
    console.print(
        f"streamed {len(records)} records, wrote {written['written']} "
        f"({written['duplicates']} duplicates)"
    )


@app.command("mastodon-register")
def mastodon_register(
    instance: str = typer.Option("https://mastodon.social", "--instance"),
) -> None:
    """Register this app against an instance and print what to paste into .env."""
    from ingest.sources.mastodon import register_app

    credentials = register_app(instance)
    console.print("[bold]App registered.[/bold] Client credentials (keep these out of git):")
    console.print(credentials)
    console.print(
        "\nNow create an access token for the app (read scope is enough) and set:\n"
        f"  MASTODON_API_BASE_URL={instance}\n"
        "  MASTODON_ACCESS_TOKEN=<token>"
    )


# --- inspection -----------------------------------------------------------


@app.command()
def stats(source: str | None = typer.Argument(None, help="limit to one source")) -> None:
    """Per-source summary of the corpus on disk."""
    store = ParquetStore(get_settings())
    rows = [r for r in store.stats() if source is None or r["source"] == source]
    if not rows:
        console.print("[yellow]no corpus on disk yet. run `python -m ingest.cli fetch-all`.")
        raise typer.Exit(code=0)

    table = Table(title="corpus summary", header_style="bold")
    for column in (
        "source",
        "records",
        "authors",
        "first",
        "last",
        "threaded %",
        "median chars",
        "langs",
        "partitions",
    ):
        table.add_column(column, justify="right" if column != "source" else "left")
    for row in rows:
        table.add_row(
            row["source"],
            f"{row['records']:,}",
            f"{row['authors']:,}",
            (row["first"] or "-")[:10],
            (row["last"] or "-")[:10],
            f"{row['threaded_pct']}",
            f"{row['median_chars']}",
            f"{row['langs']}",
            f"{row['partitions']}",
        )
    total = sum(r["records"] for r in rows)
    console.print(table)
    console.print(f"total records: {total:,}")


@app.command()
def validate(
    source: str | None = typer.Argument(None, help="limit to one source"),
    sample: int | None = typer.Option(None, "--sample", help="validate only N rows per source"),
) -> None:
    """Re-validate the corpus on disk against the schema.

    The acceptance criterion is zero validation errors after a full run, so this
    reads what is actually stored rather than trusting what was written.
    """
    settings = get_settings()
    store = ParquetStore(settings)
    sources = [source] if source else [row["source"] for row in store.stats()]
    if not sources:
        console.print("[yellow]no corpus on disk yet.")
        raise typer.Exit(code=0)

    failures = 0
    for name in sources:
        checked = 0
        errors: list[str] = []
        for row in store.iter_records(name):
            if sample is not None and checked >= sample:
                break
            checked += 1
            payload = dict(row)
            payload.pop("date", None)  # hive partition column, not a schema field
            payload["raw"] = json.loads(payload.get("raw") or "{}")
            try:
                Record(**payload)
            except Exception as exc:
                errors.append(f"{payload.get('id')}: {str(exc).splitlines()[0]}")
        failures += len(errors)
        status = "[green]ok" if not errors else f"[red]{len(errors)} invalid"
        console.print(f"{name}: {checked:,} rows checked -> {status}")
        for line in errors[:5]:
            console.print(f"  [red]{line}")

    if failures:
        raise typer.Exit(code=1)
    console.print("[green]corpus is schema-valid")


@app.command()
def manifest() -> None:
    """Print the artifact manifest (source url, sha256, bytes, rows)."""
    settings = get_settings()
    if not settings.manifest_path.exists():
        console.print("[yellow]no manifest yet; run a fetch first.")
        raise typer.Exit(code=0)
    entries = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    table = Table(title=str(settings.manifest_path), header_style="bold")
    for column in ("artifact", "rows", "bytes", "sha256", "fetched_at"):
        table.add_column(column, overflow="fold")
    for key, entry in sorted(entries.items()):
        table.add_row(
            key,
            f"{entry.get('rows'):,}" if isinstance(entry.get("rows"), int) else "-",
            f"{entry.get('bytes'):,}" if isinstance(entry.get("bytes"), int) else "-",
            (entry.get("sha256") or "-")[:16],
            (entry.get("fetched_at") or "-")[:19],
        )
    console.print(table)


@app.command("show-config")
def show_config() -> None:
    """What the pipeline will do, and which credentials it can see.

    Prints presence, never values -- a secret that reaches a terminal ends up in
    a scrollback buffer, a screenshot, or a bug report.
    """
    settings = get_settings()
    table = Table(title="configuration", header_style="bold")
    table.add_column("setting")
    table.add_column("value", overflow="fold")
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row("user_agent", settings.user_agent)
    table.add_row("http rate limit", f"{settings.http_rate_limit_per_sec}/s")
    for key, present in (
        ("MASTODON_ACCESS_TOKEN", bool(settings.mastodon_access_token)),
        ("YOUTUBE_API_KEY", bool(settings.youtube_api_key)),
        ("NEWSAPI_KEY", bool(settings.newsapi_key)),
        ("KAGGLE credentials", settings.has_credentials("reddit_kaggle")),
    ):
        table.add_row(key, "[green]set" if present else "[yellow]absent (source will skip)")
    console.print(table)

    sources = sources_config()
    topics = topics_config()
    console.print(
        f"\ntopics: {', '.join(t.get('id', '?') for t in topics.get('topics', []))}"
        f"\nconvokit corpora: {', '.join(sources.get('reddit_convokit', {}).get('corpora', []))}"
        f"\nrss feeds: {len(sources.get('news_rss', {}).get('feeds', []))}"
    )


# --- helpers --------------------------------------------------------------


def _print_results(results) -> None:
    table = Table(title="run summary", header_style="bold")
    for column in ("source", "fetched", "written", "dupes", "dropped", "status"):
        table.add_column(column, justify="right" if column != "source" else "left")
    for result in results:
        row = result.as_row()
        colour = {"ok": "green", "skipped": "yellow", "error": "red"}[row["status"]]
        table.add_row(
            row["source"],
            f"{row['fetched']:,}",
            f"{row['written']:,}",
            f"{row['duplicates']:,}",
            f"{row['dropped']:,}",
            f"[{colour}]{row['status']}",
        )
    console.print(table)
    for result in results:
        if result.dropped:
            console.print(f"  {result.source} drops: {dict(result.dropped)}")
        if result.flags:
            console.print(f"  {result.source} flags: {dict(result.flags)}")
        if result.skipped_reason:
            console.print(f"  [yellow]{result.source}: {result.skipped_reason}")
        if result.error:
            console.print(f"  [red]{result.source}: {result.error}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
