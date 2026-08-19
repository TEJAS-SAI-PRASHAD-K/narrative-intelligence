"""Phase 2 command line: train, evaluate, score, cluster, ablate, report.

Every command that produces a number prints the split strategy alongside it. A
bare metric is not a result, and the CLI is where that discipline is either
enforced or quietly abandoned.

Two flags recur:

``--demo``
    Run on the committed fixtures instead of real data. Everything works with no
    network, no credentials and no downloaded benchmarks. Nothing produced this
    way is a result, and every artifact it writes says so.

``--dry-run``
    Compute but do not write. Useful for checking coverage before committing a
    scoring pass to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from modeling.config import (
    get_settings,
    model_version,
    run_fingerprint,
    scoring_config,
    set_all_seeds,
    setup_logging,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Narrative Intelligence Phase 2: modeling and scoring.",
)
console = Console()
log = logging.getLogger("modeling.cli")


def _bootstrap(verbose: bool = False) -> None:
    setup_logging("DEBUG" if verbose else None)
    settings = get_settings()
    settings.ensure_dirs()
    seed = set_all_seeds()
    log.debug("seed=%d device=%s", seed, settings.resolve_device())


# ---------------------------------------------------------------------------
# inspection
# ---------------------------------------------------------------------------
@app.command()
def datasets(
    demo: bool = typer.Option(False, "--demo", help="Check the fixtures instead."),
) -> None:
    """What benchmark data is on disk, and how to get what is not."""
    _bootstrap()
    from modeling.datasets import availability_table, unsatisfied_datasets

    table = Table(title="benchmark availability", show_lines=False)
    for column in ("dataset", "access", "on disk", "fixture", "note"):
        table.add_column(column)
    for row in availability_table(demo):
        table.add_row(
            row["key"],
            row["access"],
            "[green]yes[/]" if row["available"] else "[red]no[/]",
            "[green]yes[/]" if row["fixture"] else "[red]no[/]",
            row["note"] or "",
        )
    console.print(table)

    missing = unsatisfied_datasets(demo)
    if missing:
        console.print(
            f"\n[yellow]{len(missing)} dataset(s) not on disk: {', '.join(missing)}.[/] "
            "Every benchmark this project uses is access-gated; run a command with "
            "[bold]--demo[/] to exercise the pipeline on fixtures, or follow the steps "
            "printed by the loader."
        )
    else:
        console.print("\n[green]every benchmark slot is filled.[/]")


@app.command()
def registry() -> None:
    """Checkpoints in the local cache. Weights never enter git."""
    _bootstrap()
    from modeling.registry import list_local

    entries = list_local()
    if not entries:
        console.print("[yellow]no local checkpoints.[/] Train one, or set HF_REPO / GDRIVE_DIR.")
        return
    table = Table(title="local checkpoint cache")
    for column in ("module", "version", "sha256", "trained at", "path"):
        table.add_column(column)
    for entry in entries:
        table.add_row(
            entry["module"],
            entry["version"],
            entry["sha256"] or "-",
            str(entry.get("trained_at") or "-"),
            entry["path"],
        )
    console.print(table)


@app.command(name="show-config")
def show_config() -> None:
    """Seeds, device, paths and library versions -- the reproducibility block."""
    _bootstrap()
    console.print_json(json.dumps(run_fingerprint(), indent=2, default=str))


@app.command(name="warm-cache")
def warm_cache_command() -> None:
    """Pre-download the auxiliary models so scoring runs work offline."""
    _bootstrap()
    from modeling.aux import aux_model_names, warm_cache

    results = warm_cache(aux_model_names())
    for name, ok in results.items():
        console.print(f"{'[green]cached[/]' if ok else '[red]failed[/]'} {name}")
    if not all(results.values()):
        raise typer.Exit(code=1)


@app.command()
def stats() -> None:
    """Row counts per scored table, and what the manifest claims."""
    _bootstrap()
    from modeling.io import ScoredStore

    store = ScoredStore()
    summary = store.summary()
    if not summary:
        console.print("[yellow]nothing scored yet.[/] Run `modeling score --all`.")
        return
    manifest = store.manifest()
    table = Table(title="scored tables")
    for column in ("table", "rows", "files", "model versions", "generated at"):
        table.add_column(column)
    for row in summary:
        entry = manifest.get(row["table"], {})
        versions = entry.get("model_versions", {})
        table.add_row(
            row["table"],
            str(row["rows"]),
            str(row["files"]),
            ", ".join(f"{k}={v}" for k, v in sorted(versions.items())) or "-",
            str(entry.get("generated_at", "-")),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@app.command()
def score(
    stage: str | None = typer.Argument(
        None, help="Single stage to run (aux, cluster, ...). Omit with --all."
    ),
    all_stages: bool = typer.Option(False, "--all", help="Run every configured stage."),
    demo: bool = typer.Option(False, "--demo", help="Score the committed fixture corpus."),
    sources: str | None = typer.Option(None, help="Comma-separated source filter."),
    start: str | None = typer.Option(None, help="YYYY-MM-DD lower bound."),
    end: str | None = typer.Option(None, help="YYYY-MM-DD upper bound."),
    limit: int | None = typer.Option(None, help="Cap the number of records."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute but do not write."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Produce the scored Parquet tables. Resumable and idempotent."""
    _bootstrap(verbose)
    if not all_stages and stage is None:
        raise typer.BadParameter("give a stage name or --all")

    from modeling.scoring import run_stages

    configured = list(scoring_config().get("stages") or ["aux"])
    stages = configured if all_stages else [stage]
    unknown = [s for s in stages if s not in configured]
    if unknown:
        console.print(
            f"[yellow]stage(s) {unknown} are not in configs/scoring.yaml; running anyway[/]"
        )

    summary = run_stages(
        stages,
        demo=demo,
        sources=[s.strip() for s in sources.split(",")] if sources else None,
        start=start,
        end=end,
        limit=limit,
        dry_run=dry_run,
    )

    table = Table(title="scoring run" + (" [DEMO -- not a result]" if demo else ""))
    for column in ("stage", "table", "written", "updated", "unchanged", "note"):
        table.add_column(column)
    for row in summary:
        table.add_row(
            row["stage"],
            row.get("table", "-"),
            str(row.get("written", 0)),
            str(row.get("updated", 0)),
            str(row.get("unchanged", 0)),
            row.get("note", ""),
        )
    console.print(table)
    if dry_run:
        console.print("[yellow]--dry-run: nothing was written.[/]")


# ---------------------------------------------------------------------------
# training and evaluation
# ---------------------------------------------------------------------------
@app.command()
def train(
    module: str = typer.Argument(..., help="misinfo | stance | bot | deepfake"),
    data: Path | None = typer.Option(None, help="Local benchmark path."),
    demo: bool = typer.Option(False, "--demo", help="Train on the committed fixture."),
    epochs: int | None = typer.Option(None, help="Override the configured epochs."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Train one module, then write its eval report and model card."""
    _bootstrap(verbose)
    from modeling.training import train_module

    # A training failure must never be silent.
    #
    # An estimator aborting inside native code (XGBoost on a single-class fold,
    # for instance) took the interpreter down with SIGSEGV and no message at
    # all -- exit 139, empty output, and nothing written. Silence is
    # indistinguishable from success to anyone reading a terminal, so every
    # failure is caught, named, and given a non-zero exit.
    try:
        result = train_module(module, data_path=data, demo=demo, epochs=epochs)
    except Exception as exc:
        log.exception("training %s failed", module)
        console.print(f"\n[red]training {module} failed:[/] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"\n[bold]{module}[/] {result.headline()}")
    if result.artifacts:
        console.print("artifacts:")
        for path in result.artifacts:
            console.print(f"  {path}")
    if result.skipped:
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    module: str = typer.Argument(..., help="Module to evaluate."),
    data: Path | None = typer.Option(None, help="Local benchmark path."),
    demo: bool = typer.Option(False, "--demo"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Evaluate a trained module without retraining it."""
    _bootstrap(verbose)
    from modeling.training import evaluate_module

    result = evaluate_module(module, data_path=data, demo=demo)
    console.print(f"\n[bold]{module}[/] {result.headline()}")


@app.command()
def report(
    module: str | None = typer.Argument(None, help="Limit to one module."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Regenerate artifacts/eval/** from saved predictions, without retraining."""
    _bootstrap(verbose)
    from modeling.eval.report import regenerate_reports

    written = regenerate_reports(module)
    if not written:
        console.print(
            "[yellow]no saved predictions found.[/] Run `modeling train <module>` or "
            "`modeling evaluate <module>` first."
        )
        return
    for path in written:
        console.print(f"wrote {path}")


@app.command()
def cluster(
    demo: bool = typer.Option(False, "--demo"),
    sources: str | None = typer.Option(None, help="Comma-separated source filter."),
    limit: int | None = typer.Option(None),
    audit: int = typer.Option(0, help="Print N clusters for manual coherence audit."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Embed and cluster the corpus into narratives."""
    _bootstrap(verbose)
    from modeling.scoring import run_stages

    summary = run_stages(
        ["cluster"],
        demo=demo,
        sources=[s.strip() for s in sources.split(",")] if sources else None,
        limit=limit,
    )
    for row in summary:
        console.print(
            f"{row['stage']}/{row.get('table', '-')}: +{row.get('written', 0)} "
            f"~{row.get('updated', 0)} ={row.get('unchanged', 0)} {row.get('note', '')}"
        )
    if audit:
        from modeling.text.cluster import audit_table

        console.print(audit_table(audit))


@app.command()
def ablate(
    demo: bool = typer.Option(False, "--demo"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Run the module-ablation table.

    The fusion used here is provisional and for measurement only -- the product
    formula is a Phase 4 decision, deliberately.
    """
    _bootstrap(verbose)
    from modeling.eval.ablation import run_ablation

    result = run_ablation(demo=demo)
    console.print(result.render())


@app.command(name="sample-for-labelling")
def sample_for_labelling(
    what: str = typer.Argument(..., help="misinfo | narratives"),
    n: int = typer.Option(100, help="How many rows to sample."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Write a blank-label CSV for hand-labelling.

    This is how the two most informative tables in the whole report get
    produced, and neither can be automated:

    * ``misinfo`` — 100 real corpus records. Scoring them measures the
      benchmark-to-corpus transfer gap, which is the only number that describes
      production behaviour.
    * ``narratives`` — narrative-level risk labels, which the ablation table
      needs before its quality columns mean anything.
    """
    _bootstrap(verbose)

    from modeling.io import CorpusReader, ScoredStore

    settings = get_settings()
    target_dir = settings.artifacts_dir / "hand_labels"
    target_dir.mkdir(parents=True, exist_ok=True)

    if what == "misinfo":
        records = CorpusReader(settings).records(columns=["id", "source", "text", "lang"])
        if not len(records):
            console.print("[red]no corpus on disk[/]")
            raise typer.Exit(code=1)
        # Stratified by source so the sample is not all Reddit: the transfer gap
        # is a per-platform question, and a single-source sample cannot see it.
        per_source = max(1, n // records["source"].nunique())
        sample = (
            records.groupby("source", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_source), random_state=settings.seed))
            .head(n)
        )
        scores = ScoredStore(settings).read("record_scores")
        if len(scores) and "misinfo_prob" in scores.columns:
            sample = sample.merge(
                scores[["record_id", "misinfo_prob"]].rename(
                    columns={"misinfo_prob": "score"}
                ),
                left_on="id",
                right_on="record_id",
                how="left",
            )
        sample["label"] = ""  # 1 = misinformation-like, 0 = not
        path = target_dir / "misinfo_corpus_sample.csv"
        columns = [c for c in ("id", "source", "lang", "text", "score", "label") if c in sample]
        sample[columns].to_csv(path, index=False)
    elif what == "narratives":
        narratives = ScoredStore(settings).read("narratives")
        if not len(narratives):
            console.print("[red]no narratives on disk; run `modeling score cluster` first[/]")
            raise typer.Exit(code=1)
        sample = narratives.nlargest(min(n, len(narratives)), "size")[
            ["narrative_id", "label", "size", "author_count", "coherence"]
        ].rename(columns={"label": "narrative_label"})
        sample["risk"] = ""  # 0-1, your judgement of how concerning this narrative is
        path = target_dir / "narratives.csv"
        sample.to_csv(path, index=False)
    else:
        raise typer.BadParameter("what must be 'misinfo' or 'narratives'")

    console.print(f"wrote {path}")
    console.print(
        "[yellow]Fill in the blank column by hand[/], then rerun "
        f"`modeling {'evaluate misinfo' if what == 'misinfo' else 'ablate'}`. "
        "Until it is filled in, the corresponding table reports the gap as unmeasured."
    )


@app.command(name="model-version")
def model_version_command(module: str = typer.Argument(...)) -> None:
    """The version string this module stamps into every scored row."""
    _bootstrap()
    console.print(model_version(module))


if __name__ == "__main__":
    app()
