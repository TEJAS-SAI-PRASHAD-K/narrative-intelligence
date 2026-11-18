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
from rich.markup import escape
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


@app.command(name="checkpoint-hash")
def checkpoint_hash(
    module: str = typer.Argument(..., help="misinfo | stance | laya"),
    version: str | None = typer.Option(
        None, "--version", help="Checkpoint folder name. Default: the version in models.yaml."
    ),
) -> None:
    """Print the sha256 of a local checkpoint directory, to put in configs/models.yaml.

    Reads the directory under MODELS_DIR. It never pulls from the Hub.
    """
    _bootstrap()
    from modeling.registry import directory_sha256, sibling_folders

    path, digest = directory_sha256(module, version)
    if digest is None:
        found = sibling_folders(module)
        console.print(f"[red]no checkpoint files in {path}[/]")
        console.print(f"folders under models/{module}: {', '.join(found) if found else 'none'}")
        raise typer.Exit(1)
    typer.echo(digest)
    typer.echo(f"{module}/{path.name}  {path}", err=True)


@app.command(name="checkpoint-pull")
def checkpoint_pull(
    module: str = typer.Argument(..., help="misinfo | stance | laya"),
    version: str | None = typer.Option(
        None, "--version", help="Checkpoint folder name. Default: the version in models.yaml."
    ),
) -> None:
    """Resolve a checkpoint, pull it from HF_REPO if absent, and verify its sha256."""
    _bootstrap()
    from modeling.config import module_config
    from modeling.registry import (
        CheckpointIntegrityError,
        CheckpointNotFound,
        resolve,
        sha256_versions,
    )

    known = sha256_versions(module)
    wanted = version or str(module_config(module).get("version", ""))
    if known and wanted not in known:
        console.print(f"[red]no sha256 for {module}/{wanted}. Pass --version with one of:[/]")
        console.print(", ".join(known))
        raise typer.Exit(1)

    try:
        found = resolve(module, version)
    except CheckpointIntegrityError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
    except CheckpointNotFound as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    assert found is not None
    state = "verified" if found.verified else "UNVERIFIED (no sha256 in configs/models.yaml)"
    console.print(f"{found.module}/{found.version}  origin={found.origin}  {state}")
    console.print(f"path: {found.path}")
    console.print(f"sha256: {found.sha256}")


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
    module: str = typer.Argument(..., help="misinfo | stance | bot"),
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
    sample: Path | None = typer.Option(
        None,
        help="misinfo only: the hand-labelled sample CSV. "
        "Default artifacts/hand_labels/misinfo_corpus_sample_en.csv.",
    ),
    allow_model_labels: bool = typer.Option(
        False,
        "--allow-model-labels",
        help="misinfo only: use labels that a model made and no person reviewed. "
        "Every output then says so.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Evaluate a trained module without retraining it.

    For ``misinfo`` the output ends with a corpus-transfer section computed from the
    hand-labelled sample. It says so when no sample exists, and refuses to compute
    metrics while fewer than MIN_LABELLED_ROWS rows carry a label. The label is the
    ``adjudicated_label`` column. A sheet with labels from a model that no person
    reviewed is refused unless ``--allow-model-labels`` is set. Exits 1 on refusal.
    """
    _bootstrap(verbose)
    from modeling.training import evaluate_module

    result = evaluate_module(module, data_path=data, demo=demo)
    console.print(f"\n[bold]{module}[/] {result.headline()}")
    if module != "misinfo":
        return

    from modeling.config import module_config
    from modeling.eval.corpus_transfer import evaluate_transfer
    from modeling.io import ScoredStore

    settings = get_settings()
    path = sample or settings.artifacts_dir / "hand_labels" / "misinfo_corpus_sample_en.csv"
    if not path.exists():
        console.print(
            f"\n[yellow]corpus transfer: NOT MEASURED[/] - no labelled sample at {path}. "
            "Run `modeling sample-for-labelling misinfo --lang en`, label the sheet by hand "
            "(1 = misinformation-like, 0 = not) as docs/labelling-guide.md says, then rerun "
            "this command."
        )
        return
    from modeling.eval.labelling import resolve_sheet_file

    resolved = resolve_sheet_file(path, allow_model_labels=allow_model_labels)
    transfer = evaluate_transfer(
        resolved,
        ScoredStore(settings).read("record_scores"),
        language_allowed=lambda lang: settings.language_allowed("misinfo", lang),
        model_version=str(module_config("misinfo").get("version", "v0.0.0-unset")),
        threshold=module_config("misinfo").get("decision_threshold"),
    )
    benchmark = result.report.macro_f1.value if result.report is not None else None
    console.print()
    for line in transfer.render(benchmark_macro_f1=benchmark):
        console.print(line, markup=False, highlight=False, soft_wrap=True)
    if not transfer.computed:
        raise typer.Exit(code=1)


@app.command(name="evaluate-laya-hindi")
def evaluate_laya_hindi(
    sample: Path | None = typer.Option(
        None, help="The Hindi sheet. Default artifacts/hand_labels/misinfo_corpus_sample_hi.csv."
    ),
    threshold: float | None = typer.Option(
        None,
        help="Decision threshold. Default is laya.decision_threshold, which is null until set.",
    ),
    allow_model_labels: bool = typer.Option(False, "--allow-model-labels"),
    n_bootstrap: int = typer.Option(1000, "--n-bootstrap", help="Resamples for each interval."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Laya on Hindi titles: transfer metrics and calibration error against adjudicated labels.

    The values are the ``misinfo_prob`` that the ``laya`` scoring stage wrote to
    ``record_scores``. This command runs no model.
    """
    _bootstrap(verbose)
    from modeling.eval.relabel import laya_hindi_section

    section = laya_hindi_section(
        sample or get_settings().artifacts_dir / "hand_labels" / "misinfo_corpus_sample_hi.csv",
        allow_model_labels=allow_model_labels,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
    )
    for line in section.lines:
        console.print(line, markup=False, highlight=False, soft_wrap=True)
    if not section.ok:
        raise typer.Exit(code=1)


@app.command(name="label-agreement")
def label_agreement(
    sheet: Path = typer.Argument(..., help="A labelled sheet with label_a and label_b."),
    fill_agreed: bool = typer.Option(
        False,
        "--fill-agreed",
        help="Copy the shared label into the adjudicated columns where both labellers agree. "
        "Rows where they disagree stay blank for a person to settle.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Cohen's kappa with a bootstrap interval, raw agreement and the disagreement list."""
    _bootstrap(verbose)
    from modeling.eval.labelling import (
        DEFAULT_SEED,
        agreement_report,
        read_sheet,
        render_agreement,
    )
    from modeling.eval.labelling import (
        fill_agreed as fill,
    )

    frame = read_sheet(sheet)
    for line in render_agreement(agreement_report(frame, seed=DEFAULT_SEED), DEFAULT_SEED):
        console.print(line, markup=False, highlight=False, soft_wrap=True)
    if fill_agreed:
        import shutil

        filled, counts = fill(frame)
        # Keep the sheet as it was, then replace it in one step, so a crash cannot leave a
        # half-written sheet that holds hand labels.
        backup = sheet.with_name(sheet.name + ".bak")
        shutil.copy2(sheet, backup)
        temporary = sheet.with_name(sheet.name + ".tmp")
        filled.to_csv(temporary, index=False)
        temporary.replace(sheet)
        console.print(
            f"filled {counts['filled']} cells, {counts['left_for_adjudication']} left. "
            f"The sheet as it was before is in {backup}.",
            soft_wrap=True,
        )


@app.command(name="relabel-eval")
def relabel_eval(
    en_sheet: Path | None = typer.Option(None, help="Default hand_labels/..._en.csv."),
    hi_sheet: Path | None = typer.Option(None, help="Default hand_labels/..._hi.csv."),
    out_dir: Path | None = typer.Option(None, help="Default artifacts/eval/labelling."),
    allow_model_labels: bool = typer.Option(False, "--allow-model-labels"),
    laya_threshold: float | None = typer.Option(None, "--laya-threshold"),
    n_bootstrap: int = typer.Option(1000, "--n-bootstrap", help="Resamples for each interval."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Read the filled sheets and rewrite every labelling output in one run.

    For each sheet that exists: the agreement report, the disagreement list and the
    transfer metrics (``misinfo`` for English, Laya for Hindi). A sheet that is refused
    writes its refusal. Exits 1 when any sheet is refused or not measured. A sheet whose
    usable rows hold one class is refused for ROC-AUC, precision and recall, and still reports
    the positive rate, the flag rate and the calibration. The seed is 20260813.
    """
    _bootstrap(verbose)
    from modeling.eval.relabel import run_relabel_eval

    settings = get_settings()
    root = settings.artifacts_dir
    outcome = run_relabel_eval(
        en_sheet=en_sheet or root / "hand_labels" / "misinfo_corpus_sample_en.csv",
        hi_sheet=hi_sheet or root / "hand_labels" / "misinfo_corpus_sample_hi.csv",
        out_dir=out_dir or root / "eval" / "labelling",
        allow_model_labels=allow_model_labels,
        laya_threshold=laya_threshold,
        n_bootstrap=n_bootstrap,
    )
    for path in outcome.written:
        console.print(f"wrote {path}", soft_wrap=True, highlight=False)
    for line in outcome.lines:
        console.print(line, markup=False, highlight=False, soft_wrap=True)
    if not outcome.ok:
        raise typer.Exit(code=1)


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


@app.command(name="evaluate-factchecks")
def evaluate_factchecks(
    scores: Path = typer.Argument(..., help="Directory with one <model>.parquet per model."),
    baseline: str = typer.Option("tfidf", help="Model the others are compared against."),
    precision: float = typer.Option(0.85, help="Validation precision target for each threshold."),
    n_bootstrap: int = typer.Option(2000, "--n-bootstrap", help="Resamples for every interval."),
    model_version: list[str] = typer.Option(
        None, "--model-version", help="name=version, repeatable. Recorded in the report."
    ),
    manifest: Path | None = typer.Option(
        None, help="Frozen evaluation set CSV. Default configs/factcheck_eval_manifest.csv."
    ),
    links: Path | None = typer.Option(
        None, help="Claim links artifact. Default data/factchecks/claim_links.json."
    ),
    split: str | None = typer.Option(None, help="How val and test were split, in one line."),
    label: str = typer.Option("comparison", help="Output goes to artifacts/eval/factcheck/LABEL."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Three-way comparison on the Indian fact-check set, from precomputed scores.

    Needs no model weights and no GPU. Exits 1 and writes nothing when an input is
    malformed, the test splits differ, or the frozen evaluation set has changed.
    """
    _bootstrap(verbose)
    from modeling.eval.factcheck_eval import EvalInputError, gate_text, run_comparison

    versions: dict[str, str] = {}
    for item in model_version or []:
        name, sep, version = item.partition("=")
        if not sep or not name or not version:
            raise typer.BadParameter(f"--model-version needs name=version, got {item!r}")
        versions[name] = version
    try:
        result, paths = run_comparison(
            scores,
            manifest_path=manifest,
            links_path=links,
            baseline=baseline,
            precision_target=precision,
            n_bootstrap=n_bootstrap,
            versions=versions,
            split_description=split,
            label=label,
        )
    except EvalInputError as exc:
        console.print(f"[red]refused:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if not result["evaluation_set"]["frozen"]:
        console.print("[bold yellow]evaluation set not frozen[/] (see the report)")
    for name, comparison in result["comparisons"].items():
        console.print(
            f"{name}: {'ship' if comparison['ship'] else 'do not ship'} - {comparison['reason']}"
        )
    for name, gate in result["hindi_gate"].items():
        console.print(f"{name}: Hindi gate {gate_text(gate)} - {gate['reason']}")
    for path in paths:
        console.print(f"wrote {path}")


@app.command(name="dedup-claims")
def dedup_claims(
    data_dir: Path | None = typer.Option(
        None, "--data-dir", help="Fact-check data directory to read. Default: DATA_DIR."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Directory for claim_links.json and the sample. Default: DATA_DIR.",
    ),
    sample_size: int | None = typer.Option(
        None, "--sample-size", help="Pairs in the hand-check sample."
    ),
    no_embedding: bool = typer.Option(
        False, "--no-embedding", help="Run only the simhash pass."
    ),
    overwrite_sample: bool = typer.Option(
        False,
        "--overwrite-sample",
        help="Replace a hand-check sample that already holds labels.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Link claims by simhash and multilingual embedding similarity, read side.

    Writes `claim_links.json` (cluster ids, links, counts per pass) and a
    hand-check sample CSV under the output directory. `factchecks.parquet` is
    read and never modified; a pass that cannot run is reported with a reason
    code and no collapse count.
    """
    _bootstrap(verbose)
    from ingest.config import Settings
    from ingest.factcheck import FACTCHECKS_FILENAME, read_factcheck_rows
    from modeling.eval import claim_dedup as cd
    from modeling.eval.factcheck_eval import EvalInputError, title_predicate_settings

    read_dir = Path(data_dir) if data_dir else get_settings().data_dir
    out_dir = Path(out) if out else get_settings().data_dir
    ingest_settings = Settings(data_dir=read_dir, _env_file=None)
    source = ingest_settings.data_dir / "factchecks" / FACTCHECKS_FILENAME
    rows = read_factcheck_rows(ingest_settings)
    if not rows:
        console.print(f"[yellow]no fact-check rows at {source}.")
        raise typer.Exit(code=0)
    try:
        config = cd.claim_dedup_config()
    except cd.ClaimDedupConfigError as exc:
        console.print(f"[red]refused:[/] {exc}")
        raise typer.Exit(code=1) from exc
    if no_embedding:
        config = cd.config_without_embedding(config)

    try:
        merge_tier4 = title_predicate_settings()["include_in_primary_evaluation"]
    except EvalInputError as exc:
        console.print(f"[red]refused:[/] {exc}")
        raise typer.Exit(code=1) from exc
    sample_path = cd.hand_check_sample_path(out_dir)
    side_path = cd.side_sample_path(sample_path)
    # Validate every existing sample before any write, so a refusal leaves the disk as it was.
    try:
        keep_sample = not overwrite_sample and cd.sample_has_labels(sample_path)
        if keep_sample:
            cd.embedding_precision(sample_path)
            if cd.sample_has_labels(side_path):
                raise cd.ClaimSampleError(
                    f"{side_path} has labels and {sample_path} has labels. Neither is "
                    "replaced. Move one of them, or pass --overwrite-sample to replace both"
                )
    except cd.ClaimSampleError as exc:
        console.print(f"[red]refused:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    result = cd.run_claim_dedup(rows, config=config, merge_tier4=merge_tier4)
    links_path = cd.claim_links_path(out_dir)
    cd.write_claim_links(
        result,
        links_path,
        source=source if source.exists() else None,
        sample_path=sample_path,
    )
    sample_size_used = sample_size or config.sample_size
    if keep_sample:
        # A labelled sample is hand work. The new blank sample goes beside it.
        written_sample = side_path
        cd.write_hand_check_sample(result, rows, written_sample, size=sample_size_used)
        console.print(
            f"[yellow]warning:[/] {escape(str(sample_path))} has labels and is kept "
            f"(reason code {cd.PrecisionReason.SAMPLE_HAS_LABELS.value}). "
            f"The new blank sample is {escape(str(written_sample))}. "
            "Use --overwrite-sample to replace it."
        )
    else:
        written_sample = sample_path
        cd.write_hand_check_sample(result, rows, sample_path, size=sample_size_used)
        # An older side file no longer matches the main sample.
        side_path.unlink(missing_ok=True)

    for name in ("simhash", "embedding"):
        outcome = result.outcome(name)
        if outcome is None:
            continue
        count = "not measured" if outcome.collapsed is None else str(outcome.collapsed)
        reason = f" ({outcome.reason})" if outcome.reason else ""
        console.print(f"{name}: {count} collapsed{reason}")
    embedding_outcome = result.outcome("embedding")
    pairs = (
        cd.embedding_pairs(result.links)
        if embedding_outcome is not None and embedding_outcome.collapsed is not None
        else None
    )
    try:
        threshold, source = (
            (embedding_outcome.threshold, "links")
            if embedding_outcome is not None and embedding_outcome.threshold is not None
            else cd.run_threshold({})
        )
        precision = cd.embedding_precision(
            sample_path,
            pairs=pairs,
            threshold=threshold,
            threshold_source=source,
            link_similarities=(
                cd.embedding_similarities(result.links) if pairs is not None else None
            ),
        )
    except cd.ClaimSampleError as exc:
        console.print(f"[red]refused:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    console.print(cd.describe_precision(precision.as_dict()), markup=False)
    console.print(f"wrote {links_path}", markup=False)
    console.print(f"wrote {written_sample}", markup=False)


@app.command(name="freeze-factchecks")
def freeze_factchecks(
    out: Path | None = typer.Option(
        None, help="Manifest path. Default configs/factcheck_eval_manifest.csv."
    ),
    links: Path | None = typer.Option(
        None, help="Claim links artifact. Default data/factchecks/claim_links.json."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing manifest. Say why in the commit."
    ),
) -> None:
    """Write the frozen evaluation set (record id, URL, claim SHA-256) from factchecks.parquet.

    The claim text is hashed and never written. Commit the CSV. Refuses to replace an
    existing manifest without --overwrite. A claim links artifact, when present,
    decides the set together with the simhash pass.
    """
    _bootstrap()
    from modeling.eval.factcheck_eval import (
        EvalInputError,
        freeze_manifest_from_data,
        publisher_shares_from_data,
    )

    try:
        path, count = freeze_manifest_from_data(
            get_settings().data_dir, out, overwrite=overwrite, links_path=links
        )
    except EvalInputError as exc:
        console.print(f"[red]refused:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"froze {count} known-false claims -> {path}")
    shares = publisher_shares_from_data(get_settings().data_dir)
    console.print(
        f"Hindi claims: {shares['n_hindi_claims']} from {shares['n_publishers']} publisher(s)"
    )
    for row in shares["publishers"]:
        console.print(
            f"  {row['publisher']}: {row['n_hindi_claims']} ({row['share']:.1%} of Hindi claims)"
        )
    if shares["reason"]:
        console.print(
            f"[yellow]{shares['reason']}[/]: Hindi claims rest on fewer than 2 publishers"
        )


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


@app.command(name="coordination-report")
def coordination_report_command(
    sources: str | None = typer.Option(
        None,
        help="Comma-separated sources to measure. May include sources that the production "
        "eligible set leaves out, such as telegram. This is a research measurement.",
    ),
    null_shuffles: int = typer.Option(200, help="Time shuffles per null. Must be 1 or more."),
    out: Path | None = typer.Option(
        None, help="Output directory. Default: artifacts/eval/coordination."
    ),
    demo: bool = typer.Option(False, "--demo"),
    limit: int | None = typer.Option(None),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Observed against time-shuffled modularity, overall and for each source.

    Reads the corpus and writes only report.md and report.json under --out. It
    never writes to data/scored, so a source outside the production eligible set
    cannot reach author scores. A source with too few authors or edges appears
    with a reason code. Telegram posts that carry `forwarded_from` are dropped and
    counted as `telegram_forward`. The p value cannot fall below 1 / (1 + shuffles).
    """
    _bootstrap(verbose)
    from modeling.accounts.coordination import CoordinationDetector, write_report
    from modeling.scoring import _reader_for

    if null_shuffles < 1:
        raise typer.BadParameter("--null-shuffles must be 1 or more")
    settings = get_settings()
    wanted = [s.strip() for s in sources.split(",") if s.strip()] if sources else None
    records = _reader_for(demo, settings).records(sources=wanted, limit=limit)
    if not len(records):
        console.print("[red]no records for that source filter[/red]")
        raise typer.Exit(code=1)
    detector = CoordinationDetector(settings)
    detector.null_shuffles = null_shuffles
    if wanted:
        # Research only: the report reads the sources the operator names.
        detector.eligible_sources = set(wanted)
    if 1 / (1 + null_shuffles) > 0.05:
        console.print(
            f"[yellow]p floor {1 / (1 + null_shuffles):.3f} is above 0.05 with "
            f"{null_shuffles} shuffles; no p value in this run can reach 0.05[/yellow]"
        )
    result = detector.detect(records, per_source=True)
    target = out or settings.eval_dir / ("coordination_demo" if demo else "coordination")
    for path in write_report(result, target):
        console.print(f"wrote {path}", soft_wrap=True, highlight=False)


@app.command(name="compare-embedder")
def compare_embedder_command(
    model: str = typer.Option(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedder to compare with the one in configs/models.yaml.",
    ),
    out: Path | None = typer.Option(
        None, help="Output directory. Default: artifacts/eval/embedder/<model>."
    ),
    demo: bool = typer.Option(False, "--demo"),
    sources: str | None = typer.Option(None, help="Comma-separated source filter."),
    limit: int | None = typer.Option(None),
    min_cluster_size: str | None = typer.Option(
        None,
        "--min-cluster-size",
        help="Comma-separated sweep, for example 5,10,15. Both embedders run at each size.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Cluster with another embedder and print the checks and ablation beside the default.

    Writes only under --out. It never touches data/scored, the shared scored
    manifest or artifacts/eval/ablation, and it does not change the default model.
    """
    _bootstrap(verbose)
    from modeling.eval.embedder_compare import compare_embedder
    from modeling.text.embed import EmbeddingDimensionMismatch

    try:
        sweep = (
            [int(v) for v in min_cluster_size.split(",") if v.strip()]
            if min_cluster_size
            else None
        )
    except ValueError as exc:
        raise typer.BadParameter("--min-cluster-size takes whole numbers, such as 5,10,15") from exc
    try:
        result = compare_embedder(
            model,
            demo=demo,
            out_dir=out,
            sources=[s.strip() for s in sources.split(",")] if sources else None,
            limit=limit,
            min_cluster_sizes=sweep,
        )
    except (EmbeddingDimensionMismatch, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    for path in result.written:
        console.print(f"wrote {path}", soft_wrap=True, highlight=False)
    if not result.clustered:
        console.print("[red]the alternative produced no narratives; see comparison.md[/red]")
        raise typer.Exit(code=1)


def _draw_misinfo_sample(
    target_dir: Path,
    *,
    n: int,
    lang: str | None,
    seed: int | None,
    embeddings: Path | None,
    force: bool,
) -> Path:
    """Draw the weighted sheet for ``sample-for-labelling misinfo`` and its sampling record."""
    import ingest.factcheck as factcheck
    from ingest.config import Settings as IngestSettings
    from modeling.config import FactCheckDomainsMissing, fact_check_domains
    from modeling.eval.claim_dedup import (
        ClaimDedupConfigError,
        claim_dedup_config,
        default_embedder,
    )
    from modeling.eval.corpus_transfer import SAMPLE_LANGUAGES
    from modeling.eval.labelling import (
        AllocationError,
        build_weighted_sample,
        claim_evidence,
        design_path,
        load_sampling_config,
    )
    from modeling.io import CorpusReader

    settings = get_settings()
    if lang not in SAMPLE_LANGUAGES:
        raise typer.BadParameter(
            f"--lang is required and must be one of {', '.join(SAMPLE_LANGUAGES)}"
        )
    path = target_dir / f"misinfo_corpus_sample_{lang}.csv"
    if path.exists() and not force:
        console.print(
            f"[red]{path} exists and may hold labels.[/] Move it, or pass --force to replace it."
        )
        raise typer.Exit(code=1)
    try:
        domains = fact_check_domains()
        config = load_sampling_config()
    except (FactCheckDomainsMissing, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    records = CorpusReader(settings).records(
        columns=["id", "source", "source_detail", "text", "lang", "raw", "simhash"]
    )
    if not len(records):
        console.print("[red]no corpus on disk[/]")
        raise typer.Exit(code=1)

    rows = factcheck.read_deduplicated_factchecks(
        IngestSettings(data_dir=settings.data_dir, _env_file=None)
    ).kept
    embedder_factory = None
    expected_model = None
    if embeddings is not None:
        try:
            dedup = claim_dedup_config()
        except ClaimDedupConfigError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        expected_model = config.embedding_model or dedup.model_name
        embedder_factory = lambda: default_embedder(expected_model or "", settings)  # noqa: E731
    evidence = claim_evidence(
        rows,
        config,
        embeddings_path=embeddings,
        embedder_factory=embedder_factory,
        expected_model=expected_model,
    )
    try:
        built = build_weighted_sample(
            records,
            n=n,
            # Hindi titles go to Laya, so the Hindi pool uses Laya's language list.
            language_allowed=lambda value: settings.language_allowed(
                "laya" if lang == "hi" else "misinfo", value
            ),
            fact_checker_domains=domains,
            config=config,
            claims=evidence,
            lang=lang,
            seed=seed,
        )
    except AllocationError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"pool {built.pool} records"
        + (f" in lang={lang}" if lang else "")
        + "; left out: "
        + (", ".join(f"{k}={v}" for k, v in sorted(built.excluded.items())) or "none")
    )
    if built.degraded:
        console.print(
            "degraded paths: " + ", ".join(f"{k}={v}" for k, v in sorted(built.degraded.items()))
        )
    for note in built.notes:
        console.print(f"[yellow]{note}[/]", soft_wrap=True)
    console.print(
        "claim matches in the scorable pool: "
        + ", ".join(f"{k}={v}" for k, v in built.claim_match_counts.items())
    )
    if not len(built.frame):
        console.print("[red]no scorable record in the pool; nothing to label[/]")
        raise typer.Exit(code=1)
    # No model score goes in the sheet: a labeller who sees one anchors on it.
    # `evaluate misinfo` joins the scores from record_scores by id.
    built.frame.to_csv(path, index=False)
    # The design file holds the stratum and the weight. Labellers must not receive it.
    built.design.to_csv(design_path(path), index=False)
    record = {
        "seed": built.seed,
        "n_requested": n,
        "n_drawn": len(built.frame),
        "lang": lang,
        "pool": built.pool,
        "left_out": built.excluded,
        "degraded": built.degraded,
        "claim_matches_in_scorable_pool": built.claim_match_counts,
        "strata": built.strata,
        "notes": built.notes,
        "weight": "stratum pool size over stratum rows drawn; evaluation uses pool over rows used",
        "claim_match": {
            "simhash_hamming": config.simhash_hamming,
            "min_similarity": config.min_similarity,
            "embedding_model": expected_model,
        },
    }
    path.with_suffix(".sampling.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    console.print(
        "strata (code: meaning, pool, drawn): "
        + "; ".join(f"{k}: {v['name']}, {v['pool']}, {v['drawn']}" for k, v in built.strata.items())
    )
    return path


@app.command(name="sample-for-labelling")
def sample_for_labelling(
    what: str = typer.Argument(..., help="misinfo | narratives"),
    n: int = typer.Option(100, help="How many rows to sample."),
    lang: str | None = typer.Option(
        None, "--lang", help="misinfo only, required: draw the en or the hi sample."
    ),
    seed: int | None = typer.Option(
        None,
        help="misinfo only: sampling seed. Default is the seed in configs/labelling.yaml "
        "(20260813). The sheet records it.",
    ),
    embeddings: Path | None = typer.Option(
        None,
        help="misinfo only: Parquet file with `id` and `embedding` for corpus records. "
        "It turns on the embedding match to fact-check claims. Without it, only simhash "
        "matches count and the output says so.",
    ),
    force: bool = typer.Option(
        False, "--force", help="misinfo only: replace a sheet that already exists."
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Write a blank-label CSV for hand-labelling.

    This is how the two most informative tables in the whole report get
    produced, and neither can be automated:

    * ``misinfo`` — real corpus records, drawn with weights towards likely
      positives (Reddit, Mastodon, YouTube and Telegram posts, and posts that match a
      fact-check claim) and without fact-checker articles. Each row records its
      stratum and weight. Scoring the labelled rows measures the benchmark-to-corpus
      transfer gap, which is the only number that describes production behaviour.
    * ``narratives`` — narrative-level risk labels, which the ablation table
      needs before its quality columns mean anything.
    """
    _bootstrap(verbose)

    from modeling.io import ScoredStore

    settings = get_settings()
    target_dir = settings.artifacts_dir / "hand_labels"
    target_dir.mkdir(parents=True, exist_ok=True)

    if what == "misinfo":
        path = _draw_misinfo_sample(
            target_dir, n=n, lang=lang, seed=seed, embeddings=embeddings, force=force
        )
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
        "[yellow]Fill in the blank columns by hand[/], then rerun "
        f"`modeling {'relabel-eval' if what == 'misinfo' else 'ablate'}`. "
        "Until they are filled in, the corresponding table reports the gap as unmeasured."
    )


@app.command(name="model-version")
def model_version_command(module: str = typer.Argument(...)) -> None:
    """The version string this module stamps into every scored row."""
    _bootstrap()
    console.print(model_version(module))


if __name__ == "__main__":
    app()
