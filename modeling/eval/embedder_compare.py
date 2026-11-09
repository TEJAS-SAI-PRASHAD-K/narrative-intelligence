"""Compare an alternative embedder with the configured one, side by side.

The operator runs this on the GPU host before deciding whether the default in
``configs/models.yaml`` should change. It does three things.

1. Copies the scored tables the current run produced into a directory of its
   own, minus the narrative tables, so every score the clustering reads is the
   same for both models.
2. Runs the existing cluster stage there with the alternative model. That is the
   same code as ``modeling cluster``, so the clustering checks are the ones
   already in use: cluster count, clustered fraction, silhouette and noise.
3. Runs the existing ablation over that store, and writes both ablation tables
   with the clustering checks in one ``comparison.md``.

Nothing is written to ``data/scored``, to the shared scored manifest, or to
``artifacts/eval/ablation``. A model that fails the dimension check raises
before any file is written to the output directory.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modeling.config import ModelingSettings, get_settings, module_config
from modeling.eval.ablation import AblationResult, run_ablation, write_ablation
from modeling.io import ScoredStore

#: Tables the cluster stage owns. They are not copied, so the alternative run
#: starts with no previous narratives to carry ids from.
NARRATIVE_TABLES = ("narratives", "narrative_membership")


@dataclass
class EmbedderComparison:
    model_name: str
    baseline_model: str
    out_dir: Path
    stage_notes: list[str] = field(default_factory=list)
    alternative_checks: dict[str, Any] = field(default_factory=dict)
    baseline_checks: dict[str, Any] = field(default_factory=dict)
    alternative_ablation: AblationResult | None = None
    baseline_ablation: AblationResult | None = None
    written: list[Path] = field(default_factory=list)

    @property
    def clustered(self) -> bool:
        return bool(self.alternative_checks)


def default_out_dir(model_name: str, settings: ModelingSettings | None = None) -> Path:
    settings = settings or get_settings()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()
    return settings.eval_dir / "embedder" / slug


def compare_embedder(
    model_name: str,
    *,
    demo: bool = False,
    out_dir: Path | None = None,
    sources: list[str] | None = None,
    limit: int | None = None,
    settings: ModelingSettings | None = None,
) -> EmbedderComparison:
    """Run the alternative embedder through the clustering checks and the ablation."""
    from modeling.scoring import _store_for, run_stages

    settings = settings or get_settings()
    baseline_model = str(module_config("embed").get("model_name"))
    if model_name == baseline_model:
        raise ValueError(f"{model_name} is already the configured embedder; nothing to compare")

    source = _store_for(demo, settings)
    out = Path(out_dir) if out_dir else default_out_dir(model_name, settings)
    _refuse_shared_output(out, source, settings)

    # Load the model first. A wrong width raises here, before anything is written.
    from modeling.text.embed import Embedder

    Embedder(settings, model_name=model_name).load()

    scored = out / "scored"
    if scored.exists():
        shutil.rmtree(scored)
    scored.mkdir(parents=True)
    _copy_scored_tables(source, scored)
    store = ScoredStore(settings, root=scored, manifest_path=scored / "manifest.json")

    result = EmbedderComparison(model_name=model_name, baseline_model=baseline_model, out_dir=out)
    summary = run_stages(
        ["cluster"],
        demo=demo,
        sources=sources,
        limit=limit,
        settings=settings,
        store=store,
        embed_model=model_name,
    )
    result.stage_notes = [str(row.get("note", "")) for row in summary]

    entry = store.manifest().get("narratives") or {}
    result.alternative_checks = _checks(entry)
    result.baseline_checks = _checks(source.manifest().get("narratives") or {})

    result.baseline_ablation = run_ablation(demo=demo, settings=settings, store=source)
    result.alternative_ablation = run_ablation(demo=demo, settings=settings, store=store)
    result.written = _write(result, settings)
    return result


def _checks(manifest_entry: dict[str, Any]) -> dict[str, Any]:
    """The clustering checks the cluster stage stores in the manifest."""
    if not manifest_entry:
        return {}
    return {
        "embed_model": manifest_entry.get("embed_model"),
        "embedding_dim": manifest_entry.get("embedding_dim"),
        "narratives": manifest_entry.get("rows"),
        **(manifest_entry.get("diagnostics") or {}),
    }


def _copy_scored_tables(source: ScoredStore, target: Path) -> None:
    if not source.root.exists():
        return
    for item in source.root.iterdir():
        if item.name in NARRATIVE_TABLES or item.name == "manifest.json":
            continue
        if item.is_dir():
            shutil.copytree(item, target / item.name)
        else:
            shutil.copy2(item, target / item.name)


def _refuse_shared_output(out: Path, source: ScoredStore, settings: ModelingSettings) -> None:
    """Never write into, or over, a directory the pipeline itself uses."""
    out = out.resolve()
    shared = [
        source.root,
        settings.scored_dir,
        settings.scored_dir.parent / "scored_demo",
        settings.eval_dir / "ablation",
    ]
    for path in shared:
        path = Path(path).resolve()
        if out == path or path in out.parents or out in path.parents:
            raise ValueError(
                f"output directory {out} overlaps {path}. Pick a separate directory, "
                "for example artifacts/eval/embedder/<name>."
            )


_CHECK_ROWS = (
    ("embedding model", "embed_model"),
    ("dimension", "embedding_dim"),
    ("narratives", "narratives"),
    ("clusters", "n_clusters"),
    ("clustered fraction", "clustered_fraction"),
    ("silhouette", "silhouette"),
)


def _render(result: EmbedderComparison) -> str:
    def cell(checks: dict[str, Any], key: str) -> str:
        value = checks.get(key)
        return "n/a" if value is None else str(value)

    lines = [
        "# Embedder comparison",
        "",
        f"- baseline: `{result.baseline_model}`",
        f"- alternative: `{result.model_name}`",
        "",
        "The alternative ran through the same cluster stage and the same ablation as the",
        "baseline, over the same scored tables. The default in `configs/models.yaml` is",
        "not changed by this run.",
        "",
        "## Clustering checks",
        "",
    ]
    if not result.baseline_checks:
        lines += [
            "The baseline has no narratives on disk, so its column is empty. Run",
            "`python -m modeling.cli cluster` with the default model first.",
            "",
        ]
    if not result.clustered:
        lines += [
            "**The alternative produced no narratives.** Notes from the stage:",
            "",
            *[f"- {note}" for note in result.stage_notes],
            "",
        ]
    lines += ["| check | baseline | alternative |", "|---|---|---|"]
    for label, key in _CHECK_ROWS:
        lines.append(
            f"| {label} | {cell(result.baseline_checks, key)} | "
            f"{cell(result.alternative_checks, key)} |"
        )
    for title, ablation in (
        ("Ablation, baseline", result.baseline_ablation),
        ("Ablation, alternative", result.alternative_ablation),
    ):
        lines += ["", f"## {title}", ""]
        lines.append(ablation.render().replace("# Module ablation\n", "", 1) if ablation else "")
    return "\n".join(lines) + "\n"


def _write(result: EmbedderComparison, settings: ModelingSettings) -> list[Path]:
    out = result.out_dir
    written: list[Path] = []
    path = out / "clustering.json"
    path.write_text(
        json.dumps(
            {
                "baseline": result.baseline_checks,
                "alternative": result.alternative_checks,
                "stage_notes": result.stage_notes,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(path)
    if result.alternative_ablation is not None:
        written += write_ablation(result.alternative_ablation, settings=settings, target_dir=out)
    path = out / "comparison.md"
    path.write_text(_render(result), encoding="utf-8")
    written.append(path)
    return written
