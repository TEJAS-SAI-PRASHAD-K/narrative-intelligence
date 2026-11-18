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

With ``min_cluster_sizes`` it also clusters again for each size, with both
embedders, into stores of their own under ``sweep/``. Each run reports language
purity: the share of a cluster that belongs to its majority language.

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

import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config
from modeling.eval.ablation import AblationResult, run_ablation, write_ablation
from modeling.io import ScoredStore

#: Tables the cluster stage owns. They are not copied, so the alternative run
#: starts with no previous narratives to carry ids from.
NARRATIVE_TABLES = ("narratives", "narrative_membership")


#: A cluster whose majority language holds less than this share is mixed.
MIXED_BELOW = 0.9

#: Language codes that carry no language. Their records are counted and left out of purity.
UNKNOWN_LANGUAGES = frozenset({"", "und", "unknown", "none", "nan"})


def language_purity(
    membership: pd.DataFrame, lang_by_id: dict[str, Any], *, mixed_below: float = MIXED_BELOW
) -> dict[str, Any]:
    """Language purity of every cluster.

    Purity of one cluster is the share of its members in the majority language.
    The weighted mean uses the member count of each cluster as the weight. A
    cluster is mixed when its purity is below ``mixed_below``. A member with no
    language, or one the corpus does not list, is counted in ``lang_unknown`` and
    left out of the share. A cluster with no member that has a language is
    reported with ``purity`` None and does not enter the mean.
    """
    if membership is None or not len(membership):
        return {"status": "no_clusters", "clusters": [], "n_clusters": 0}
    counts: dict[str, dict[str, int]] = {}
    unknown: dict[str, int] = {}
    for record_id, cluster in zip(
        membership["record_id"].astype(str), membership["narrative_id"].astype(str), strict=True
    ):
        lang = lang_by_id.get(record_id)
        key = None if lang is None or str(lang).strip().lower() in UNKNOWN_LANGUAGES else str(lang)
        if key is None:
            unknown[cluster] = unknown.get(cluster, 0) + 1
            counts.setdefault(cluster, {})
            continue
        counts.setdefault(cluster, {})
        counts[cluster][key] = counts[cluster].get(key, 0) + 1

    rows: list[dict[str, Any]] = []
    for cluster in sorted(counts):
        langs = counts[cluster]
        known = sum(langs.values())
        majority = max(langs, key=lambda k: (langs[k], k)) if langs else None
        rows.append(
            {
                "narrative_id": cluster,
                "members": known + unknown.get(cluster, 0),
                "known_members": known,
                "lang_unknown": unknown.get(cluster, 0),
                "majority_lang": majority,
                "purity": round(langs[majority] / known, 4) if known else None,
                "languages": dict(sorted(langs.items())),
            }
        )
    scored = [r for r in rows if r["purity"] is not None]
    total = sum(r["known_members"] for r in scored)
    weighted = sum(r["purity"] * r["known_members"] for r in scored) / total if total else None
    return {
        "status": "measured" if scored else "no_known_language",
        "n_clusters": len(rows),
        "weighted_mean_purity": round(weighted, 4) if weighted is not None else None,
        "mixed_clusters": sum(r["purity"] < mixed_below for r in scored),
        "mixed_below": mixed_below,
        "lang_unknown": sum(unknown.values()),
        "clusters": rows,
    }


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
    baseline_purity: dict[str, Any] = field(default_factory=dict)
    alternative_purity: dict[str, Any] = field(default_factory=dict)
    #: One row per embedder and ``min_cluster_size`` of the sweep.
    sweep: list[dict[str, Any]] = field(default_factory=list)

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
    min_cluster_sizes: list[int] | None = None,
    settings: ModelingSettings | None = None,
) -> EmbedderComparison:
    """Run the alternative embedder through the clustering checks and the ablation.

    ``min_cluster_sizes`` adds a sweep. Both embedders cluster again at each size.
    """
    for size in min_cluster_sizes or []:
        if size < 2:
            raise ValueError(f"min_cluster_size must be 2 or more, got {size}")
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

    lang_by_id = _languages(demo, sources, settings)
    result.alternative_purity = language_purity(store.read("narrative_membership"), lang_by_id)
    result.baseline_purity = language_purity(source.read("narrative_membership"), lang_by_id)
    for size in min_cluster_sizes or []:
        for role, name in (("baseline", baseline_model), ("alternative", model_name)):
            result.sweep.append(
                _sweep_run(
                    role, name, size, source, out, lang_by_id,
                    demo=demo, sources=sources, limit=limit, settings=settings,
                )
            )

    result.baseline_ablation = run_ablation(demo=demo, settings=settings, store=source)
    result.alternative_ablation = run_ablation(demo=demo, settings=settings, store=store)
    result.written = _write(result, settings)
    return result


def _languages(demo: bool, sources: list[str] | None, settings: ModelingSettings) -> dict:
    """Record id to language code, read from the corpus."""
    from modeling.scoring import _reader_for

    frame = _reader_for(demo, settings).records(sources=sources, columns=["id", "lang"])
    if not len(frame):
        return {}
    return dict(zip(frame["id"].astype(str), frame["lang"], strict=True))


def _sweep_run(
    role: str,
    model_name: str,
    size: int,
    source: ScoredStore,
    out: Path,
    lang_by_id: dict,
    *,
    demo: bool,
    sources: list[str] | None,
    limit: int | None,
    settings: ModelingSettings,
) -> dict[str, Any]:
    """Cluster once with one embedder at one size, in a store of its own."""
    from modeling.scoring import run_stages

    target = out / "sweep" / f"{role}-mcs{size}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    _copy_scored_tables(source, target)
    store = ScoredStore(settings, root=target, manifest_path=target / "manifest.json")
    notes = run_stages(
        ["cluster"],
        demo=demo,
        sources=sources,
        limit=limit,
        settings=settings,
        store=store,
        embed_model=model_name,
        min_cluster_size=size,
    )
    checks = _checks(store.manifest().get("narratives") or {})
    return {
        "embedder": role,
        "model": model_name,
        "min_cluster_size": size,
        "checks": checks,
        "purity": language_purity(store.read("narrative_membership"), lang_by_id),
        "notes": [str(row.get("note", "")) for row in notes],
    }


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
    lines += ["", "## Language purity", "", *_purity_lines(result)]
    if result.sweep:
        lines += ["", "## Sweep over min_cluster_size", "", *_sweep_lines(result.sweep)]
    for title, ablation in (
        ("Ablation, baseline", result.baseline_ablation),
        ("Ablation, alternative", result.alternative_ablation),
    ):
        lines += ["", f"## {title}", ""]
        lines.append(ablation.render().replace("# Module ablation\n", "", 1) if ablation else "")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _purity_lines(result: EmbedderComparison) -> list[str]:
    lines = [
        "Purity is the share of a cluster in its majority language. A cluster is mixed",
        f"when that share is below {MIXED_BELOW}. The mean weights each cluster by its members.",
        "",
        "| embedder | clusters | weighted mean purity | mixed clusters | no language |",
        "|---|---|---|---|---|",
    ]
    for label, purity in (("baseline", result.baseline_purity),
                          ("alternative", result.alternative_purity)):
        lines.append(
            f"| {label} | {_fmt(purity.get('n_clusters'))} "
            f"| {_fmt(purity.get('weighted_mean_purity'))} "
            f"| {_fmt(purity.get('mixed_clusters'))} | {_fmt(purity.get('lang_unknown'))} |"
        )
    for label, purity in (("baseline", result.baseline_purity),
                          ("alternative", result.alternative_purity)):
        rows = purity.get("clusters") or []
        if not rows:
            continue
        lines += [
            "",
            f"Clusters, {label}:",
            "",
            "| cluster | members | majority | purity | languages |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['narrative_id']} | {row['members']} | {_fmt(row['majority_lang'])} "
                f"| {_fmt(row['purity'])} | {row['languages']} |"
            )
    return lines


def _sweep_lines(sweep: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| embedder | min_cluster_size | clusters | clustered fraction | silhouette "
        "| weighted mean purity | mixed clusters |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(sweep, key=lambda r: (r["min_cluster_size"], r["embedder"])):
        checks, purity = row["checks"], row["purity"]
        status = "" if checks else " (no clusters)"
        lines.append(
            f"| {row['embedder']}{status} | {row['min_cluster_size']} "
            f"| {_fmt(checks.get('n_clusters', 0))} | {_fmt(checks.get('clustered_fraction'))} "
            f"| {_fmt(checks.get('silhouette'))} | {_fmt(purity.get('weighted_mean_purity'))} "
            f"| {_fmt(purity.get('mixed_clusters'))} |"
        )
    return lines


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
                "language_purity": {
                    "baseline": result.baseline_purity,
                    "alternative": result.alternative_purity,
                },
                "sweep": result.sweep,
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
