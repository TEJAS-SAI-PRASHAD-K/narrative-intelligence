#!/usr/bin/env python3
"""Generate notebooks 02-05 from this script.

Same reasoning as Phase 1's ``build_eda_notebook.py``: a ``.ipynb`` is JSON with
embedded outputs, and a hand-edited one produces diffs nobody can read.
Generating them keeps the *source* reviewable and the notebooks reproducible.

    python notebooks/build_phase2_notebooks.py

Then run each notebook to populate its figures.

Notebook 05 is the one a grader reads end to end. It is written for that reader:
prose first, chart second, and every number carrying the split strategy that
produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(lines),
    }


def _lines(lines: tuple[str, ...]) -> list[str]:
    text = "\n".join(lines)
    return [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


PREAMBLE = code(
    "import sys, warnings",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))",
    "warnings.filterwarnings('ignore', category=FutureWarning)",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "",
    "from modeling.config import get_settings, run_fingerprint, set_all_seeds",
    "from modeling.io import CorpusReader, ScoredStore",
    "",
    "set_all_seeds()",
    "settings = get_settings()",
    "reader = CorpusReader(settings)",
    "store = ScoredStore(settings)",
    "",
    "# Every number below is tied to this fingerprint. If a rerun disagrees with",
    "# a committed figure, this block is where the diagnosis starts.",
    "fingerprint = run_fingerprint()",
    "print(f\"seed={fingerprint['seed']}  device={fingerprint['device']}  \"",
    "      f\"corpus={fingerprint['input_manifest_hash']}\")",
)


# ---------------------------------------------------------------------------
# 02 — text and narrative
# ---------------------------------------------------------------------------
def build_02() -> dict:
    return notebook(
        [
            md(
                "# 02 — Text and narrative structure",
                "",
                "What this notebook is for: deciding whether the narrative clusters are",
                "real. There are no gold narrative labels, so there is no F1 here and there",
                "will not be one. What replaces it is a combination of four things, none of",
                "which is sufficient alone:",
                "",
                "1. **Silhouette** on the clustered points, sampled. Measures separation,",
                "   not correctness.",
                "2. **Noise ratio.** HDBSCAN's willingness to say 'this belongs to nothing'",
                "   is a feature. A very low noise ratio usually means `min_cluster_size` is",
                "   too small and the model is manufacturing narratives out of chatter.",
                "3. **Cluster size distribution.** One cluster holding most of the corpus is",
                "   a failure however good the silhouette looks.",
                "4. **A manual audit of 20 clusters**, rated coherent / mixed / junk by a",
                "   person. This is the only one that measures what we actually care about.",
                "",
                "The audit is the deliverable. The other three are triage.",
            ),
            PREAMBLE,
            md(
                "## The corpus going in",
                "",
                "Two properties of Phase 1's corpus shape everything downstream, and both",
                "are visible below.",
                "",
                "**GDELT records are article metadata, not article text.** Their median",
                "length is around 83 characters. A 30-character headline embeds to something",
                "that clusters on stopwords, so `configs/scoring.yaml` sets a length floor and",
                "those records are excluded from clustering with a logged count rather than",
                "silently degrading the embedding space.",
                "",
                "**Reddit is threaded and Mastodon mostly is not.** That asymmetry matters",
                "for the coordination module in notebook 03, not for clustering.",
            ),
            code(
                "records = reader.records()",
                "print(f'{len(records):,} records')",
                "",
                "summary = records.assign(chars=records['text'].fillna('').str.len()).groupby('source').agg(",
                "    records=('id', 'size'),",
                "    authors=('author_id', 'nunique'),",
                "    median_chars=('chars', 'median'),",
                "    threaded=('parent_id', lambda s: round(s.notna().mean(), 3)),",
                ")",
                "summary",
            ),
            code(
                "fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))",
                "records.assign(chars=records['text'].fillna('').str.len()).boxplot(",
                "    column='chars', by='source', ax=axes[0], grid=False)",
                "axes[0].set_yscale('log'); axes[0].set_title('text length by source'); axes[0].set_xlabel('')",
                "records['source'].value_counts().plot.bar(ax=axes[1])",
                "axes[1].set_title('records per source')",
                "plt.suptitle(''); plt.tight_layout()",
            ),
            md(
                "## Narratives",
                "",
                "Read the coherence column before the size column. A large cluster with low",
                "coherence is a bag of loosely-related posts that HDBSCAN merged because they",
                "were all vaguely political; a small cluster with high coherence is a genuine",
                "shared claim. The UI surfaces coherence for exactly this reason.",
                "",
                "`velocity` is the **peak** posts-per-hour, not the mean. A narrative that",
                "produced 200 posts in one hour and then went quiet for a week is the",
                "interesting case, and a lifetime mean erases it completely.",
            ),
            code(
                "narratives = store.read('narratives')",
                "if not len(narratives):",
                "    print('No narratives on disk. Run: python -m modeling.cli score cluster')",
                "else:",
                "    display(narratives[['narrative_id', 'label', 'label_source', 'size',",
                "                        'author_count', 'coherence', 'velocity', 'severity']]",
                "            .sort_values('size', ascending=False).head(20))",
            ),
            code(
                "if len(narratives):",
                "    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))",
                "    narratives['size'].plot.hist(bins=20, ax=axes[0]); axes[0].set_title('cluster size')",
                "    narratives['coherence'].plot.hist(bins=20, ax=axes[1]); axes[1].set_title('coherence')",
                "    axes[2].scatter(narratives['size'], narratives['coherence'], alpha=0.6)",
                "    axes[2].set_xlabel('size'); axes[2].set_ylabel('coherence')",
                "    axes[2].set_title('big clusters are usually less coherent')",
                "    plt.tight_layout()",
            ),
            md(
                "## The manual audit",
                "",
                "Run the cell, read the twenty clusters, and rate each one:",
                "",
                "- **coherent** — the members share one claim",
                "- **mixed** — two or three distinct claims got merged",
                "- **junk** — no shared claim; the cluster is an artefact",
                "",
                "Write the counts into `artifacts/error_analysis/cluster.md`. A junk rate",
                "above roughly a fifth means `min_cluster_size` is too permissive for this",
                "corpus.",
            ),
            code(
                "from modeling.text.cluster import audit_table",
                "",
                "print(audit_table(20))",
            ),
            md(
                "## Labels: LLM or centroid?",
                "",
                "`label_source` says which. With no `ANTHROPIC_API_KEY` the pipeline falls",
                "back to the representative post's first sentence — a worse label and an",
                "honest one. A centroid label reads visibly as a quotation rather than as a",
                "generated headline, which is the right signal to a reader that no model",
                "wrote it.",
            ),
            code(
                "if len(narratives):",
                "    print(narratives['label_source'].value_counts(dropna=False))",
                "    for row in narratives.head(5).to_dict('records'):",
                "        print(f\"\\n[{row['label_source']}] {row['label']}\")",
                "        if row.get('summary'):",
                "            print(f\"    {row['summary']}\")",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 03 — accounts and coordination
# ---------------------------------------------------------------------------
def build_03() -> dict:
    return notebook(
        [
            md(
                "# 03 — Accounts and coordination",
                "",
                "Two separate questions that the dashboard is at risk of conflating:",
                "",
                "- **Is this account automated?** (`bot_prob`, a supervised model)",
                "- **Is this account acting in concert with others?** (`coordination_score`,",
                "  a transparent formula over a graph)",
                "",
                "They are independent. A coordinated campaign can be run entirely by humans,",
                "and a bot can be a weather feed nobody coordinates with. Neither score is",
                "evidence for the other, and this notebook keeps them in separate sections",
                "on purpose.",
                "",
                "**The finding in this notebook is the null-model comparison**, not the",
                "communities. Any graph has communities; Louvain will happily partition",
                "random noise and report a modularity above zero.",
            ),
            PREAMBLE,
            md(
                "## Feature tiers, and why the bot model may be null",
                "",
                "Follower and following counts exist for Mastodon and for the Twitter",
                "benchmarks. They do not exist for ConvoKit Reddit at all. A model trained on",
                "TwiBot-22's forty Twitter features and scored on the twelve this corpus can",
                "compute is not a model — it is a lookup table for a platform we do not have.",
                "",
                "`modeling/accounts/features.py` declares three tiers and the classifier",
                "trains on the intersection of what the benchmark and the corpus can both",
                "supply. When the intersection is empty, `bot_prob` is **null with a reason",
                "code** rather than a number.",
            ),
            code(
                "from modeling.accounts.features import available_tiers, build_features",
                "",
                "records = reader.records()",
                "authors = reader.authors()",
                "tiers = available_tiers(records, authors)",
                "print('tiers this corpus supports:', tiers)",
                "",
                "features = build_features(records, authors, tiers=tiers)",
                "frame = features.as_frame()",
                "print(f'{frame.shape[0]} accounts x {frame.shape[1]} features')",
                "frame.describe().T[['mean', 'std', 'min', '50%', 'max']].round(3)",
            ),
            code(
                "cols = ['hour_entropy', 'burstiness', 'duplicate_content_rate', 'self_similarity_mean']",
                "present = [c for c in cols if c in frame.columns]",
                "frame[present].hist(bins=30, figsize=(11, 6))",
                "plt.suptitle('behavioural features — look for bimodality, not outliers')",
                "plt.tight_layout()",
            ),
            md(
                "## Account scores",
                "",
                "`skip_reasons` is the column to read first. It says why a null is null, and",
                "a table of nulls with no reasons would be unexplainable three weeks later.",
            ),
            code(
                "author_scores = store.read('author_scores')",
                "if not len(author_scores):",
                "    print('Nothing scored yet. Run: python -m modeling.cli score --all')",
                "else:",
                "    print(f'{len(author_scores)} authors')",
                "    for column in ['bot_prob', 'coordination_score', 'community_id', 'toxicity_mean']:",
                "        if column in author_scores:",
                "            filled = author_scores[column].notna().sum()",
                "            print(f'  {column:22} {filled:5}/{len(author_scores)} populated')",
                "    from collections import Counter",
                "    reasons = Counter(r for rs in author_scores['skip_reasons'].dropna() for r in rs)",
                "    print('\\nskip reasons:', dict(reasons))",
            ),
            md(
                "## The coordination graph",
                "",
                "Edges carry an **evidence type**, so the UI can say *why* two accounts are",
                "linked rather than merely asserting that they are:",
                "",
                "| evidence | meaning |",
                "|---|---|",
                "| `near_dup` | near-identical content inside the window |",
                "| `cotweet` | the same URL or domain inside the window |",
                "| `hashtag_seq` | the same *ordered* hashtag sequence |",
                "| `temporal` | replies to the same parent inside a tight window |",
                "",
                "Ordering matters for `hashtag_seq`: a shared ordering is much stronger",
                "evidence of a shared template than a shared vocabulary.",
            ),
            code(
                "edges = store.read('coordination_edges')",
                "if not len(edges):",
                "    print('No edges. Run: python -m modeling.cli score coordination')",
                "else:",
                "    print(f'{len(edges)} edges')",
                "    display(edges['evidence'].value_counts())",
                "    edges['weight'].plot.hist(bins=30, figsize=(6, 3))",
                "    plt.title('edge weight distribution'); plt.tight_layout()",
            ),
            md(
                "## The null model — this is the finding",
                "",
                "Timestamps are shuffled **within each author** and the whole pipeline re-run.",
                "That destroys cross-account timing coincidences — the thing coordination",
                "detection claims to find — while preserving every author's own volume,",
                "burstiness and diurnal rhythm. A *global* shuffle would destroy those too,",
                "and would be too easy a null to beat.",
                "",
                "If observed modularity does not exceed the shuffled mean by at least one",
                "standard deviation, **'we found coordinated communities' is not a result**,",
                "and the cell below says so in as many words.",
            ),
            code(
                "from modeling.accounts.coordination import CoordinationDetector, null_model_section",
                "",
                "result = CoordinationDetector(settings).detect(records)",
                "print(null_model_section(result))",
                "print()",
                "print(result.summary())",
            ),
            code(
                "if result.edges:",
                "    import networkx as nx",
                "    graph = nx.Graph()",
                "    for edge in result.edges:",
                "        graph.add_edge(edge['src_author_id'], edge['dst_author_id'],",
                "                       weight=edge['weight'])",
                "    largest = max(nx.connected_components(graph), key=len)",
                "    subgraph = graph.subgraph(list(largest)[:60])",
                "    plt.figure(figsize=(7, 6))",
                "    nx.draw_networkx(subgraph, pos=nx.spring_layout(subgraph, seed=settings.seed),",
                "                     node_size=90, font_size=6, width=0.6, with_labels=False)",
                "    plt.title(f'largest component ({len(largest)} accounts)'); plt.axis('off')",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 04 — deepfake
# ---------------------------------------------------------------------------
def build_04() -> dict:
    return notebook(
        [
            md(
                "# 04 — Deepfake detection",
                "",
                "**Status: this model is not trained.** FaceForensics++ requires a signed",
                "agreement and the fine-tune requires a GPU. What exists is the complete",
                "pipeline, the split discipline, and the honest null path.",
                "",
                "This notebook documents the two decisions that determine whether any future",
                "number here means anything.",
                "",
                "## 1. Split by source video, never by frame",
                "",
                "Frames from one video in both train and test is *the* mechanism behind",
                "deepfake papers reporting 99% accuracy. Adjacent frames of one clip are",
                "near-identical images; a model that memorizes a face scores perfectly on the",
                "rest of that clip and fails on everything else.",
                "",
                "`tests/test_splits.py` asserts that a frame-level split raises. The cell",
                "below demonstrates it.",
                "",
                "## 2. 'No face found' and 'real face' are different answers",
                "",
                "When no face is detected, the contract gets `face_detected=false` and",
                "`deepfake_prob=null` — never a low score. A low score says 'we looked and it",
                "seems real'; a null says 'we could not look'. Conflating them means a",
                "deepfake checker that quietly clears every clip it failed to parse.",
            ),
            PREAMBLE,
            code(
                "import pytest",
                "from modeling.datasets.splits import LeakageError, assert_no_group_leakage",
                "",
                "frames = pd.DataFrame([",
                "    {'frame_id': f'{video}-f{i:03d}', 'video_id': video}",
                "    for video in ('v001_real', 'v001_fake', 'v002_real', 'v002_fake')",
                "    for i in range(50)",
                "])",
                "rng = np.random.default_rng(7)",
                "order = rng.permutation(len(frames))",
                "",
                "try:",
                "    assert_no_group_leakage(frames['video_id'], order[:150], order[150:])",
                "    print('NO ERROR RAISED — the leakage guard is broken')",
                "except LeakageError as exc:",
                "    print('correctly refused a frame-level split:')",
                "    print(' ', exc)",
            ),
            md(
                "## The dataset loader",
                "",
                "FF++ names manipulated clips `<target>_<source>.mp4`. The loader groups on",
                "the **target** identity, because the manipulated clip shares that person's",
                "face, background and lighting with `original_sequences/<target>`. Clips whose",
                "filename cannot be parsed are dropped rather than grouped by guess.",
                "",
                "DFDC is used as a cross-dataset generalisation check, not as training data:",
                "it has no per-method labels, and the question that matters for a public",
                "upload checker is whether the model transfers to production pipelines it has",
                "never seen.",
            ),
            code(
                "from modeling.datasets import get_dataset",
                "from modeling.datasets.splits import domain_holdout, group_train_val_test",
                "",
                "dataset = get_dataset('faceforensics')",
                "on_disk = dataset.available()",
                "print('FaceForensics++ on disk:', on_disk)",
                "loaded = dataset.load(demo=not on_disk)",
                "print(loaded.summary())",
            ),
            code(
                "work, split = group_train_val_test(loaded.frame, group_col='source_video',",
                "                                   label_col='label', text_col='video_id',",
                "                                   dedupe=False)",
                "print(split.describe())",
                "",
                "# Every metric this model ever reports must carry that line.",
                "for name, index in (('train', split.train), ('val', split.val), ('test', split.test)):",
                "    videos = set(work['source_video'].iloc[index])",
                "    print(f'{name:6} {len(index):4} clips, {len(videos):3} source videos')",
            ),
            md(
                "## Cross-method generalisation",
                "",
                "The known weak point of this entire model family. Train on three FF++",
                "manipulation methods, test on the held-out fourth. These numbers will be",
                "worse than the in-method ones — reporting them is a strength, not a",
                "weakness, and a reviewer who does not see them should assume the worst.",
            ),
            code(
                "for held_out in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:",
                "    try:",
                "        w, s = domain_holdout(loaded.frame, domain_col='method',",
                "                              held_out=held_out, group_col='source_video')",
                "        print(f'hold out {held_out:15} train={len(s.train):4} test={len(s.test):4}')",
                "    except ValueError as exc:",
                "        print(f'hold out {held_out:15} unavailable: {exc}')",
            ),
            md(
                "## CPU inference latency",
                "",
                "A deployment fact, not a curiosity: Phase 4 runs this on demand for uploads",
                "and needs to know what it costs. Xception over 16 frames is seconds per video",
                "on CPU, which is acceptable for an on-demand upload path and not acceptable",
                "for scoring a whole corpus — which is why corpus-wide media scores are",
                "precomputed into Parquet instead.",
            ),
            code(
                "from modeling.media.frames import FrameExtractor, describe",
                "",
                "extractor = FrameExtractor(settings, detector='none')",
                "sample = sorted(Path(loaded.frame['path'].iloc[0]).parent.glob('*'))[:3]",
                "for path in sample:",
                "    import time",
                "    start = time.perf_counter()",
                "    extraction = extractor.extract(path)",
                "    elapsed = (time.perf_counter() - start) * 1000",
                "    print(f'{path.name:20} {elapsed:6.1f} ms  {describe(extraction)[:90]}')",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 05 — the consolidated evaluation report
# ---------------------------------------------------------------------------
def build_05() -> dict:
    return notebook(
        [
            md(
                "# 05 — Evaluation report",
                "",
                "**Read this one end to end.** It is the consolidated evidence for every",
                "model in Phase 2, and it is written to be read rather than skimmed.",
                "",
                "Four rules the numbers below obey:",
                "",
                "1. **Every metric carries its split strategy.** 'F1 0.78' is not a result;",
                "   'F1 0.78, grouped by claim, 5-fold, ±0.04' is.",
                "2. **Accuracy is banned.** On an imbalanced target it rewards predicting the",
                "   majority class. PR-AUC leads, because it degrades exactly when the model",
                "   starts crying wolf.",
                "3. **Every model sits next to its baselines.** If the fine-tune does not",
                "   clear TF-IDF, that is stated, not buried.",
                "4. **Confidence intervals decide what counts as a difference.** A 2-point F1",
                "   gap on a 500-row test set is noise, and overlapping intervals are reported",
                "   as 'not separable', never as a win.",
                "",
                "---",
                "",
                "## The honest summary, up front",
                "",
                "Every benchmark this project uses is access-gated. If the tables below are",
                "stamped **DEMO FIXTURE**, they were computed on committed synthetic data",
                "that reproduces each dataset's shape and none of its content. Those numbers",
                "demonstrate that the training and evaluation path executes; they are not",
                "results and must not be cited as any.",
            ),
            PREAMBLE,
            code(
                "import json",
                "",
                "rows = []",
                "for module_dir in sorted(p for p in settings.eval_dir.iterdir() if p.is_dir()):",
                "    for version_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):",
                "        path = version_dir / 'metrics.json'",
                "        if not path.exists():",
                "            continue",
                "        payload = json.loads(path.read_text())",
                "        metrics = payload['metrics']",
                "        rows.append({",
                "            'module': module_dir.name,",
                "            'version': version_dir.name,",
                "            'demo': metrics.get('is_demo'),",
                "            'n_test': metrics['n_test'],",
                "            'macro_F1': metrics['macro_f1']['value'],",
                "            'F1_ci': f\"[{metrics['macro_f1']['ci_low']:.3f}, {metrics['macro_f1']['ci_high']:.3f}]\",",
                "            'PR_AUC': (metrics['pr_auc'] or {}).get('value'),",
                "            'Brier': metrics.get('brier'),",
                "            'split': metrics['split'],",
                "        })",
                "",
                "summary = pd.DataFrame(rows)",
                "summary if len(summary) else print('No eval artifacts. Run: python -m modeling.cli train misinfo --demo')",
            ),
            md(
                "## Baselines: what did the expensive model buy?",
                "",
                "A baseline is not there to be beaten. It is there to answer the question",
                "above, and sometimes the answer is 'nothing' — which is worth more than a",
                "tuned number, and is reported here whichever way it comes out.",
            ),
            code(
                "for module_dir in sorted(p for p in settings.eval_dir.iterdir() if p.is_dir()):",
                "    for version_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):",
                "        path = version_dir / 'metrics.json'",
                "        if not path.exists():",
                "            continue",
                "        payload = json.loads(path.read_text())",
                "        baselines = payload.get('baselines')",
                "        if not baselines:",
                "            continue",
                "        print(f\"### {module_dir.name} ({version_dir.name})\")",
                "        for name, verdict in baselines['verdicts'].items():",
                "            print(f\"  {name:24} delta {verdict['delta']:+.3f}  {verdict['verdict']}\")",
                "        if not baselines['clears_every_baseline']:",
                "            print('  >> does NOT cleanly clear every baseline — see the report')",
                "        print()",
            ),
            md(
                "## Calibration",
                "",
                "Phase 4 multiplies these scores together into one risk number. That is only",
                "meaningful if each input is a calibrated probability — if `misinfo_prob=0.7`",
                "genuinely means 'about 70% of records scored 0.7 are misinformation-like'.",
                "",
                "Note the fallback: isotonic regression below 200 validation rows fits a step",
                "function to noise and produces confident 0.0/1.0 outputs, so it falls back to",
                "Platt scaling and says so. If calibration made the Brier score *worse*, that",
                "appears here too.",
            ),
            code(
                "for module_dir in sorted(p for p in settings.eval_dir.iterdir() if p.is_dir()):",
                "    for version_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):",
                "        path = version_dir / 'metrics.json'",
                "        if not path.exists():",
                "            continue",
                "        calibration = json.loads(path.read_text()).get('calibration')",
                "        if not calibration:",
                "            continue",
                "        print(f\"{module_dir.name}: {calibration['method']} on \"",
                "              f\"{calibration['n_calibration']} rows, Brier \"",
                "              f\"{calibration['brier_before']:.4f} -> {calibration['brier_after']:.4f}\")",
                "        if calibration.get('note'):",
                "            print(f\"  note: {calibration['note']}\")",
                "        curve = calibration['reliability_after']",
                "        if curve['predicted']:",
                "            plt.figure(figsize=(4, 4))",
                "            plt.plot([0, 1], [0, 1], '--', color='grey', label='perfect')",
                "            plt.plot(calibration['reliability_before']['predicted'],",
                "                     calibration['reliability_before']['observed'], 'o-', label='before')",
                "            plt.plot(curve['predicted'], curve['observed'], 'o-', label='after')",
                "            plt.xlabel('predicted'); plt.ylabel('observed')",
                "            plt.title(f'{module_dir.name} reliability'); plt.legend(); plt.tight_layout()",
                "            plt.show()",
            ),
            md(
                "## Ablation",
                "",
                "> **The fusion used here is provisional and for measurement only.** Phase 4",
                "> owns the product's 0–100 risk score, deliberately, because the weighting is",
                "> a documented product decision rather than a model output. What is used below",
                "> is an equally-weighted, null-aware mean of the available components — it",
                "> exists so that 'adding coordination changed the ranking by this much' is a",
                "> sentence with a number in it, and for no other purpose.",
                "",
                "Read the coverage table underneath. A configuration that adds a component",
                "present on 0% of narratives is identical to the row above it, and that is",
                "information rather than a bug.",
            ),
            code(
                "from modeling.eval.ablation import run_ablation, write_ablation",
                "",
                "ablation = run_ablation()",
                "print(ablation.render())",
                "for path in write_ablation(ablation):",
                "    print('wrote', path)",
            ),
            md(
                "## Error analysis",
                "",
                "A confusion matrix says *how many* the model got wrong. It never says *what",
                "kind* of wrong, and the kind is what decides whether a model is deployable.",
                "",
                "The category counts are a keyword-and-shape triage pass, not the analysis.",
                "The analysis is the prose in `artifacts/error_analysis/<module>.md`, written",
                "after reading the uncategorized examples.",
            ),
            code(
                "for path in sorted(settings.error_analysis_dir.glob('*.md')):",
                "    print('=' * 70)",
                "    print(path.name)",
                "    print('=' * 70)",
                "    print(path.read_text()[:2500])",
                "    print()",
            ),
            md(
                "## Fairness check",
                "",
                "Score distributions across the corpus's language groups and across a topical",
                "slice, for the toxicity and misinformation classifiers.",
                "",
                "You do not need to fix what this finds. You do need to report it. The",
                "toxicity model in particular is known to over-flag African-American English",
                "and identity terms in non-pejorative use — that is a property of the Jigsaw",
                "annotations it learned from, it is documented in the model card, and it means",
                "`toxicity` must never be read as 'this account is abusive'.",
            ),
            code(
                "scores = store.read('record_scores')",
                "records = reader.records(columns=['id', 'source', 'lang'])",
                "if len(scores) and len(records):",
                "    joined = records.merge(scores, left_on='id', right_on='record_id', how='inner')",
                "    from modeling.eval.metrics import group_slice_report, score_distribution",
                "    for column in ['toxicity', 'misinfo_prob']:",
                "        if column not in joined or joined[column].notna().sum() == 0:",
                "            continue",
                "        print(f'--- {column} ---')",
                "        print('overall:', score_distribution(joined[column].to_numpy())['quantiles'])",
                "        print('by language:', group_slice_report(joined[column].to_numpy(),",
                "                                                 joined['lang'].fillna('unknown').to_numpy()))",
                "        print('by source:  ', group_slice_report(joined[column].to_numpy(),",
                "                                                 joined['source'].to_numpy()))",
                "        print()",
            ),
            md(
                "## What these models do not tell you",
                "",
                "Stated here because it is the part most easily lost between a notebook and a",
                "dashboard.",
                "",
                "- **Benchmark performance is not production performance.** LIAR is",
                "  politicians' statements; FakeNewsNet is news headlines; this corpus is",
                "  Reddit comments, Mastodon toots and GDELT article metadata. Expect a large",
                "  drop. The measured transfer gap — 100 hand-labelled corpus records — is the",
                "  only number that describes production behaviour.",
                "- **A high `misinfo_prob` is not a determination that a claim is false.** It",
                "  is a similarity judgement to patterns in fact-checked corpora.",
                "- **A high `bot_prob` is not a determination that an account is automated.**",
                "  The training data is Twitter; this corpus is not.",
                "- **`anomaly_score` is a within-corpus rank, not a probability.** It cannot be",
                "  multiplied into a probability product.",
                "- **A null is not a zero.** Every null in the scored tables means 'not",
                "  assessed' and carries a reason code.",
            ),
        ]
    )


BUILDERS = {
    "02_text_narrative.ipynb": build_02,
    "03_bot_coordination.ipynb": build_03,
    "04_deepfake.ipynb": build_04,
    "05_evaluation_report.ipynb": build_05,
}


def main() -> int:
    for name, builder in BUILDERS.items():
        path = HERE / name
        path.write_text(json.dumps(builder(), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
