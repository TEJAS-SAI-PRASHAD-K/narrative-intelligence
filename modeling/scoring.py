"""Batch scoring: read Phase 1, run stages, write the scored tables.

One stage per scored table (or per group of columns in one table). Stages run in
the order given by ``configs/scoring.yaml``, because some genuinely depend on
others: ``author_scores.narratives_touched`` needs clustering to have run, and
``author_scores.community_id`` needs the coordination graph.

Everything here is **resumable and idempotent**. A stage asks
``ScoredStore.already_scored`` which keys already carry the current model
versions, skips computing those, and the writer then has nothing new to write.
Killing a run halfway and restarting is safe; rerunning a finished run is a
no-op.

A stage whose model is not trained does not crash the run. It writes nulls with
a reason code and says so in the summary. That is the difference between "we
have not built this yet", which is honest, and an untrained model emitting
confident numbers, which is not.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ingest.config import REPO_ROOT
from modeling.config import ModelingSettings, get_settings, scoring_config
from modeling.io import CorpusReader, ScoredStore

log = logging.getLogger(__name__)

#: A Phase-1-shaped corpus committed for the demo path. Tiny, synthetic, and
#: enough to exercise every stage end to end with no network.
DEMO_CORPUS = REPO_ROOT / "tests" / "fixtures" / "corpus"


@dataclass
class StageContext:
    """Everything a stage needs, resolved once per run."""

    settings: ModelingSettings
    reader: CorpusReader
    store: ScoredStore
    records: pd.DataFrame
    authors: pd.DataFrame
    demo: bool
    dry_run: bool
    known_record_ids: set[str] = field(default_factory=set)
    known_author_ids: set[str] = field(default_factory=set)

    def note(self, message: str) -> str:
        return f"[demo] {message}" if self.demo else message


StageFn = Callable[[StageContext], list[dict[str, Any]]]
_STAGES: dict[str, StageFn] = {}


def stage(name: str) -> Callable[[StageFn], StageFn]:
    def decorate(fn: StageFn) -> StageFn:
        _STAGES[name] = fn
        return fn

    return decorate


def available_stages() -> list[str]:
    return sorted(_STAGES)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
@stage("aux")
def _aux_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Toxicity, sentiment, emotion and post-level anomaly -> record_scores."""
    from modeling.aux import aux_versions, run_aux_pass

    versions = aux_versions()
    done = ctx.store.already_scored("record_scores", versions)
    pending = ctx.records.loc[~ctx.records["id"].astype(str).isin({k[0] for k in done})]
    if not len(pending):
        log.info("aux: all %d records already carry the current versions", len(ctx.records))
        return [
            {
                "stage": "aux",
                "table": "record_scores",
                "written": 0,
                "updated": 0,
                "unchanged": len(ctx.records),
                "note": ctx.note("resumed: nothing to do"),
            }
        ]

    # `pending` decides *whether* to run, not *what* to run.
    #
    # anomaly_score is a within-corpus percentile rank, so it is only meaningful
    # relative to the whole record set. Fitting the IsolationForest on a resumed
    # subset would quietly give a record a different score depending on how the
    # previous run happened to die -- a reproducibility hole that no test on the
    # scorer itself would catch. The text scorers are per-row and hit their
    # caches for the already-done rows, so re-running the full set costs a
    # dictionary lookup each and buys a stable anomaly column.
    log.info(
        "aux: %d of %d records need scoring; running the full set so anomaly ranks "
        "stay corpus-relative",
        len(pending),
        len(ctx.records),
    )
    frame = run_aux_pass(ctx.records, ctx.authors, settings=ctx.settings)
    if not len(frame):
        return [{"stage": "aux", "table": "record_scores", "note": ctx.note("no rows produced")}]

    coverage = _coverage_note(frame, ("toxicity", "sentiment_score", "emotion", "anomaly_score"))
    if ctx.dry_run:
        return [
            {
                "stage": "aux",
                "table": "record_scores",
                "written": 0,
                "note": ctx.note(f"dry-run; would write {len(frame)} rows. {coverage}"),
            }
        ]

    result = ctx.store.write("record_scores", frame, known_keys=ctx.known_record_ids or None)
    ctx.store.update_manifest(
        table="record_scores",
        rows=len(ctx.store.read("record_scores")),
        model_versions=versions,
        extra={"is_demo": ctx.demo},
    )
    return [{"stage": "aux", "table": "record_scores", **result, "note": ctx.note(coverage)}]


@stage("cluster")
def _cluster_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Embed and cluster into narratives -> narratives + narrative_membership."""
    from modeling.config import module_config
    from modeling.text.cluster import NarrativeClusterer
    from modeling.text.embed import Embedder

    policy = scoring_config().get("source_policy") or {}
    records = _apply_length_floor(ctx.records, policy)
    if len(records) < 2:
        return [{"stage": "cluster", "note": ctx.note("too few records to cluster")}]

    embedder = Embedder(ctx.settings)
    if not embedder.load():
        return [
            {
                "stage": "cluster",
                "note": ctx.note(
                    "embedding model unavailable; run `modeling warm-cache` or install "
                    "the 'modeling' extra"
                ),
            }
        ]

    embeddings = embedder.embed_records(records)
    if not len(embeddings):
        return [{"stage": "cluster", "note": ctx.note("nothing embeddable")}]

    # Previous run's narratives, for cross-run id stability. Read before writing.
    previous = ctx.store.read("narratives")
    scores = ctx.store.read("record_scores")

    clusterer = NarrativeClusterer(ctx.settings)
    result = clusterer.fit(
        records, embeddings, record_scores=scores if len(scores) else None, previous=previous
    )
    if not result.narratives:
        return [
            {
                "stage": "cluster",
                "note": ctx.note(
                    f"no clusters found ({result.noise_ratio:.0%} noise); try a smaller "
                    "min_cluster_size in configs/models.yaml"
                ),
            }
        ]

    versions = {
        "embed": str(module_config("embed").get("version")),
        "cluster": str(module_config("cluster").get("version")),
    }
    now = utcnow_local()
    narrative_rows = [n.as_row(versions, now) for n in result.narratives]
    membership_rows = [
        {
            "record_id": member,
            "narrative_id": n.narrative_id,
            "membership_prob": float(n.membership.get(member, 1.0)),
            "is_representative": member in n.representative_ids,
            "generated_at": now,
        }
        for n in result.narratives
        for member in n.member_ids
    ]

    note = (
        f"{len(result.narratives)} narratives, {result.noise_ratio:.0%} noise, "
        f"silhouette={result.diagnostics.get('silhouette')}, "
        f"ids carried={len(result.transitions.get('carried', []))}"
    )
    if ctx.dry_run:
        return [{"stage": "cluster", "table": "narratives", "note": ctx.note("dry-run; " + note)}]

    # narratives is not merge-friendly: a rerun replaces the whole set, because
    # a narrative that no longer exists must disappear rather than linger as a
    # stale row. Membership follows the same rule for the same reason.
    out = []
    written = ctx.store.write("narratives", pd.DataFrame(narrative_rows), merge=False)
    out.append({"stage": "cluster", "table": "narratives", **written, "note": ctx.note(note)})
    written = ctx.store.write(
        "narrative_membership",
        pd.DataFrame(membership_rows),
        known_keys=ctx.known_record_ids or None,
        merge=False,
    )
    out.append({"stage": "cluster", "table": "narrative_membership", **written, "note": ""})

    ctx.store.update_manifest(
        table="narratives",
        rows=len(narrative_rows),
        model_versions=versions,
        extra={
            "is_demo": ctx.demo,
            "embedding_dim": embeddings.dim,
            "diagnostics": result.diagnostics,
            "transitions": {k: len(v) for k, v in result.transitions.items()},
        },
    )
    return out


def _apply_length_floor(records: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    """Drop records too short to embed meaningfully, per-source.

    GDELT carries article metadata rather than body text (median ~83 characters
    in this corpus). A 30-character headline embeds to something that clusters
    on stopwords, so including it does not add a narrative -- it adds noise that
    the noise ratio then reports as a clustering failure.
    """
    if "source" not in records.columns:
        return records
    keep = pd.Series(True, index=records.index)
    for source, rules in policy.items():
        floor = (rules or {}).get("min_chars_for_clustering")
        if not floor:
            continue
        mask = records["source"].astype(str) == source
        short = mask & (records["text"].fillna("").astype(str).str.len() < int(floor))
        if short.any():
            log.info(
                "cluster: excluding %d %s records below the %d-character floor "
                "(short-text degradation, see configs/scoring.yaml)",
                int(short.sum()),
                source,
                int(floor),
            )
        keep &= ~short
    return records.loc[keep]


def utcnow_local():
    from modeling.io import utcnow

    return utcnow()


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_stages(
    stages: Sequence[str],
    *,
    demo: bool = False,
    sources: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    settings: ModelingSettings | None = None,
) -> list[dict[str, Any]]:
    """Run the named stages over the corpus and return a per-stage summary."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    config = scoring_config()
    corpus_config = config.get("corpus") or {}

    reader = _reader_for(demo, settings)
    resolved_sources = sources or corpus_config.get("sources")
    records = reader.records(
        sources=resolved_sources,
        start=start or corpus_config.get("start_date"),
        end=end or corpus_config.get("end_date"),
        limit=limit or corpus_config.get("limit"),
    )
    if not len(records):
        where = DEMO_CORPUS if demo else settings.normalized_dir
        raise SystemExit(
            f"no records found at {where}.\n"
            + (
                "  the demo corpus fixture is missing; regenerate it with "
                "`python scripts/make_fixtures.py corpus`"
                if demo
                else "  build the Phase 1 corpus first: `python -m ingest.cli fetch --all`, "
                "or run with --demo"
            )
        )
    authors = reader.authors(sources=resolved_sources)

    ctx = StageContext(
        settings=settings,
        reader=reader,
        store=_store_for(demo, settings),
        records=records,
        authors=authors,
        demo=demo,
        dry_run=dry_run,
        known_record_ids=set(records["id"].astype(str)) if "id" in records else set(),
        known_author_ids=set(authors["author_id"].astype(str)) if len(authors) else set(),
    )
    log.info(
        "scoring %d records / %d authors from %d source(s)%s",
        len(records),
        len(authors),
        records["source"].nunique() if "source" in records else 0,
        " [DEMO -- not a result]" if demo else "",
    )

    summary: list[dict[str, Any]] = []
    for name in stages:
        fn = _STAGES.get(name)
        if fn is None:
            summary.append(
                {
                    "stage": name,
                    "note": f"not implemented; available: {', '.join(available_stages())}",
                }
            )
            log.warning("stage %r is not implemented; skipping", name)
            continue
        try:
            summary.extend(fn(ctx))
        except Exception as exc:  # pragma: no cover - stage-level failure
            log.exception("stage %s failed", name)
            summary.append({"stage": name, "note": f"FAILED: {type(exc).__name__}: {exc}"})
    return summary


def _reader_for(demo: bool, settings: ModelingSettings) -> CorpusReader:
    if not demo:
        return CorpusReader(settings)
    if not DEMO_CORPUS.exists():
        raise SystemExit(
            f"demo corpus fixture missing at {DEMO_CORPUS}; "
            "regenerate with `python scripts/make_fixtures.py corpus`"
        )
    return CorpusReader(settings, root=DEMO_CORPUS / "normalized")


def _store_for(demo: bool, settings: ModelingSettings) -> ScoredStore:
    # Demo output goes to its own directory so a demo run can never overwrite
    # scores computed on the real corpus. Mixing them would be the worst kind of
    # bug: a dashboard showing fixture numbers with no sign of it.
    root: Path | None = settings.scored_dir.parent / "scored_demo" if demo else None
    return ScoredStore(settings, root=root)


def _coverage_note(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    parts = []
    total = len(frame)
    for column in columns:
        if column in frame.columns:
            filled = int(frame[column].notna().sum())
            parts.append(f"{column}={100 * filled / total:.0f}%" if total else f"{column}=n/a")
    return "coverage " + " ".join(parts) if parts else ""
