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

import numpy as np
import pandas as pd

from ingest.config import REPO_ROOT
from modeling.config import ModelingSettings, get_settings, scoring_config
from modeling.io import CorpusReader, ScoredStore, as_list

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
    #: Set by the coordination stage and consumed by the accounts stage, which
    #: needs community_id and coordination_score. The dependency is why
    #: configs/scoring.yaml orders coordination before accounts.
    coordination: Any = None
    #: Overrides ``embed.model_name`` for the cluster stage. Only the embedder
    #: comparison sets it, together with a store of its own.
    embed_model: str | None = None

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

    frame = _keep_misinfo_skips(frame, ctx.store.read("record_scores"))
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
    # The embed entry in models.yaml says which languages are embedded. It is
    # `*` today, so this keeps every record; narrowing it narrows clustering.
    records = records.loc[
        records["lang"].map(lambda lang: ctx.settings.language_allowed("embed", lang))
    ]
    if len(records) < 2:
        return [{"stage": "cluster", "note": ctx.note("too few records to cluster")}]

    embedder = Embedder(ctx.settings, model_name=ctx.embed_model)
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
        # The name as well as the version: changing `embed.model_name` without
        # bumping the version would otherwise leave stale centroids unmarked.
        "embed_model": embedder.model_name,
        "cluster": str(module_config("cluster").get("version")),
    }
    now = utcnow_local()
    narrative_rows = [n.as_row(versions, now) for n in result.narratives]

    # Carry the label forward for narratives whose id survived.
    #
    # Clustering does not produce labels -- the summarize stage does -- so a
    # re-cluster that wrote `label = None` would blank every label, the
    # summarize stage would regenerate them, and the two stages would churn the
    # table against each other on every run. A carried id keeps its label until
    # something re-summarizes it, which is also what the UI expects: the
    # narrative the user is looking at does not lose its name because the
    # corpus grew.
    if len(previous):
        carried = {
            str(row["narrative_id"]): row
            for row in previous.to_dict(orient="records")
        }
        for row in narrative_rows:
            prior = carried.get(str(row["narrative_id"]))
            if prior is None:
                continue
            for column in ("label", "label_source", "summary"):
                if prior.get(column) is not None and row.get(column) is None:
                    row[column] = prior[column]
            # And its provenance. The summarize stage adds its own entries to
            # this map; blanking them here would leave the two stages
            # overwriting each other's model_versions on every run.
            from modeling.io import _as_version_dict

            row["model_versions"] = {
                **_as_version_dict(prior.get("model_versions")),
                **row["model_versions"],
            }
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
            "embed_model": embedder.model_name,
            "diagnostics": result.diagnostics,
            "transitions": {k: len(v) for k, v in result.transitions.items()},
        },
    )
    return out


@stage("misinfo")
def _misinfo_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Calibrated misinformation probability -> record_scores.misinfo_prob."""
    from modeling.config import module_config
    from modeling.registry import resolve
    from modeling.text.misinfo_clf import MisinfoClassifier

    version = str(module_config("misinfo").get("version"))
    checkpoint = resolve("misinfo", version, required=False)
    if checkpoint is None:
        return [
            {
                "stage": "misinfo",
                "table": "record_scores",
                "note": ctx.note(
                    "no trained checkpoint; misinfo_prob stays null. Train with "
                    "`modeling train misinfo` once a benchmark is on disk."
                ),
            }
        ]

    classifier = MisinfoClassifier(ctx.settings)
    if not classifier.load(checkpoint.path):
        return [
            {
                "stage": "misinfo",
                "table": "record_scores",
                "note": ctx.note("checkpoint failed to load (see log); misinfo_prob stays null"),
            }
        ]

    demo_note = warn_if_demo_checkpoint("misinfo", checkpoint, ctx.demo)
    versions = {"misinfo": checkpoint_stamp("misinfo", version, checkpoint)}
    done = {key[0] for key in ctx.store.already_scored("record_scores", versions)}
    reasons = ctx.records.apply(
        lambda row: _text_skip_reason(ctx.settings, "misinfo", row["text"], row["lang"], 20),
        axis=1,
    )
    skip_note = _record_misinfo_skips(ctx, reasons)
    eligible = ctx.records.loc[reasons.isna() & ~ctx.records["id"].astype(str).isin(done)]
    if not len(eligible):
        return [
            {
                "stage": "misinfo",
                "table": "record_scores",
                "unchanged": len(ctx.records),
                "note": ctx.note("resumed: nothing to do" + skip_note),
            }
        ]

    scores = classifier.predict(eligible["text"].astype(str).tolist())
    frame = pd.DataFrame(
        {
            "record_id": eligible["id"].astype(str).to_numpy(),
            "source": eligible["source"].astype(str).to_numpy(),
            "misinfo_prob": scores,
            "model_versions": [versions] * len(eligible),
            "scored_at": utcnow_local(),
        }
    )
    if ctx.dry_run:
        return [
            {"stage": "misinfo", "table": "record_scores",
             "note": ctx.note(f"dry-run; would score {len(frame)} rows{skip_note}")}
        ]

    # A plain merge write: ScoredStore merges column-wise, so the aux pass's
    # toxicity and sentiment survive a frame that only carries misinfo_prob.
    result = ctx.store.write("record_scores", frame, known_keys=ctx.known_record_ids or None)
    return [
        {
            "stage": "misinfo",
            "table": "record_scores",
            **result,
            "note": ctx.note(
                f"scored {len(frame)}/{len(ctx.records)} records"
                + skip_note
                + (f" [{demo_note}]" if demo_note else "")
            ),
        }
    ]


def _text_skip_reason(
    settings: ModelingSettings, module: str, text: Any, lang: Any, min_chars: int
) -> str | None:
    """Why a text stage declines a record, in the order the aux scorers use.

    The codes are those of ``modeling.aux.base.SkipReason``, so a record skipped
    by this stage and one skipped by an aux scorer read the same in
    ``record_scores.skip_reasons``.
    """
    from modeling.aux.base import SkipReason

    if text is None or pd.isna(text) or not str(text).strip():
        return SkipReason.EMPTY_TEXT
    if len(str(text).strip()) < min_chars:
        return SkipReason.TEXT_TOO_SHORT
    if not settings.language_allowed(module, lang):
        return SkipReason.UNSUPPORTED_LANGUAGE
    return None


def _keep_misinfo_skips(frame: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Carry the ``misinfo:``, ``stance:`` and ``laya:`` entries on disk into a fresh aux frame.

    ``skip_reasons`` is one list per record, and a merge write replaces the list.
    The aux pass knows nothing about the misinfo, laya and stance stages, so a rerun of
    the aux stage would erase the reasons that those stages recorded.
    """
    if not len(existing) or "skip_reasons" not in existing.columns:
        return frame
    prior = {
        str(rid): [
            r
            for r in (str(x) for x in as_list(value))
            if r.startswith(("misinfo:", "stance:", "laya:"))
        ]
        for rid, value in zip(existing["record_id"], existing["skip_reasons"], strict=True)
    }
    out = frame.copy()
    out["skip_reasons"] = [
        [*(str(r) for r in as_list(reasons)), *prior.get(str(rid), [])]
        for rid, reasons in zip(out["record_id"], out["skip_reasons"], strict=True)
    ]
    return out


def _record_misinfo_skips(ctx: StageContext, reasons: pd.Series) -> str:
    """Write ``misinfo:<reason>`` into ``record_scores.skip_reasons``; return a note.

    The column is shared with the aux pass, and a merge write replaces a list
    rather than appending to it, so the aux entries already on disk are read
    back and kept. Entries this stage wrote on an earlier run are replaced.
    """
    skipped = ctx.records.loc[reasons.notna()]
    if not len(skipped):
        return ""
    counts = reasons.dropna().value_counts().to_dict()
    note = "; skipped " + ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items()))
    if ctx.dry_run:
        return note

    existing = ctx.store.read("record_scores")
    prior: dict[str, list[str]] = {}
    if len(existing) and "skip_reasons" in existing.columns:
        prior = {
            str(rid): [str(r) for r in as_list(value)]
            for rid, value in zip(existing["record_id"], existing["skip_reasons"], strict=True)
        }
    rows = []
    for row, reason in zip(skipped.itertuples(index=False), reasons.dropna(), strict=True):
        record_id = str(row.id)
        kept = [r for r in prior.get(record_id, []) if not r.startswith("misinfo:")]
        rows.append(
            {
                "record_id": record_id,
                "source": str(row.source),
                "skip_reasons": [*kept, f"misinfo:{reason}"],
                "scored_at": utcnow_local(),
            }
        )
    ctx.store.write("record_scores", pd.DataFrame(rows), known_keys=ctx.known_record_ids or None)
    return note


# ---------------------------------------------------------------------------
# laya: Hindi misinfo_prob
# ---------------------------------------------------------------------------
def _laya_stage_config(settings: ModelingSettings) -> dict[str, Any]:
    """The ``laya`` config with the Hindi directory filled in from ``LAYA_HI_CHECKPOINT``.

    The committed ``finetuned_checkpoints.hi`` is null because the path is a property of
    the host. A value in the file wins. A relative setting resolves against ``MODELS_DIR``.
    """
    import copy

    from modeling.config import module_config

    config = copy.deepcopy(module_config("laya"))
    tuned = dict(config.get("finetuned_checkpoints") or {})
    override = getattr(settings, "laya_hi_checkpoint", None)
    if not tuned.get("hi") and override:
        path = Path(str(override)).expanduser()
        tuned["hi"] = str(path if path.is_absolute() else settings.models_dir / path)
    config["finetuned_checkpoints"] = tuned
    return config


def _laya_refusals(
    scorer: Any, languages: Sequence[str]
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, Any]]]:
    """Per language, ``(reason, message)`` when the stage cannot score it, else a calibrator.

    A Laya checkpoint serves ``misinfo_prob`` only with a calibrator beside it, and only
    from the fine-tuned directory. The base checkpoint is never a substitute.
    """
    from modeling.text.laya_calibration import LayaCalibratorError, load_calibrator
    from modeling.text.laya_clf import LayaCheckpointMissing, LayaSkipReason

    availability = scorer.available()
    if not availability.ok:
        return {
            lang: (LayaSkipReason.MODEL_UNAVAILABLE, availability.reason) for lang in languages
        }, {}
    refused: dict[str, tuple[str, str]] = {}
    usable: dict[str, tuple[str, Any]] = {}
    for lang in languages:
        tuned = scorer.finetuned_path_for(lang)
        if not tuned:
            refused[lang] = (
                LayaSkipReason.CHECKPOINT_MISSING,
                f"no fine-tuned Laya checkpoint is set for lang={lang}. Set "
                "LAYA_HI_CHECKPOINT or laya.finetuned_checkpoints.hi. The base checkpoint "
                "is not a substitute.",
            )
            continue
        try:
            scorer.checkpoint_for(lang)
        except LayaCheckpointMissing as exc:
            refused[lang] = (LayaSkipReason.CHECKPOINT_MISSING, str(exc))
            continue
        try:
            loaded = load_calibrator(Path(tuned).expanduser())
        except LayaCalibratorError as exc:
            refused[lang] = (exc.reason, str(exc))
            continue
        usable[lang] = (f"{scorer.model_version_for(lang)}+{loaded.stamp}", loaded)
    return refused, usable


@stage("laya")
def _laya_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Calibrated Laya score of the title -> ``record_scores.misinfo_prob``, for Hindi.

    Scope is the ``scoring_languages`` of the ``laya`` module, Hindi only. The English slot
    belongs to the ``misinfo`` module, and a record outside the scope gets no value and no
    reason from this stage. A record that ``misinfo`` scored, or that holds any other
    non-null ``misinfo_prob``, is left as it is.

    The stage scores the record's title, read by ``modeling.titles.title_of`` as the
    comparison reads ``corpus_title``. The Hindi gate passed on titles. On Hindi full text the
    same model had a lift of 0.367 [0.280, 0.487], so full text is outside what the gate
    measured. A record with no title gets ``laya:no_title``, and ``text_too_short`` applies to
    the title.

    The value written is the calibrator's output, never the raw score. The calibrator comes
    from the fine-tuned checkpoint directory and was fitted on English benchmark labels, so
    its calibration on Hindi is not measured. When the package, the checkpoint or the
    calibrator is not usable, every in-scope record gets ``laya:<reason>``, a null
    ``misinfo_prob`` and no ``laya`` stamp, and the stage reports the refusal. The summary
    counts the raw values outside the fitted range and the distinct calibrated values. It
    warns on a collapse and does not refuse.

    A record scored under the current stamp is not scored again. A skipped record has its
    value cleared and its stamp retracted, so ``already_scored`` cannot return it.
    """
    from modeling.io import _as_version_dict
    from modeling.text.laya_calibration import MIN_DISTINCT_OUTPUTS
    from modeling.text.laya_clf import MIN_CHARS, LayaScorer, LayaSkipReason
    from modeling.titles import title_of

    config = _laya_stage_config(ctx.settings)
    languages = [str(lang).strip().lower() for lang in config.get("scoring_languages") or []]
    declared = {str(lang).lower() for lang in config.get("languages") or []}
    problem = None
    if not languages:
        problem = "laya.scoring_languages is empty"
    elif "en" in languages:
        problem = "laya.scoring_languages names en, and the English slot belongs to misinfo"
    elif not set(languages) <= declared:
        problem = f"laya.scoring_languages {languages} is not within laya.languages"
    if problem:
        log.error("laya stage refused: %s. Nothing was written.", problem)
        return [
            {
                "stage": "laya",
                "table": "record_scores",
                "refused": "scoring_languages_invalid",
                "note": ctx.note(f"REFUSED: {problem}; nothing written"),
            }
        ]

    langs = ctx.records["lang"].map(
        lambda value: None if value is None or value != value else str(value).lower()
    )
    in_scope = ctx.records.loc[langs.isin(languages)]
    if not len(in_scope):
        return [
            {
                "stage": "laya",
                "table": "record_scores",
                "note": ctx.note(f"no records in {languages}; nothing to do"),
            }
        ]

    existing = ctx.store.read("record_scores")
    prior_versions: dict[str, dict[str, str]] = {}
    prior_prob: dict[str, Any] = {}
    prior_reasons: dict[str, list[str]] = {}
    if len(existing):
        for row in existing.to_dict(orient="records"):
            rid = str(row["record_id"])
            prior_versions[rid] = _as_version_dict(row.get("model_versions"))
            value = row.get("misinfo_prob")
            prior_prob[rid] = None if value is None or value != value else value
            prior_reasons[rid] = [str(r) for r in as_list(row.get("skip_reasons"))]

    def laya_owned(rid: str) -> bool:
        return "laya" in prior_versions.get(rid, {})

    def foreign(rid: str) -> bool:
        """Does another module own this record's misinfo_prob?"""
        if "misinfo" in prior_versions.get(rid, {}):
            return True
        return prior_prob.get(rid) is not None and not laya_owned(rid)

    candidates = [r for r in in_scope.itertuples(index=False) if not foreign(str(r.id))]
    protected = [r for r in in_scope.itertuples(index=False) if foreign(str(r.id))]

    scorer = LayaScorer(settings=ctx.settings, _config=config)
    refused: dict[str, tuple[str, str]] = {}
    usable: dict[str, tuple[str, Any]] = {}
    if candidates:
        refused, usable = _laya_refusals(
            scorer, sorted({str(r.lang).lower() for r in candidates})
        )
        for lang, (reason, message) in refused.items():
            log.error("laya stage refused for lang=%s (laya:%s): %s", lang, reason, message)

    done: dict[str, set[str]] = {
        lang: {k[0] for k in ctx.store.already_scored("record_scores", {"laya": stamp})}
        for lang, (stamp, _) in usable.items()
    }

    if "raw" not in ctx.records.columns:
        log.warning("laya: the records have no `raw` column, so no record has a title")
    titles: dict[str, str] = {}
    reasons: dict[str, str | None] = {}
    pending = []
    for row in candidates:
        rid, lang = str(row.id), str(row.lang).lower()
        if lang in refused:
            reasons[rid] = refused[lang][0]
            continue
        title = title_of(getattr(row, "raw", None))
        if title is None:
            reasons[rid] = LayaSkipReason.NO_TITLE
            continue
        reason = _text_skip_reason(ctx.settings, "laya", title, lang, MIN_CHARS)
        if reason is not None:
            reasons[rid] = reason
        elif rid not in done.get(lang, set()):
            reasons[rid] = None
            titles[rid] = title
            pending.append(row)

    if ctx.dry_run:
        return [
            {
                "stage": "laya",
                "table": "record_scores",
                "note": ctx.note(f"dry-run; would score {len(pending)} rows"),
            }
        ]

    values: dict[str, float] = {}
    collapse: dict[str, dict[str, Any]] = {}
    if pending:
        frame_in = pd.DataFrame(
            {
                "id": [str(r.id) for r in pending],
                "text": [titles[str(r.id)] for r in pending],
                "lang": pd.Series([str(r.lang).lower() for r in pending], dtype=object),
            }
        )
        outcome = scorer.score(frame_in)
        by_lang: dict[str, list[tuple[str, float]]] = {}
        lang_of = dict(zip(frame_in["id"], frame_in["lang"], strict=True))
        for rid in frame_in["id"]:
            if rid in outcome.values:
                raw = float(outcome.values[rid])
                if raw == raw and 0.0 <= raw <= 1.0:
                    by_lang.setdefault(lang_of[rid], []).append((rid, raw))
                else:
                    reasons[rid] = LayaSkipReason.INVALID_SCORE
            else:
                reasons[rid] = outcome.skipped.get(rid) or LayaSkipReason.INFERENCE_ERROR
        for lang, pairs in by_lang.items():
            loaded = usable[lang][1]
            raws = np.array([raw for _, raw in pairs])
            calibrated = loaded.calibrator.transform(raws)
            for (rid, _), value in zip(pairs, calibrated, strict=True):
                values[rid] = float(value)
                reasons[rid] = None
            low, high = loaded.fit_range
            outside = int(np.sum((raws < low) | (raws > high)))
            distinct = int(len(np.unique(calibrated)))
            collapsed = distinct < MIN_DISTINCT_OUTPUTS or outside * 2 > len(pairs)
            collapse[lang] = {
                "scored": len(pairs),
                "out_of_range": outside,
                "distinct_calibrated": distinct,
                "collapse_warning": collapsed,
            }
            (log.warning if collapsed else log.info)(
                "laya lang=%s: %d scored, %d raw values outside the fitted range "
                "[%.4f, %.4f], %d distinct calibrated values%s",
                lang, len(pairs), outside, low, high, distinct,
                f". COLLAPSE WARNING: fewer than {MIN_DISTINCT_OUTPUTS} distinct values or more "
                "than half the raw values out of range; the written values carry little "
                "information"
                if collapsed
                else "",
            )

    def kept(rid: str) -> list[str]:
        return [r for r in prior_reasons.get(rid, []) if not r.startswith("laya:")]

    rows: list[dict[str, Any]] = []
    retract: dict[tuple[str, ...], set[str]] = {}
    now = utcnow_local()
    for row in candidates:
        rid = str(row.id)
        if rid not in reasons:
            continue
        entry: dict[str, Any] = {"record_id": rid, "source": str(row.source), "scored_at": now}
        if rid in values:
            entry.update(
                misinfo_prob=values[rid],
                skip_reasons=kept(rid),
                model_versions={"laya": usable[str(row.lang).lower()][0]},
            )
        else:
            entry.update(
                misinfo_prob=None,
                skip_reasons=[*kept(rid), f"laya:{reasons[rid]}"],
                model_versions={},
            )
            retract[(rid,)] = {"laya"}
        rows.append(entry)

    # A record that misinfo or another writer now owns must not keep a laya stamp.
    cleanup = []
    for row in protected:
        rid = str(row.id)
        if laya_owned(rid) or kept(rid) != prior_reasons.get(rid, []):
            cleanup.append(
                {
                    "record_id": rid,
                    "source": str(row.source),
                    "skip_reasons": kept(rid),
                    "model_versions": {},
                    "scored_at": now,
                }
            )
            retract[(rid,)] = {"laya"}

    counts = pd.Series([r for r in reasons.values() if r is not None], dtype=object)
    skip_counts = counts.value_counts().to_dict()
    skip_note = (
        "; skipped " + ", ".join(f"{n} {reason}" for reason, n in sorted(skip_counts.items()))
        if skip_counts
        else ""
    )
    refused_note = (
        "; REFUSED " + ", ".join(f"{lang}: laya:{code}" for lang, (code, _) in refused.items())
        if refused
        else ""
    )
    totals = {"written": 0, "updated": 0, "unchanged": 0}
    for part in (rows, cleanup):
        if part:
            result = ctx.store.write(
                "record_scores",
                pd.DataFrame(part),
                known_keys=ctx.known_record_ids or None,
                drop_model_versions=retract,
            )
            for key in totals:
                totals[key] += result[key]
    if not rows and not cleanup:
        return [
            {
                "stage": "laya",
                "table": "record_scores",
                "unchanged": len(in_scope),
                "note": ctx.note("resumed: nothing to do"),
            }
        ]
    summary: dict[str, Any] = {"stage": "laya", "table": "record_scores", **totals}
    if refused:
        summary["refused"] = {lang: code for lang, (code, _) in refused.items()}
    if collapse:
        summary["calibration"] = collapse
        skip_note += "; calibration " + ", ".join(
            f"{lang}: {c['out_of_range']} out of range, {c['distinct_calibrated']} distinct"
            + (" COLLAPSE WARNING" if c["collapse_warning"] else "")
            for lang, c in collapse.items()
        )
    summary["note"] = ctx.note(
        f"scored {len(values)}/{len(in_scope)} in-scope records; left {len(protected)} to other "
        f"modules" + skip_note + refused_note
    )
    return [summary]


@stage("stance")
def _stance_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Stance of each record toward the representative claim of its narrative.

    Without a registered checkpoint this is the null path: ``stance`` and
    ``stance_conf`` stay null, as the contract permits, because an untrained
    model producing confident labels is worse than an empty column.

    With one, every record is either scored or skipped with a ``stance:<reason>``
    entry in ``record_scores.skip_reasons``. The reasons are the aux codes
    (``empty_text``, ``text_too_short``, ``unsupported_language``) plus
    ``no_narrative`` (noise bucket or never clustered), ``no_claim_text``
    (the narrative's representative is not in this corpus slice),
    ``claim_unsupported_language`` (the claim is outside the stance language
    gate) and ``is_claim`` (the record is the claim itself). A record that is
    skipped now has any stance from an earlier run cleared, since the claim it
    was scored against may no longer exist.

    Every eligible record is rescored on each run. The score depends on the
    narrative's current claim as well as on the checkpoint, and the resume key
    covers only the checkpoint.
    """
    from modeling.config import module_config
    from modeling.registry import resolve
    from modeling.text.stance_clf import StanceClassifier

    version = str(module_config("stance").get("version"))
    checkpoint = resolve("stance", version, required=False)
    if checkpoint is None:
        return [
            {
                "stage": "stance",
                "table": "record_scores",
                "note": ctx.note(
                    "not trained; stance and stance_conf are written as null, as the "
                    "contract permits. See artifacts/model_cards/stance.md."
                ),
            }
        ]

    classifier = StanceClassifier(ctx.settings)
    if not classifier.load(checkpoint.path):
        return [
            {
                "stage": "stance",
                "table": "record_scores",
                "note": ctx.note(
                    f"checkpoint refused ({classifier.load_error}); stance and stance_conf "
                    "stay null. See the log for the cause."
                ),
            }
        ]
    if not len(ctx.records):
        return [{"stage": "stance", "table": "record_scores", "note": ctx.note("no records")}]

    demo_note = warn_if_demo_checkpoint("stance", checkpoint, ctx.demo)
    claims = _narrative_claims(ctx)
    reasons: list[str | None] = []
    pairs: list[tuple[str, str]] = []
    for row in ctx.records.itertuples(index=False):
        reason, claim = _stance_skip_reason(ctx, claims, row)
        reasons.append(reason)
        if claim is not None:
            pairs.append((claim, str(row.text)))

    skipped = pd.Series(reasons, dtype=object)
    counts = skipped.dropna().value_counts().to_dict()
    skip_note = (
        "; skipped " + ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items()))
        if counts
        else ""
    )
    if ctx.dry_run:
        return [
            {
                "stage": "stance",
                "table": "record_scores",
                "note": ctx.note(f"dry-run; would score {len(pairs)} rows{skip_note}"),
            }
        ]

    predictions = iter(classifier.predict(pairs))
    existing = ctx.store.read("record_scores")
    prior: dict[str, list[str]] = {}
    if len(existing) and "skip_reasons" in existing.columns:
        prior = {
            str(rid): [str(r) for r in as_list(value)]
            for rid, value in zip(existing["record_id"], existing["skip_reasons"], strict=True)
        }
    versions = {"stance": checkpoint_stamp("stance", version, checkpoint)}
    rows = []
    retract: dict[tuple[str, ...], set[str]] = {}
    for row, reason in zip(ctx.records.itertuples(index=False), reasons, strict=True):
        record_id = str(row.id)
        kept = [r for r in prior.get(record_id, []) if not r.startswith("stance:")]
        entry: dict[str, Any] = {"record_id": record_id, "source": str(row.source)}
        if reason is None:
            prediction = next(predictions)
            entry.update(
                stance=prediction.label,
                stance_conf=prediction.confidence,
                skip_reasons=kept,
                model_versions=versions,
            )
        else:
            entry.update(
                stance=None,
                stance_conf=None,
                skip_reasons=[*kept, f"stance:{reason}"],
                model_versions={},
            )
            # The merge unions version maps, so a stamp from an earlier scored
            # run would survive and `already_scored` would still return this row.
            retract[(record_id,)] = {"stance"}
        entry["scored_at"] = utcnow_local()
        rows.append(entry)

    result = ctx.store.write(
        "record_scores",
        pd.DataFrame(rows),
        known_keys=ctx.known_record_ids or None,
        drop_model_versions=retract,
    )
    return [
        {
            "stage": "stance",
            "table": "record_scores",
            **result,
            "note": ctx.note(
                f"scored {len(pairs)}/{len(ctx.records)} records"
                + skip_note
                + (f" [{demo_note}]" if demo_note else "")
            ),
        }
    ]


@dataclass
class _NarrativeClaims:
    """Which claim each record is scored against, and where that fails."""

    claim_of: dict[str, str] = field(default_factory=dict)
    claim_lang: dict[str, Any] = field(default_factory=dict)
    claim_record: dict[str, str] = field(default_factory=dict)
    narrative_of: dict[str, str] = field(default_factory=dict)
    # Narratives whose representatives all had text, but none long enough.
    too_short: set[str] = field(default_factory=set)


_STANCE_MIN_CHARS = 20


def _narrative_claims(ctx: StageContext) -> _NarrativeClaims:
    """Pick each record's narrative and each narrative's representative claim.

    A record in several narratives takes the one with the highest membership
    probability. The claim is the narrative's best-ranked representative whose
    text is present in ``ctx.records`` and at least ``_STANCE_MIN_CHARS`` long,
    the same floor as for the record text: a premise such as "ok" says nothing.
    """
    out = _NarrativeClaims()
    membership = ctx.store.read("narrative_membership")
    if not len(membership):
        return out

    texts = ctx.records.set_index(ctx.records["id"].astype(str))
    ranked = membership.assign(
        record_id=membership["record_id"].astype(str),
        narrative_id=membership["narrative_id"].astype(str),
        membership_prob=membership["membership_prob"].fillna(0.0),
    ).sort_values(["membership_prob", "record_id"], ascending=[False, True], kind="stable")

    out.narrative_of = (
        ranked.drop_duplicates("record_id").set_index("record_id")["narrative_id"].to_dict()
    )
    representatives = ranked.loc[ranked["is_representative"].fillna(False).astype(bool)]
    for narrative_id, group in representatives.groupby("narrative_id", sort=False):
        for record_id in group["record_id"]:
            if record_id not in texts.index:
                continue
            text = texts.loc[record_id, "text"]
            if text is None or pd.isna(text) or not str(text).strip():
                continue
            if len(str(text).strip()) < _STANCE_MIN_CHARS:
                out.too_short.add(narrative_id)
                continue
            out.too_short.discard(narrative_id)
            out.claim_of[narrative_id] = str(text)
            out.claim_lang[narrative_id] = texts.loc[record_id, "lang"]
            out.claim_record[narrative_id] = record_id
            break
    return out


def _stance_skip_reason(
    ctx: StageContext, claims: _NarrativeClaims, row: Any
) -> tuple[str | None, str | None]:
    """``(reason, None)`` when a record gets no stance, else ``(None, claim_text)``."""
    from modeling.aux.base import SkipReason

    record_id = str(row.id)
    reason = _text_skip_reason(ctx.settings, "stance", row.text, row.lang, _STANCE_MIN_CHARS)
    if reason is not None:
        return reason, None
    narrative_id = claims.narrative_of.get(record_id)
    if narrative_id is None:
        return "no_narrative", None
    if narrative_id not in claims.claim_of:
        if narrative_id in claims.too_short:
            return "claim_text_too_short", None
        return "no_claim_text", None
    if claims.claim_record[narrative_id] == record_id:
        return "is_claim", None
    if not ctx.settings.language_allowed("stance", claims.claim_lang[narrative_id]):
        return f"claim_{SkipReason.UNSUPPORTED_LANGUAGE}", None
    return None, claims.claim_of[narrative_id]


@stage("summarize")
def _summarize_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """LLM narrative labels and summaries, with the centroid fallback."""
    from modeling.config import module_config
    from modeling.text.summarize import NarrativeSummarizer

    narratives = ctx.store.read("narratives")
    if not len(narratives):
        return [{"stage": "summarize", "note": ctx.note("no narratives; run the cluster stage")}]

    membership = ctx.store.read("narrative_membership")
    representatives = (
        membership.loc[membership["is_representative"].fillna(False)]
        .groupby("narrative_id")["record_id"]
        .apply(list)
        .to_dict()
        if len(membership)
        else {}
    )
    texts = (
        ctx.records.set_index("id")["text"].astype(str).to_dict()
        if "id" in ctx.records
        else {}
    )

    payload = [
        {
            "narrative_id": row["narrative_id"],
            "centroid": row.get("centroid"),
            "representative_ids": representatives.get(row["narrative_id"], []),
        }
        for row in narratives.to_dict("records")
    ]
    summarizer = NarrativeSummarizer(ctx.settings)
    run = summarizer.summarize(payload, texts)

    # Only the columns this stage owns. ScoredStore merges column-wise, so
    # everything the cluster stage computed survives untouched and the two
    # stages stop overwriting each other on every run.
    updated = narratives[["narrative_id"]].copy()
    updated["label"] = updated["narrative_id"].map(
        lambda n: run.results[n].label if n in run.results else None
    )
    updated["label_source"] = updated["narrative_id"].map(
        lambda n: run.results[n].label_source if n in run.results else None
    )
    updated["summary"] = updated["narrative_id"].map(
        lambda n: run.results[n].summary if n in run.results else None
    )
    versions = {
        "summarize": str(module_config("summarize").get("version")),
        "llm_model": str(module_config("summarize").get("llm_model")),
    }
    updated["model_versions"] = [versions] * len(updated)

    note = run.cost_note(summarizer.client.model)
    if not summarizer.client.available:
        note = "no API key; centroid labels only. " + note
    if ctx.dry_run:
        return [{"stage": "summarize", "table": "narratives", "note": ctx.note("dry-run; " + note)}]

    result = ctx.store.write("narratives", updated)
    return [{"stage": "summarize", "table": "narratives", **result, "note": ctx.note(note)}]


@stage("coordination")
def _coordination_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Co-behaviour graph, communities and the null-model comparison."""
    from modeling.accounts.coordination import CoordinationDetector

    detector = CoordinationDetector(ctx.settings)
    result = detector.detect(ctx.records)
    # Set before any early return: a run with no edges still measured every
    # account it saw, and the accounts stage must score those as 0.0, not skip them.
    ctx.coordination = result
    if not result.edges:
        return [
            {
                "stage": "coordination",
                "table": "coordination_edges",
                "note": ctx.note(
                    f"no edges above the weight floor over {result.n_records} eligible records"
                ),
            }
        ]

    now = utcnow_local()
    edges = pd.DataFrame([{**edge, "generated_at": now} for edge in result.edges])
    note = (
        f"{len(edges)} edges, {len(result.community_sizes)} communities, "
        f"modularity {result.modularity:.3f} vs null "
        f"{result.null_modularity:.3f}±{result.null_std:.3f} "
        f"({'exceeds' if result.exceeds_null else 'DOES NOT EXCEED'} the null)"
    )
    if ctx.dry_run:
        return [
            {"stage": "coordination", "table": "coordination_edges",
             "note": ctx.note("dry-run; " + note)}
        ]

    written = ctx.store.write("coordination_edges", edges, merge=False)
    ctx.store.update_manifest(
        table="coordination_edges",
        rows=len(edges),
        model_versions={"coordination": detector.version},
        extra={"is_demo": ctx.demo, **result.summary()},
    )
    return [
        {"stage": "coordination", "table": "coordination_edges", **written, "note": ctx.note(note)}
    ]


@stage("accounts")
def _accounts_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Assemble author_scores from every per-author signal available."""
    from modeling.accounts.bot_clf import (
        BotModel,
        extract_weak_bot_labels,
        recalibrate_on_domain,
        shap_contributions,
    )
    from modeling.accounts.features import available_tiers, build_features
    from modeling.config import module_config
    from modeling.registry import resolve

    if not len(ctx.authors):
        return [{"stage": "accounts", "note": ctx.note("no author roll-ups in the corpus")}]

    tiers = available_tiers(ctx.records, ctx.authors)
    features = build_features(ctx.records, ctx.authors, tiers=tiers)
    versions: dict[str, str] = {}
    reasons: dict[str, list[str]] = {a: [] for a in features.account_ids}

    # --- bot probability ---------------------------------------------------
    bot_probs: dict[str, float] = {}
    top_features: dict[str, list[dict[str, Any]]] = {}
    bot_version = str(module_config("bot").get("version"))
    checkpoint = resolve("bot", bot_version, required=False)
    warn_if_demo_checkpoint("bot", checkpoint, ctx.demo)
    model = BotModel.load(checkpoint.path) if checkpoint else None
    if model is None:
        for account in features.account_ids:
            reasons[account].append("bot:model_unavailable")
    else:
        absent = [name for name in model.feature_names if name not in features.names]
        if absent:
            # The corpus cannot even compute these columns.
            log.warning(
                "bot model needs feature(s) %s that this corpus cannot compute; bot_prob "
                "stays null. This is the feature-intersection discipline doing its job.",
                absent,
            )
            for account in features.account_ids:
                reasons[account].append("bot:feature_intersection_empty")
        else:
            # A column existing is not the same as a column carrying data.
            #
            # build_features emits every declared feature, filling what the
            # platform cannot supply with 0 and setting a paired `*_is_missing`
            # indicator. So the name-level check above passes for the whole
            # corpus while Reddit, YouTube, news and GDELT actually supply
            # *nothing*: no followers, no account age. Scored anyway, the model
            # saw 0 followers / 0 following / 0 days old -- a region of feature
            # space absent from its Twitter training data, where a tree ensemble
            # simply falls into whichever leaf its splits happen to route to.
            #
            # Measured on this corpus: every one of 2021 accounts came back
            # between 0.938 and 0.991. A constant 0.99 "bot" for every human on
            # Reddit is exactly the confident garbage a null exists to prevent,
            # and it is worse than no score because it looks like a finding.
            #
            # So the guard is per-account and about data: an account is scored
            # only when the features the model actually relies on are present
            # for *that account*.
            supplied = _features_supplied(features, model.feature_names)
            n_supplied = int(supplied.sum())
            if n_supplied == 0:
                log.warning(
                    "no account has the features the bot model needs (%s); bot_prob "
                    "stays null for all %d accounts",
                    ", ".join(model.feature_names[:3]) + "...",
                    len(features.account_ids),
                )
            for account, ok in zip(features.account_ids, supplied, strict=True):
                if not ok:
                    reasons[account].append("bot:features_not_supplied")

            if n_supplied:
                log.info(
                    "bot: scoring %d/%d accounts that actually supply the model's "
                    "features; the rest get null with a reason code",
                    n_supplied,
                    len(features.account_ids),
                )
                subset = features.subset(model.feature_names)
                matrix = subset.matrix[supplied]
                probabilities = model.predict_proba(matrix)
                contributions = shap_contributions(
                    model.estimator, matrix, model.feature_names,
                    int(module_config("bot").get("shap_top_k", 5)),
                )
                scored_ids = [
                    account
                    for account, ok in zip(features.account_ids, supplied, strict=True)
                    if ok
                ]

                # Recalibrate against the platform's own self-declared bot flag.
                #
                # The benchmark calibration does not survive the domain shift:
                # measured here, every scored account landed above the operating
                # threshold while only ~15% declare themselves bots. The ranking
                # transferred, the probabilities did not, and bot_prob is
                # contractually a calibrated probability Phase 4 multiplies into
                # a fused score. The flag is used for calibration only -- never
                # as a feature, because an undeclared bot is precisely the one
                # that omits it.
                weak = extract_weak_bot_labels(ctx.authors)
                aligned = weak.reindex(scored_ids).dropna()
                if len(aligned):
                    index = {account: i for i, account in enumerate(scored_ids)}
                    positions = [index[a] for a in aligned.index]
                    calibrator, calibration_report = recalibrate_on_domain(
                        probabilities[positions], aligned.to_numpy()
                    )
                    if calibrator is not None:
                        probabilities = calibrator.transform(probabilities)
                        versions["bot_domain_calibration"] = (
                            f"{calibration_report['method']}-mastodon-selfdeclared"
                        )
                    ctx.store.update_manifest(
                        table="author_scores",
                        rows=len(scored_ids),
                        model_versions={"bot": bot_version},
                        extra={"domain_recalibration": calibration_report},
                    )

                for account, probability, contribution in zip(
                    scored_ids, probabilities, contributions, strict=True
                ):
                    bot_probs[account] = float(probability)
                    top_features[account] = contribution
                versions["bot"] = checkpoint_stamp("bot", bot_version, checkpoint)

    # --- coordination ------------------------------------------------------
    coordination = getattr(ctx, "coordination", None)
    if coordination is None:
        for account in features.account_ids:
            reasons[account].append("coordination:not_run")
    else:
        versions["coordination"] = str(module_config("coordination").get("version"))
        for account in features.account_ids:
            if coordination.score_for(account) is None:
                reasons[account].append("coordination:not_in_detector_input")

    # --- aggregates from record_scores ------------------------------------
    scores = ctx.store.read("record_scores")
    aggregates = _author_aggregates(ctx.records, scores)
    membership = ctx.store.read("narrative_membership")
    narratives_touched = _narratives_touched(ctx.records, membership)

    now = utcnow_local()
    source_of = ctx.authors.set_index("author_id")["source"].astype(str).to_dict()
    rows = []
    for account in features.account_ids:
        aggregate = aggregates.get(account, {})
        rows.append(
            {
                "author_id": account,
                "source": source_of.get(account, account.split(":", 1)[0]),
                "bot_prob": bot_probs.get(account),
                "bot_top_features": top_features.get(account),
                "coordination_score": (
                    coordination.score_for(account) if coordination else None
                ),
                "community_id": (
                    coordination.communities.get(account) if coordination else None
                ),
                "community_size": (
                    coordination.community_sizes.get(coordination.communities.get(account))
                    if coordination and account in coordination.communities
                    else None
                ),
                "anomalous": aggregate.get("anomaly_mean"),
                "toxicity_mean": aggregate.get("toxicity_mean"),
                "dominant_sentiment": aggregate.get("dominant_sentiment"),
                "dominant_emotion": aggregate.get("dominant_emotion"),
                "narratives_touched": narratives_touched.get(account, []),
                "skip_reasons": reasons[account],
                "model_versions": versions,
                "scored_at": now,
            }
        )

    frame = pd.DataFrame(rows)
    note = f"{len(frame)} authors, tiers={','.join(tiers)}"
    if not bot_probs:
        note += "; bot_prob null (see skip_reasons)"
    if ctx.dry_run:
        return [
            {"stage": "accounts", "table": "author_scores",
             "note": ctx.note("dry-run; " + note)}
        ]

    written = ctx.store.write(
        "author_scores", frame, known_keys=ctx.known_author_ids or None, merge=False
    )
    ctx.store.update_manifest(
        table="author_scores", rows=len(frame), model_versions=versions,
        extra={"is_demo": ctx.demo, "tiers": tiers},
    )
    return [{"stage": "accounts", "table": "author_scores", **written, "note": ctx.note(note)}]


# ---------------------------------------------------------------------------
# helpers shared by the stages
# ---------------------------------------------------------------------------
def _features_supplied(features, feature_names: Sequence[str]):
    """Which accounts actually carry the features a model depends on.

    A feature is "not supplied" for an account when its paired
    ``<feature>_is_missing`` indicator is set. The indicators themselves are
    exempt -- they are metadata about the others, and requiring them to be
    "present" would reject every row.

    Returns a boolean mask over ``features.account_ids``. Conservative on
    purpose: an account is scored only when *every* substantive feature the
    model uses is real for it. A tree ensemble handed a half-imputed row still
    returns a confident number, and there is nothing behind it.
    """
    import numpy as np

    frame = features.as_frame()
    mask = np.ones(len(frame), dtype=bool)
    for name in feature_names:
        if name.endswith("_is_missing"):
            continue
        indicator = f"{name}_is_missing"
        if indicator in frame.columns:
            mask &= frame[indicator].to_numpy() < 0.5
    return mask


def _author_aggregates(records: pd.DataFrame, scores: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-author means and modes over record_scores.

    Means are computed over the subset where the metric exists, never over an
    imputed zero -- the same rule Phase 1 applies to engagement.
    """
    if not len(scores) or "id" not in records.columns:
        return {}
    joined = records[["id", "author_id"]].merge(
        scores, left_on="id", right_on="record_id", how="inner"
    )
    if not len(joined):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for author_id, group in joined.groupby("author_id"):
        entry: dict[str, Any] = {}
        for column, name in (("toxicity", "toxicity_mean"), ("anomaly_score", "anomaly_mean")):
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                entry[name] = float(values.mean()) if len(values) else None
        if "sentiment" in group.columns:
            modes = group["sentiment"].dropna()
            entry["dominant_sentiment"] = str(modes.mode().iloc[0]) if len(modes) else None
        if "emotion" in group.columns:
            entry["dominant_emotion"] = _dominant_emotion(group["emotion"])
        out[str(author_id)] = entry
    return out


def _dominant_emotion(column: pd.Series) -> str | None:
    totals: dict[str, float] = {}
    for value in column.dropna():
        if not isinstance(value, dict):
            continue
        for name, score in value.items():
            totals[name] = totals.get(name, 0.0) + float(score or 0.0)
    if not totals:
        return None
    return max(totals, key=totals.__getitem__)


def _narratives_touched(records: pd.DataFrame, membership: pd.DataFrame) -> dict[str, list[str]]:
    if not len(membership) or "id" not in records.columns:
        return {}
    joined = records[["id", "author_id"]].merge(
        membership, left_on="id", right_on="record_id", how="inner"
    )
    if not len(joined):
        return {}
    return (
        joined.groupby("author_id")["narrative_id"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )


def checkpoint_stamp(module: str, version: str, checkpoint) -> str:
    """The version string a scored row carries: ``<version>+<sha8>``.

    The configured version alone is not a staleness signal. Retraining a model
    writes new weights under the same ``v0.1.0``, so every previously-scored row
    still claims the current version, ``already_scored`` skips it, and the
    scorer reports "resumed: nothing to do" while the old scores stay on disk.
    That is silent and total: a whole corpus keeps predictions from a model that
    no longer exists.

    Appending the checkpoint's content hash makes the identity follow the
    *weights* rather than a hand-maintained string, so a retrain invalidates its
    own scores with no discipline required from anyone.
    """
    if checkpoint is None or not getattr(checkpoint, "sha256", None):
        return version
    return f"{version}+{checkpoint.sha256[:8]}"


def warn_if_demo_checkpoint(module: str, checkpoint, demo_run: bool) -> str | None:
    """Loudly refuse to let a fixture-trained model masquerade as a real one.

    ``registry.json`` records ``is_demo`` for anything trained on the committed
    fixtures. Scoring a real corpus with one produces numbers that look exactly
    like results, so this is a warning the log cannot swallow.
    """
    if checkpoint is None or demo_run:
        return None
    from modeling.registry import metadata

    if metadata(module, checkpoint.version).get("is_demo"):
        note = (
            f"{module} checkpoint {checkpoint.version} was trained on the DEMO FIXTURE "
            "-- these scores are plumbing evidence, not results"
        )
        log.warning(
            "%s. Train on real benchmarks and re-score before citing anything from "
            "this column.",
            note,
        )
        return note
    return None


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
    store: ScoredStore | None = None,
    embed_model: str | None = None,
) -> list[dict[str, Any]]:
    """Run the named stages over the corpus and return a per-stage summary.

    ``store`` and ``embed_model`` exist for the embedder comparison, which runs
    the cluster stage with another model into a store of its own.
    """
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
        store=store or _store_for(demo, settings),
        embed_model=embed_model,
        records=records,
        authors=authors,
        demo=demo,
        dry_run=dry_run,
        known_record_ids=set(records["id"].astype(str)) if "id" in records else set(),
        # "Exists in Phase 1" means "appears in the corpus", which is the union
        # of the author roll-up and the author_ids on the records themselves.
        # Phase 1 writes roll-ups only for sources that expose a real account
        # (Mastodon, Reddit, YouTube); GDELT and news records carry an author_id
        # derived from the outlet domain with no roll-up behind it. Taking only
        # the roll-up would make 115 legitimate GDELT authors look like orphans
        # and abort the whole stage.
        known_author_ids=(
            (set(authors["author_id"].astype(str)) if len(authors) else set())
            | (set(records["author_id"].astype(str)) if "author_id" in records else set())
        ),
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
