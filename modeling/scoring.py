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
    #: Set by the coordination stage and consumed by the accounts stage, which
    #: needs community_id and coordination_score. The dependency is why
    #: configs/scoring.yaml orders coordination before accounts.
    coordination: Any = None

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

    versions = {"misinfo": version}
    done = {key[0] for key in ctx.store.already_scored("record_scores", versions)}
    eligible = ctx.records.loc[
        ctx.records["lang"].map(ctx.settings.language_allowed)
        & ctx.records["text"].fillna("").astype(str).str.len().ge(20)
        & ~ctx.records["id"].astype(str).isin(done)
    ]
    if not len(eligible):
        return [
            {
                "stage": "misinfo",
                "table": "record_scores",
                "unchanged": len(ctx.records),
                "note": ctx.note("resumed: nothing to do"),
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
             "note": ctx.note(f"dry-run; would score {len(frame)} rows")}
        ]

    # A plain merge write: ScoredStore merges column-wise, so the aux pass's
    # toxicity and sentiment survive a frame that only carries misinfo_prob.
    result = ctx.store.write("record_scores", frame, known_keys=ctx.known_record_ids or None)
    return [
        {
            "stage": "misinfo",
            "table": "record_scores",
            **result,
            "note": ctx.note(f"scored {len(frame)}/{len(ctx.records)} records"),
        }
    ]


@stage("stance")
def _stance_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Stance toward a narrative's representative claim.

    Deliberately a null path until a checkpoint exists. An untrained model
    producing confident stance labels would be worse than an empty column, and
    the contract explicitly allows null.
    """
    from modeling.config import module_config
    from modeling.registry import resolve

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
    return [
        {
            "stage": "stance",
            "table": "record_scores",
            "note": ctx.note(
                "checkpoint present but the scoring path is not wired; see model card"
            ),
        }
    ]


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
    # Stash for the accounts stage, which folds community_id and
    # coordination_score into author_scores.
    ctx.coordination = result
    return [
        {"stage": "coordination", "table": "coordination_edges", **written, "note": ctx.note(note)}
    ]


@stage("accounts")
def _accounts_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Assemble author_scores from every per-author signal available."""
    from modeling.accounts.bot_clf import BotModel, shap_contributions
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
                for account, probability, contribution in zip(
                    scored_ids, probabilities, contributions, strict=True
                ):
                    bot_probs[account] = float(probability)
                    top_features[account] = contribution
                versions["bot"] = bot_version

    # --- coordination ------------------------------------------------------
    coordination = getattr(ctx, "coordination", None)
    if coordination is None:
        for account in features.account_ids:
            reasons[account].append("coordination:not_run")

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
                    coordination.scores.get(account) if coordination else None
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


@stage("media")
def _media_stage(ctx: StageContext) -> list[dict[str, Any]]:
    """Deepfake scoring over records carrying media."""
    from modeling.config import module_config
    from modeling.media.deepfake_clf import DeepfakeScorer
    from modeling.registry import resolve

    if "media_urls" not in ctx.records.columns:
        return [{"stage": "media", "note": ctx.note("corpus carries no media_urls column")}]
    from modeling.io import as_list

    with_media = ctx.records.loc[ctx.records["media_urls"].map(lambda v: bool(as_list(v)))]
    if not len(with_media):
        return [{"stage": "media", "note": ctx.note("no records carry media")}]

    version = str(module_config("deepfake").get("version"))
    checkpoint = resolve("deepfake", version, required=False)
    scorer = DeepfakeScorer(ctx.settings)
    rows = scorer.score_records(with_media, checkpoint.path if checkpoint else None)
    if not rows:
        return [
            {
                "stage": "media",
                "table": "media_scores",
                "note": ctx.note(
                    f"{len(with_media)} record(s) carry media but none could be scored "
                    "(no checkpoint, or remote fetching disabled in configs/scoring.yaml)"
                ),
            }
        ]

    frame = pd.DataFrame(rows)
    if ctx.dry_run:
        return [{"stage": "media", "table": "media_scores",
                 "note": ctx.note(f"dry-run; would write {len(frame)} rows")}]
    written = ctx.store.write(
        "media_scores", frame, known_keys=ctx.known_record_ids or None, merge=False
    )
    return [{"stage": "media", "table": "media_scores", **written, "note": ctx.note("")}]


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
