"""Narrative clustering: HDBSCAN over embeddings, with stable ids across runs.

**Why HDBSCAN and not k-means.** We do not know how many narratives are in a
corpus, and most posts belong to none of them. k-means demands a k and assigns
every point, so it manufactures narratives out of background chatter. HDBSCAN
discovers the count and has a genuine noise label, which is the honest answer
for most of a corpus.

**Why dedupe before clustering.** One viral repost swarm is hundreds of
near-identical vectors. Left alone it forms its own dense cluster and dominates
every quality metric. Near-duplicates (simhash Hamming <= 3) are collapsed to a
single representative *for the clustering decision only*; the full member list
is preserved and restored afterwards, so ``size`` and ``author_count`` still
reflect reality.

**Why ids are carried forward.** The UI shows "Generated 48 days ago / Update
Now", narrative ids are user-visible, and a user may have renamed one. Minting
fresh ids on every run would orphan all of that. On rerun, new clusters are
matched to previous ones by centroid cosine similarity above a threshold and the
id is carried; only genuinely new clusters get a new id. Splits, merges and
deaths are logged rather than papered over.

**How this is evaluated.** There are no gold narrative labels, so there is no
F1. What is reported instead: silhouette on a sample, noise ratio, cluster size
distribution, and a hand-audit of 20 clusters rated coherent / mixed / junk.
That audit table is the honest evaluation of unsupervised output, and it lives
in ``artifacts/error_analysis/cluster.md``.

Implementation note: this uses ``sklearn.cluster.HDBSCAN`` rather than the
standalone ``hdbscan`` package. Same algorithm, one fewer build-fragile
dependency, and scikit-learn is already required. The standalone package's
``approximate_predict`` is not needed here because cross-run identity is handled
by centroid matching, which is more robust for this use than soft-assigning new
points to an old model.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config
from modeling.text.embed import EmbeddingResult, cosine_similarity, l2_normalize

log = logging.getLogger(__name__)

NOISE_LABEL = -1


@dataclass
class Narrative:
    """One discovered narrative, with everything the contract needs."""

    narrative_id: str
    member_ids: list[str]
    centroid: np.ndarray
    coherence: float
    size: int
    author_count: int
    first_seen: datetime | None
    last_seen: datetime | None
    platforms: list[str]
    top_domains: list[str]
    top_hashtags: list[str]
    velocity: float
    severity: float | None
    representative_ids: list[str] = field(default_factory=list)
    membership: dict[str, float] = field(default_factory=dict)
    #: "carried" | "new" -- whether this id survived from the previous run.
    id_origin: str = "new"

    def as_row(self, model_versions: dict[str, str], generated_at: datetime) -> dict[str, Any]:
        return {
            "narrative_id": self.narrative_id,
            "label": None,  # filled by the summarize stage
            "label_source": None,
            "summary": None,
            "size": int(self.size),
            "author_count": int(self.author_count),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "platforms": self.platforms,
            "top_domains": self.top_domains,
            "top_hashtags": self.top_hashtags,
            "centroid": [float(v) for v in self.centroid],
            "velocity": float(self.velocity),
            "severity": None if self.severity is None else float(self.severity),
            "coherence": float(self.coherence),
            "model_versions": model_versions,
            "generated_at": generated_at,
        }


@dataclass
class ClusteringResult:
    """The full outcome of one clustering run, including its own diagnostics."""

    narratives: list[Narrative]
    noise_ids: list[str]
    n_input: int
    n_after_dedupe: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    transitions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def noise_ratio(self) -> float:
        return len(self.noise_ids) / self.n_input if self.n_input else 0.0

    def summary(self) -> dict[str, Any]:
        sizes = [n.size for n in self.narratives]
        return {
            "narratives": len(self.narratives),
            "records_in": self.n_input,
            "records_after_dedupe": self.n_after_dedupe,
            "noise_ratio": round(self.noise_ratio, 3),
            "size_min": min(sizes) if sizes else 0,
            "size_median": int(np.median(sizes)) if sizes else 0,
            "size_max": max(sizes) if sizes else 0,
            "coherence_mean": (
                round(float(np.mean([n.coherence for n in self.narratives])), 3) if sizes else None
            ),
            "ids_carried": sum(1 for n in self.narratives if n.id_origin == "carried"),
            "ids_new": sum(1 for n in self.narratives if n.id_origin == "new"),
            **self.diagnostics,
        }


class NarrativeClusterer:
    module = "cluster"

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.min_cluster_size = int(self.config.get("min_cluster_size", 15))
        self.min_samples = int(self.config.get("min_samples", 5))
        self.selection_method = str(self.config.get("cluster_selection_method", "eom"))
        self.selection_epsilon = float(self.config.get("cluster_selection_epsilon", 0.0))
        self.simhash_hamming = int(self.config.get("simhash_hamming", 3))
        self.stability_threshold = float(self.config.get("stability_threshold", 0.85))
        self.n_representatives = int(self.config.get("representatives_per_cluster", 5))
        self.severity_percentile = float(self.config.get("severity_percentile", 75))
        self.severity_weighted = bool(self.config.get("severity_engagement_weighted", True))

    # --- dedupe ----------------------------------------------------------
    def collapse_duplicates(
        self, records: pd.DataFrame, embeddings: EmbeddingResult
    ) -> tuple[list[int], dict[int, list[str]]]:
        """Pick one representative row per near-duplicate group.

        Returns the representative positions into ``embeddings.vectors`` and a
        map from representative position to the full member id list. Bucketed by
        simhash prefix rather than compared pairwise: this sits on the hot path
        and an all-pairs scan is O(n^2).
        """
        by_id = {rid: i for i, rid in enumerate(embeddings.record_ids)}
        simhash = {}
        if "simhash" in records.columns:
            for row in records.itertuples(index=False):
                value = getattr(row, "simhash", None)
                if value is not None and not pd.isna(value):
                    simhash[str(row.id)] = int(value)

        representatives: list[int] = []
        members: dict[int, list[str]] = {}
        buckets: dict[int, list[int]] = {}  # prefix -> representative positions

        for record_id in embeddings.record_ids:
            position = by_id[record_id]
            value = simhash.get(record_id)
            if value is None:
                representatives.append(position)
                members[position] = [record_id]
                continue
            prefix = value >> 48
            match = None
            for candidate in buckets.get(prefix, []):
                other = simhash.get(embeddings.record_ids[candidate])
                if other is None:
                    continue
                if bin(value ^ other).count("1") <= self.simhash_hamming:
                    match = candidate
                    break
            if match is None:
                representatives.append(position)
                members[position] = [record_id]
                buckets.setdefault(prefix, []).append(position)
            else:
                members[match].append(record_id)

        collapsed = len(embeddings.record_ids) - len(representatives)
        if collapsed:
            log.info(
                "collapsed %d near-duplicate records into %d representatives before clustering "
                "(full member lists preserved)",
                collapsed,
                len(representatives),
            )
        return representatives, members

    # --- clustering ------------------------------------------------------
    def fit(
        self,
        records: pd.DataFrame,
        embeddings: EmbeddingResult,
        *,
        record_scores: pd.DataFrame | None = None,
        previous: pd.DataFrame | None = None,
    ) -> ClusteringResult:
        """Cluster, then build the contract rows."""
        if not len(embeddings):
            return ClusteringResult([], [], 0, 0, {"note": "no embeddings"})

        representatives, members = self.collapse_duplicates(records, embeddings)
        matrix = embeddings.vectors[representatives]

        from sklearn.cluster import HDBSCAN

        # Euclidean on L2-normalized vectors is monotonically equivalent to
        # cosine distance, which is what the embedding model was trained for.
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method=self.selection_method,
            cluster_selection_epsilon=self.selection_epsilon,
        )
        labels = clusterer.fit_predict(matrix)
        probabilities = getattr(clusterer, "probabilities_", np.ones(len(labels)))

        record_lookup = records.set_index("id", drop=False)
        narratives: list[Narrative] = []
        noise_ids: list[str] = []

        for label in sorted(set(labels)):
            member_positions = [i for i, value in enumerate(labels) if value == label]
            member_ids: list[str] = []
            for position in member_positions:
                member_ids.extend(members[representatives[position]])
            if label == NOISE_LABEL:
                noise_ids = member_ids
                continue
            vectors = matrix[member_positions]
            narratives.append(
                self._build_narrative(
                    member_ids=member_ids,
                    representative_positions=[representatives[p] for p in member_positions],
                    vectors=vectors,
                    probabilities=[float(probabilities[p]) for p in member_positions],
                    embeddings=embeddings,
                    record_lookup=record_lookup,
                    record_scores=record_scores,
                )
            )

        narratives.sort(key=lambda n: n.size, reverse=True)
        transitions = self._carry_ids(narratives, previous)

        diagnostics = self._diagnostics(matrix, labels)
        result = ClusteringResult(
            narratives=narratives,
            noise_ids=noise_ids,
            n_input=len(embeddings),
            n_after_dedupe=len(representatives),
            diagnostics=diagnostics,
            transitions=transitions,
        )
        log.info("clustering: %s", json.dumps(result.summary(), default=str))
        return result

    # --- per-cluster metrics ---------------------------------------------
    def _build_narrative(
        self,
        *,
        member_ids: list[str],
        representative_positions: list[int],
        vectors: np.ndarray,
        probabilities: list[float],
        embeddings: EmbeddingResult,
        record_lookup: pd.DataFrame,
        record_scores: pd.DataFrame | None,
    ) -> Narrative:
        rows = record_lookup.reindex([m for m in member_ids if m in record_lookup.index])
        centroid = l2_normalize(vectors.mean(axis=0, keepdims=True))[0]

        timestamps = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce").dropna()
        platforms = sorted(rows["source"].dropna().astype(str).unique().tolist())

        narrative = Narrative(
            narrative_id=_mint_id(centroid),
            member_ids=member_ids,
            centroid=centroid,
            coherence=mean_pairwise_cosine(vectors),
            size=len(member_ids),
            author_count=int(rows["author_id"].nunique()) if "author_id" in rows else 0,
            first_seen=timestamps.min().to_pydatetime() if len(timestamps) else None,
            last_seen=timestamps.max().to_pydatetime() if len(timestamps) else None,
            platforms=platforms,
            top_domains=_top_values(rows, "domains"),
            top_hashtags=_top_values(rows, "hashtags"),
            velocity=peak_posts_per_hour(timestamps),
            severity=self._severity(rows, record_scores),
        )
        narrative.membership = {
            member: 1.0 for member in member_ids
        }
        # The clustered representative carries HDBSCAN's own membership
        # probability; collapsed near-duplicates inherit it, because they are by
        # construction the same content.
        for position, probability in zip(representative_positions, probabilities, strict=True):
            for member in _members_of(position, embeddings, member_ids):
                narrative.membership[member] = probability
        narrative.representative_ids = self._pick_representatives(
            member_ids, vectors, representative_positions, embeddings, centroid, rows
        )
        return narrative

    def _severity(
        self, rows: pd.DataFrame, record_scores: pd.DataFrame | None
    ) -> float | None:
        """Aggregate member ``misinfo_prob`` into one narrative-level number.

        **The choice, and why.** Severity is the engagement-weighted mean of the
        member scores in the top quartile -- an expected-shortfall statistic, not
        a mean and not a percentile.

        A narrative is a mixture: a handful of strongly misinformation-like posts
        surrounded by a longer tail of neutral discussion *about* them. Three
        candidate aggregations, and why two of them fail:

        * **Mean.** Dominated by the tail. A narrative of 10 posts at 0.95 and 30
          at 0.02 scores 0.25, so precisely the narratives worth surfacing look
          mild.
        * **A high percentile.** Better, but it only fires when the alarming
          fraction exceeds ``100 - p``. At the 75th percentile the same example
          scores 0.02, because 75% of its posts genuinely are neutral. Tuning
          ``p`` per corpus is not a principled fix.
        * **Maximum.** Wrong in the other direction: one confident false positive
          sets the whole narrative alight.

        Averaging the top quartile takes the alarming group on its own terms --
        the neutral tail cannot dilute it, and no single outlier can define it,
        because it is averaged against its nine peers. On the example above it
        scores 0.95; on 1 post at 0.99 among 39 at 0.05 it scores 0.14.

        Engagement-weighting then reflects that a claim seen 50,000 times matters
        more than the same claim seen twice. Posts with unmeasurable engagement
        (Phase 1's `null`) fall back to weight 1 rather than being dropped or
        treated as zero-reach -- otherwise ConvoKit Reddit, which exposes no
        engagement at all, would silently contribute nothing.

        Returns ``None`` when no member has a ``misinfo_prob``: the misinfo model
        has not run, and a severity computed from nothing would be a fabrication.
        """
        if record_scores is None or not len(record_scores):
            return None
        scores = record_scores.set_index("record_id")
        joined = scores.reindex(rows["id"]) if "id" in rows else scores.reindex(rows.index)
        if "misinfo_prob" not in joined.columns:
            return None
        values = pd.to_numeric(joined["misinfo_prob"], errors="coerce")
        mask = values.notna()
        if not mask.any():
            return None
        values = values[mask].to_numpy(dtype=float)

        weights = (
            _engagement_weights(rows.loc[mask.to_numpy()])
            if self.severity_weighted
            else np.ones(len(values))
        )
        return float(weighted_tail_mean(values, weights, self.severity_percentile))

    def _pick_representatives(
        self,
        member_ids: list[str],
        vectors: np.ndarray,
        representative_positions: list[int],
        embeddings: EmbeddingResult,
        centroid: np.ndarray,
        rows: pd.DataFrame,
    ) -> list[str]:
        """Closest to the centroid, spread across platforms.

        Spread matters: three representatives all from Mastodon make a
        cross-platform narrative look like a single-platform one in the UI, and
        they give the summarizer a one-sided view of the claim.
        """
        similarities = cosine_similarity(vectors, centroid.reshape(1, -1)).ravel()
        order = np.argsort(-similarities)
        ranked = [
            (embeddings.record_ids[representative_positions[i]], float(similarities[i]))
            for i in order
        ]
        platform_of = (
            rows.set_index("id")["source"].astype(str).to_dict() if "id" in rows else {}
        )

        picked: list[str] = []
        seen_platforms: set[str] = set()
        # First pass: one per platform, best-ranked first.
        for record_id, _ in ranked:
            platform = platform_of.get(record_id, "unknown")
            if platform not in seen_platforms:
                picked.append(record_id)
                seen_platforms.add(platform)
            if len(picked) >= self.n_representatives:
                break
        # Second pass: fill the remaining slots with the next-best overall.
        for record_id, _ in ranked:
            if len(picked) >= self.n_representatives:
                break
            if record_id not in picked:
                picked.append(record_id)
        return picked

    # --- cross-run identity ----------------------------------------------
    def _carry_ids(
        self, narratives: list[Narrative], previous: pd.DataFrame | None
    ) -> dict[str, list[str]]:
        """Match new clusters to previous ones and carry the id forward.

        Greedy best-match above ``stability_threshold``, each old id claimed at
        most once. Two new clusters matching one old id means the narrative
        split: the better match keeps the id, the other is new, and the split is
        logged. An old id nobody matches is a death.
        """
        transitions: dict[str, list[str]] = {"carried": [], "new": [], "died": [], "split": []}
        if previous is None or not len(previous) or "centroid" not in previous.columns:
            transitions["new"] = [n.narrative_id for n in narratives]
            return transitions

        old_ids: list[str] = []
        old_vectors: list[np.ndarray] = []
        for row in previous.itertuples(index=False):
            centroid = getattr(row, "centroid", None)
            if centroid is None or len(centroid) == 0:
                continue
            old_ids.append(str(row.narrative_id))
            old_vectors.append(np.asarray(centroid, dtype=np.float32))
        if not old_ids:
            transitions["new"] = [n.narrative_id for n in narratives]
            return transitions

        old_matrix = np.stack(old_vectors)
        new_matrix = np.stack([n.centroid for n in narratives])
        similarity = cosine_similarity(new_matrix, old_matrix)

        # Greedy over all (new, old) pairs by descending similarity, so the best
        # available match wins rather than whichever cluster is iterated first.
        pairs = [
            (float(similarity[i, j]), i, j)
            for i in range(similarity.shape[0])
            for j in range(similarity.shape[1])
            if similarity[i, j] >= self.stability_threshold
        ]
        pairs.sort(reverse=True)
        claimed_old: set[int] = set()
        claimed_new: set[int] = set()
        contested: Counter[str] = Counter()

        for score, i, j in pairs:
            contested[old_ids[j]] += 1
            if i in claimed_new or j in claimed_old:
                continue
            narratives[i].narrative_id = old_ids[j]
            narratives[i].id_origin = "carried"
            claimed_new.add(i)
            claimed_old.add(j)
            transitions["carried"].append(old_ids[j])
            log.debug("carried %s (cosine %.3f)", old_ids[j], score)

        transitions["new"] = [n.narrative_id for n in narratives if n.id_origin == "new"]
        transitions["died"] = [old_ids[j] for j in range(len(old_ids)) if j not in claimed_old]
        transitions["split"] = [key for key, count in contested.items() if count > 1]

        log.info(
            "narrative ids: %d carried, %d new, %d died, %d split",
            len(transitions["carried"]),
            len(transitions["new"]),
            len(transitions["died"]),
            len(transitions["split"]),
        )
        return transitions

    # --- diagnostics -----------------------------------------------------
    def _diagnostics(self, matrix: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Silhouette on a sample, plus the shape of the labelling.

        Silhouette on the *clustered* points only: including noise makes the
        number meaningless, because noise has no cluster to be far from.
        """
        out: dict[str, Any] = {}
        clustered = labels != NOISE_LABEL
        n_clusters = len(set(labels[clustered].tolist()))
        out["n_clusters"] = n_clusters
        out["clustered_fraction"] = round(float(clustered.mean()), 3) if len(labels) else 0.0

        if n_clusters < 2 or clustered.sum() < 10:
            out["silhouette"] = None
            out["silhouette_note"] = "too few clusters or points to be meaningful"
            return out

        from sklearn.metrics import silhouette_score

        points = matrix[clustered]
        subset = labels[clustered]
        rng = np.random.default_rng(self.settings.seed)
        if len(points) > 2000:
            sample = rng.choice(len(points), size=2000, replace=False)
            points, subset = points[sample], subset[sample]
            out["silhouette_sampled"] = 2000
        try:
            out["silhouette"] = round(float(silhouette_score(points, subset, metric="cosine")), 3)
        except ValueError as exc:  # pragma: no cover - degenerate labelling
            out["silhouette"] = None
            out["silhouette_note"] = str(exc)
        return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    """Mean pairwise cosine similarity within a cluster -- exactly, in O(n*d).

    For L2-normalized vectors, ``sum_{i!=j} v_i . v_j == ||sum v||^2 - n``, so
    the mean over the n(n-1) ordered pairs needs no pairwise matrix at all. This
    matters: the naive version is O(n^2 * d) and clusters can be large.
    """
    n = len(vectors)
    if n < 2:
        return 1.0
    unit = l2_normalize(vectors)
    total = unit.sum(axis=0)
    numerator = float(total @ total) - n
    return float(np.clip(numerator / (n * (n - 1)), -1.0, 1.0))


def peak_posts_per_hour(timestamps: pd.Series) -> float:
    """Highest posts-per-hour in any one-hour window.

    Peak, not mean: a narrative that produced 200 posts in one hour and then
    went quiet for a week is the interesting case, and a mean over the whole
    lifespan erases it completely.
    """
    if timestamps is None or not len(timestamps):
        return 0.0
    ordered = pd.Series(sorted(pd.to_datetime(timestamps, utc=True)))
    if len(ordered) == 1:
        return 1.0
    # Seconds since the first post, computed as a timedelta.
    #
    # Not `astype("int64") // 10**9`: that returns whatever unit the column
    # happens to carry, and pandas 3 defaults datetime64 to microseconds rather
    # than nanoseconds. Dividing by 1e9 then compresses the timeline 1000x, one
    # hour swallows six weeks, and every cluster reports its entire size as its
    # peak-hour velocity -- a wrong number that looks perfectly plausible.
    values = (ordered - ordered.iloc[0]).dt.total_seconds().to_numpy()
    # Sliding window: for each start, how many posts fall within 3600 seconds.
    right = np.searchsorted(values, values + 3600, side="right")
    return float((right - np.arange(len(values))).max())


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Weighted percentile via the cumulative weight distribution."""
    if not len(values):
        return float("nan")
    order = np.argsort(values)
    values, weights = values[order], np.asarray(weights, dtype=float)[order]
    total = weights.sum()
    if total <= 0:
        return float(np.percentile(values, percentile))
    cumulative = (np.cumsum(weights) - 0.5 * weights) / total
    return float(np.interp(percentile / 100.0, cumulative, values))


def weighted_tail_mean(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Weighted mean of the values at or above the ``percentile`` threshold.

    An expected-shortfall statistic. See ``NarrativeClusterer._severity`` for why
    this rather than a mean, a percentile or a maximum.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(values):
        return float("nan")
    if len(values) == 1:
        return float(values[0])

    threshold = weighted_percentile(values, weights, percentile)
    selected = values >= threshold
    if not selected.any():  # pragma: no cover - only if threshold exceeds max
        return float(values.max())

    tail_values, tail_weights = values[selected], weights[selected]
    total = tail_weights.sum()
    if total <= 0:
        return float(tail_values.mean())
    return float((tail_values * tail_weights).sum() / total)


def _engagement_weights(rows: pd.DataFrame) -> np.ndarray:
    """Weight by engagement, treating "not measurable" as weight 1.

    Not zero: a Reddit comment with no exposed engagement is not a post nobody
    saw. Treating unmeasurable as zero-reach would silently exclude entire
    platforms from severity.
    """
    if "engagement" not in rows.columns:
        return np.ones(len(rows), dtype=float)

    def weight(value) -> float:
        if not isinstance(value, dict):
            return 1.0
        present = [v for v in value.values() if v is not None]
        if not present:
            return 1.0
        return 1.0 + max(0.0, float(sum(present)))

    return rows["engagement"].map(weight).to_numpy(dtype=float)


def _top_values(rows: pd.DataFrame, column: str, limit: int = 10) -> list[str]:
    if column not in rows.columns:
        return []
    counter: Counter[str] = Counter()
    for value in rows[column]:
        if value is None:
            continue
        try:
            counter.update(str(v) for v in value if v)
        except TypeError:
            continue
    return [value for value, _ in counter.most_common(limit)]


def _members_of(position: int, embeddings: EmbeddingResult, member_ids: list[str]) -> list[str]:
    record_id = embeddings.record_ids[position]
    return [record_id] if record_id in member_ids else []


def _mint_id(centroid: np.ndarray) -> str:
    """A stable id derived from the centroid, so a rerun on identical data with
    no previous table still produces the same id."""
    digest = hashlib.sha256(np.round(centroid, 4).tobytes()).hexdigest()
    return f"nar-{digest[:12]}"


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit_table(n: int = 20, settings: ModelingSettings | None = None) -> str:
    """Render N clusters for the manual coherence audit.

    This is the evaluation for unsupervised output. A human reads these and
    rates each coherent / mixed / junk; the counts go in
    ``artifacts/error_analysis/cluster.md``. There is no substitute and no
    automated metric that means the same thing.
    """
    from modeling.io import ScoredStore

    settings = settings or get_settings()
    store = ScoredStore(settings)
    narratives = store.read("narratives")
    membership = store.read("narrative_membership")
    if not len(narratives):
        return "no narratives on disk; run `modeling cluster` first"

    from modeling.io import CorpusReader

    records = CorpusReader(settings).records(columns=["id", "text", "source"])
    text_of = records.set_index("id")["text"].to_dict() if len(records) else {}

    lines = [
        "# Cluster audit sample",
        "",
        "Rate each: coherent (one claim) / mixed (several claims) / junk (no claim).",
        "",
    ]
    for row in narratives.sort_values("size", ascending=False).head(n).to_dict("records"):
        reps = membership.loc[
            (membership["narrative_id"] == row["narrative_id"])
            & (membership["is_representative"])
        ]["record_id"].tolist()
        lines.append(
            f"## {row['narrative_id']} — size {row['size']}, "
            f"coherence {row['coherence']:.3f}, {row['author_count']} authors, "
            f"platforms {list(row['platforms'])}"
        )
        for record_id in reps[:5]:
            snippet = str(text_of.get(record_id, "<text unavailable>"))[:160]
            lines.append(f"  - {snippet}")
        lines.append("")
        lines.append("  rating: ____________")
        lines.append("")
    return "\n".join(lines)
