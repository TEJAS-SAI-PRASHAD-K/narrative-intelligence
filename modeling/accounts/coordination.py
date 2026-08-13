"""Coordination detection: a co-behaviour graph, communities, and a null model.

This is network-level, not per-account. Two accounts are linked when they do the
same thing at nearly the same time, more often than chance explains.

**Four kinds of evidence**, each stored on the edge so the UI can say *why* two
nodes are linked rather than asserting that they are:

``near_dup``     near-identical content inside the window (simhash Hamming <= 3)
``cotweet``      the same URL or domain inside the window
``hashtag_seq``  the same ordered hashtag sequence inside the window
``temporal``     replies to the same parent inside a tight window

**The null model is the finding, not the communities.** Any graph has
communities — Louvain will happily partition random noise and report a
modularity above zero. So the corpus is re-run with timestamps shuffled within
each author, which destroys cross-account timing coincidences while preserving
every author's own volume and rhythm. If real modularity does not exceed
shuffled modularity, "we found coordinated communities" is not a result, and the
report says so.

**Combinatorics.** Naive pairwise comparison over n records is O(n^2): at 200k
records that is 2x10^10 comparisons. Two bucketing passes avoid it:

1. **Time bucketing.** Records are assigned to half-open windows of
   ``window_minutes``, with each record also placed in the *next* window so a
   pair straddling a boundary is not missed. Comparisons happen only within a
   bucket.
2. **Content bucketing.** Inside a time bucket, near-duplicate candidates are
   grouped by simhash prefix (an LSH band) before any Hamming distance is
   computed; URL and hashtag evidence group by exact key.

Complexity becomes O(sum over buckets of k^2) where k is the bucket occupancy —
linear in the corpus for a fixed posting rate, and bounded by
``MAX_BUCKET_PAIRS`` for the pathological case of one enormous burst.

**Input filtering.** Only sources with real threading and stable author identity
(ConvoKit Reddit, Mastodon, YouTube comments). Kaggle-flat Reddit carries
``parent_id = None`` honestly and is excluded — the filter is on *threading
availability*, not on a source name, so a future flat dump is excluded
automatically. The exclusion is reported.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config
from modeling.io import as_list

log = logging.getLogger(__name__)

EVIDENCE_KINDS = ("near_dup", "cotweet", "hashtag_seq", "temporal")

#: Cap on pairs generated from one bucket. A single viral burst can put tens of
#: thousands of records in one window; comparing all of them is quadratic and
#: adds nothing, because the swarm is already one obvious component.
MAX_BUCKET_PAIRS = 200_000


@dataclass
class Edge:
    src: str
    dst: str
    evidence: str
    observations: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.evidence)


@dataclass
class CoordinationResult:
    edges: list[dict[str, Any]]
    communities: dict[str, str]
    community_sizes: dict[str, int]
    scores: dict[str, float]
    modularity: float
    null_modularity: float
    null_std: float
    n_accounts: int
    n_records: int
    excluded_sources: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def exceeds_null(self) -> bool:
        """Whether the real graph is more modular than the time-shuffled one.

        The bar is one standard deviation above the shuffled mean. Below it,
        the communities are not evidence of coordination.
        """
        return self.modularity > self.null_modularity + self.null_std

    def summary(self) -> dict[str, Any]:
        return {
            "accounts": self.n_accounts,
            "records": self.n_records,
            "edges": len(self.edges),
            "communities": len(set(self.communities.values())),
            "modularity": round(self.modularity, 4),
            "null_modularity": round(self.null_modularity, 4),
            "null_std": round(self.null_std, 4),
            "exceeds_null": self.exceeds_null,
            "excluded_sources": self.excluded_sources,
            **self.diagnostics,
        }


class CoordinationDetector:
    module = "coordination"

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.window = pd.Timedelta(minutes=int(self.config.get("window_minutes", 60)))
        self.simhash_hamming = int(self.config.get("simhash_hamming", 3))
        self.min_edge_weight = float(self.config.get("min_edge_weight", 0.15))
        self.min_hashtag_sequence = int(self.config.get("min_hashtag_sequence", 2))
        self.null_shuffles = int(self.config.get("null_model_shuffles", 5))
        self.weights = dict(self.config.get("evidence_weights") or {})
        self.eligible_sources = set(self.config.get("eligible_sources") or [])

    # --- input filtering -------------------------------------------------
    def eligible_records(self, records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Keep only records usable for coordination work.

        Filtered on *threading availability and author identity*, not on a source
        name: a source whose records all carry ``parent_id = None`` cannot
        support co-reply evidence and its "authors" may not be people. The
        counts are returned so the eval report can state exactly what was
        excluded and why.
        """
        excluded: dict[str, int] = {}
        work = records.copy()

        if "source" in work.columns and self.eligible_sources:
            ineligible = ~work["source"].astype(str).isin(self.eligible_sources)
            for source, count in work.loc[ineligible, "source"].value_counts().items():
                excluded[f"{source}:source_not_eligible"] = int(count)
            work = work.loc[~ineligible]

        # Deleted authors are content without an actor. Keeping them would
        # merge every tombstoned post into one enormous pseudo-account.
        if "author_id" in work.columns:
            deleted = work["author_id"].astype(str).str.endswith(":__deleted__")
            if deleted.any():
                excluded["deleted_author"] = int(deleted.sum())
                work = work.loc[~deleted]

        # A source with no threading at all cannot contribute co-reply evidence
        # and is usually a flat dump with unreliable author identity.
        if "parent_id" in work.columns and "source" in work.columns:
            for source, group in work.groupby("source"):
                if group["parent_id"].notna().mean() == 0.0 and source != "mastodon":
                    excluded[f"{source}:no_threading"] = len(group)
                    work = work.loc[work["source"] != source]

        if excluded:
            log.info(
                "coordination input: excluded %d records (%s)",
                sum(excluded.values()),
                ", ".join(f"{k}={v}" for k, v in sorted(excluded.items())),
            )
        return work, excluded

    # --- graph construction ----------------------------------------------
    def build_edges(self, records: pd.DataFrame) -> list[Edge]:
        """Bucketed co-behaviour edge extraction. See the module docstring."""
        work = records.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(work) < 2:
            return []

        origin = work["timestamp"].min()
        window_seconds = self.window.total_seconds()
        offsets = (work["timestamp"] - origin).dt.total_seconds()
        work = work.assign(_bucket=(offsets // window_seconds).astype(int))

        edges: dict[tuple[str, str, str], Edge] = {}
        pairs_examined = 0
        truncated_buckets = 0

        # Each record participates in its own bucket and the previous one, so a
        # pair either side of a boundary is still compared exactly once.
        by_bucket: dict[int, list[int]] = defaultdict(list)
        for position, bucket in enumerate(work["_bucket"].to_numpy()):
            by_bucket[int(bucket)].append(position)

        rows = work.to_dict(orient="records")
        for bucket in sorted(by_bucket):
            members = by_bucket[bucket] + by_bucket.get(bucket - 1, [])
            if len(members) < 2:
                continue
            examined, truncated = self._edges_in_bucket([rows[i] for i in members], edges)
            pairs_examined += examined
            truncated_buckets += int(truncated)

        log.info(
            "coordination: %d edge(s) from %d bucket(s), %d candidate pair(s) examined%s",
            len(edges),
            len(by_bucket),
            pairs_examined,
            f"; {truncated_buckets} bucket(s) truncated at {MAX_BUCKET_PAIRS} pairs"
            if truncated_buckets
            else "",
        )
        return list(edges.values())

    def _edges_in_bucket(
        self, rows: list[dict[str, Any]], edges: dict[tuple[str, str, str], Edge]
    ) -> tuple[int, bool]:
        """Extract every evidence kind within one time bucket."""
        examined = 0
        truncated = False

        # -- near-duplicate content: LSH band on the simhash prefix ---------
        by_prefix: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get("simhash")
            if value is None or pd.isna(value):
                continue
            by_prefix[int(value) >> 48].append(row)
        for group in by_prefix.values():
            if len(group) < 2:
                continue
            budget = min(len(group) * (len(group) - 1) // 2, MAX_BUCKET_PAIRS)
            for count, (a, b) in enumerate(itertools.combinations(group, 2)):
                if count >= budget:
                    truncated = True
                    break
                examined += 1
                if a["author_id"] == b["author_id"]:
                    continue
                distance = bin(int(a["simhash"]) ^ int(b["simhash"])).count("1")
                if distance <= self.simhash_hamming:
                    _add(edges, a, b, "near_dup")

        # -- co-sharing a URL or domain -------------------------------------
        self._group_evidence(rows, edges, "cotweet", _url_keys)

        # -- identical hashtag sequences ------------------------------------
        self._group_evidence(
            rows,
            edges,
            "hashtag_seq",
            lambda row: _hashtag_keys(row, self.min_hashtag_sequence),
        )

        # -- co-reply to the same parent ------------------------------------
        self._group_evidence(rows, edges, "temporal", _parent_keys)

        return examined, truncated

    def _group_evidence(
        self,
        rows: list[dict[str, Any]],
        edges: dict[tuple[str, str, str], Edge],
        kind: str,
        key_fn,
    ) -> None:
        """Exact-key bucketing: everyone sharing a key is pairwise linked.

        Capped per key, because one viral URL shared by 5,000 accounts would
        otherwise generate 12.5M edges describing a single obvious fact.
        """
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for key in key_fn(row):
                buckets[key].append(row)

        for key, group in buckets.items():
            authors = {row["author_id"]: row for row in group}
            if len(authors) < 2:
                continue
            members = list(authors.values())
            if len(members) * (len(members) - 1) // 2 > MAX_BUCKET_PAIRS:
                log.debug("capping %s evidence for key %s (%d accounts)", kind, key, len(members))
                members = members[: int(np.sqrt(2 * MAX_BUCKET_PAIRS))]
            for a, b in itertools.combinations(members, 2):
                _add(edges, a, b, kind)

    # --- communities and scores ------------------------------------------
    def detect(
        self, records: pd.DataFrame, *, run_null_model: bool = True
    ) -> CoordinationResult:
        import networkx as nx

        work, excluded = self.eligible_records(records)
        if len(work) < 2:
            return CoordinationResult([], {}, {}, {}, 0.0, 0.0, 0.0, 0, 0, excluded,
                                      {"note": "no eligible records"})

        edges = self.build_edges(work)
        graph = self._to_graph(edges)
        if graph.number_of_edges() == 0:
            return CoordinationResult(
                [], {}, {}, {}, 0.0, 0.0, 0.0,
                int(work["author_id"].nunique()), len(work), excluded,
                {"note": "no edges above the weight floor"},
            )

        communities = nx.community.louvain_communities(
            graph, weight="weight", seed=self.settings.seed
        )
        modularity = float(nx.community.modularity(graph, communities, weight="weight"))

        assignment: dict[str, str] = {}
        sizes: dict[str, int] = {}
        for index, members in enumerate(
            sorted(communities, key=len, reverse=True)
        ):
            community_id = f"com-{index:04d}"
            sizes[community_id] = len(members)
            for account in members:
                assignment[str(account)] = community_id

        null_mean = null_std = 0.0
        if run_null_model:
            null_mean, null_std = self._null_model(work)

        scores = self._coordination_scores(graph, assignment, sizes, work)

        result = CoordinationResult(
            edges=[_edge_row(e, graph) for e in edges if graph.has_edge(e.src, e.dst)],
            communities=assignment,
            community_sizes=sizes,
            scores=scores,
            modularity=modularity,
            null_modularity=null_mean,
            null_std=null_std,
            n_accounts=int(work["author_id"].nunique()),
            n_records=len(work),
            excluded_sources=excluded,
            diagnostics={
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "evidence_counts": dict(Counter(e.evidence for e in edges)),
                "largest_community": max(sizes.values()) if sizes else 0,
            },
        )
        log.info("coordination: %s", result.summary())
        if not result.exceeds_null and run_null_model:
            log.warning(
                "modularity %.4f does NOT exceed the time-shuffled null (%.4f +/- %.4f). "
                "The communities found are not evidence of coordination, and the report "
                "must say so.",
                modularity,
                null_mean,
                null_std,
            )
        return result

    def _to_graph(self, edges: list[Edge]):
        """Collapse evidence-typed edges into one weighted graph.

        Weight is the evidence-type weight times a saturating function of the
        observation count. Saturating rather than linear: two accounts sharing a
        URL fifty times are more suspicious than sharing it twice, but not
        twenty-five times more, and a linear count lets one prolific pair
        dominate the whole partition.
        """
        import networkx as nx

        graph = nx.Graph()
        combined: dict[tuple[str, str], float] = defaultdict(float)
        for edge in edges:
            weight = float(self.weights.get(edge.evidence, 0.5))
            combined[(edge.src, edge.dst)] += weight * float(np.log1p(edge.observations))

        for (src, dst), weight in combined.items():
            if weight >= self.min_edge_weight:
                graph.add_edge(src, dst, weight=round(weight, 5))
        return graph

    def _null_model(self, records: pd.DataFrame) -> tuple[float, float]:
        """Modularity of the same corpus with timestamps shuffled within author.

        Shuffling *within* each author, not globally: that destroys cross-account
        timing coincidences — the thing coordination detection claims to find —
        while preserving each author's own volume, burstiness and diurnal
        pattern. A global shuffle would also destroy those, and would therefore
        be too easy a null to beat.
        """
        import networkx as nx

        rng = np.random.default_rng(self.settings.seed)
        values: list[float] = []
        for shuffle in range(self.null_shuffles):
            shuffled = records.copy()
            shuffled["timestamp"] = (
                shuffled.groupby("author_id")["timestamp"]
                .transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
            )
            graph = self._to_graph(self.build_edges(shuffled))
            if graph.number_of_edges() == 0:
                values.append(0.0)
                continue
            communities = nx.community.louvain_communities(
                graph, weight="weight", seed=self.settings.seed + shuffle
            )
            values.append(float(nx.community.modularity(graph, communities, weight="weight")))

        mean = float(np.mean(values)) if values else 0.0
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        log.info(
            "null model: %d time-shuffled run(s), modularity %.4f +/- %.4f",
            len(values), mean, std,
        )
        return mean, std

    def _coordination_scores(
        self,
        graph,
        assignment: dict[str, str],
        sizes: dict[str, int],
        records: pd.DataFrame,
    ) -> dict[str, float]:
        """A transparent formula, not a model.

        ``coordination_score`` is the mean of three components in [0, 1]:

        1. **Community size**, log-scaled against the largest community. A pair
           is weak evidence; a synchronized bloc of forty is not.
        2. **Mean edge weight**, normalized against the strongest edge in the
           graph. Captures *how much* evidence links this account to its
           neighbours, not merely that some does.
        3. **Participation rate** — the share of this account's own posts that
           contributed to at least one coordinated edge. An account with one
           coincidental match among 500 posts is not coordinating.

        Kept as an average of three interpretable terms on purpose. This number
        appears next to a person's account in a dashboard, and "the model said
        so" is not an acceptable answer to "why".
        """
        scores: dict[str, float] = {}
        if graph.number_of_edges() == 0:
            return scores

        max_size = max(sizes.values()) if sizes else 1
        max_weight = max(data["weight"] for _, _, data in graph.edges(data=True)) or 1.0
        posts_per_author = records.groupby("author_id").size().to_dict()

        # Records that contributed at least one edge, per author.
        involved: Counter[str] = Counter()
        for src, dst, data in graph.edges(data=True):
            involved[src] += int(data.get("observations", 1))
            involved[dst] += int(data.get("observations", 1))

        for account in graph.nodes:
            neighbours = list(graph.edges(account, data=True))
            if not neighbours:
                continue
            community = assignment.get(str(account))
            size_term = (
                float(np.log1p(sizes.get(community, 1)) / np.log1p(max_size))
                if max_size > 1
                else 0.0
            )
            weight_term = float(
                np.mean([d["weight"] for _, _, d in neighbours]) / max_weight
            )
            total_posts = max(1, int(posts_per_author.get(account, 1)))
            participation = min(1.0, involved.get(account, 0) / total_posts)
            scores[str(account)] = round(
                float(np.clip((size_term + weight_term + participation) / 3, 0.0, 1.0)), 5
            )
        return scores


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _add(edges: dict[tuple[str, str, str], Edge], a: dict, b: dict, kind: str) -> None:
    src, dst = sorted([str(a["author_id"]), str(b["author_id"])])
    if src == dst:
        return
    key = (src, dst, kind)
    edge = edges.get(key)
    if edge is None:
        edge = Edge(src, dst, kind, 0, a["timestamp"], a["timestamp"])
        edges[key] = edge
    edge.observations += 1
    for timestamp in (a["timestamp"], b["timestamp"]):
        edge.window_start = min(edge.window_start, timestamp)
        edge.window_end = max(edge.window_end, timestamp)


def _url_keys(row: dict[str, Any]) -> list[str]:
    """Shared URLs first, then domains as a weaker fallback.

    Both, because two accounts posting the same article via different shorteners
    share a domain but not a URL, and that is still co-sharing.
    """
    keys = []
    for url in as_list(row.get("urls")):
        keys.append(f"url:{url}")
    for domain in as_list(row.get("domains")):
        keys.append(f"domain:{domain}")
    return keys


def _hashtag_keys(row: dict[str, Any], minimum: int) -> list[str]:
    """The ordered hashtag sequence, when it is long enough to be distinctive.

    Ordered, not a set: ``#a #b #c`` and ``#c #b #a`` are different signatures,
    and a shared *ordering* is much stronger evidence of a shared template than
    a shared vocabulary. Sequences below ``minimum`` tags are ignored -- one
    common hashtag links half a corpus.
    """
    tags = [str(t).lower() for t in as_list(row.get("hashtags"))]
    if len(tags) < minimum:
        return []
    return ["tags:" + "|".join(tags)]


def _parent_keys(row: dict[str, Any]) -> list[str]:
    parent = row.get("parent_id")
    if parent is None or (isinstance(parent, float) and pd.isna(parent)):
        return []
    return [f"parent:{parent}"]


def _edge_row(edge: Edge, graph) -> dict[str, Any]:
    data = graph.get_edge_data(edge.src, edge.dst) or {}
    return {
        "src_author_id": edge.src,
        "dst_author_id": edge.dst,
        "weight": float(data.get("weight", 0.0)),
        "evidence": edge.evidence,
        "observations": int(edge.observations),
        "window_start": edge.window_start,
        "window_end": edge.window_end,
    }


def null_model_section(result: CoordinationResult) -> str:
    """The paragraph that decides whether the communities mean anything."""
    verdict = (
        "**Real modularity exceeds the time-shuffled null.** The community structure is "
        "not an artefact of graph construction."
        if result.exceeds_null
        else "**Real modularity does NOT exceed the time-shuffled null.** Any graph has "
        "communities; these are not evidence of coordination on this corpus."
    )
    return "\n".join(
        [
            "Timestamps were shuffled *within each author* and the whole pipeline re-run. "
            "That destroys cross-account timing coincidences while preserving every "
            "author's own volume, burstiness and diurnal rhythm — a global shuffle would "
            "destroy those too and would be too easy a null to beat.",
            "",
            f"- observed modularity: **{result.modularity:.4f}**",
            f"- time-shuffled null: **{result.null_modularity:.4f} ± {result.null_std:.4f}** "
            f"over shuffles",
            "",
            verdict,
        ]
    )
