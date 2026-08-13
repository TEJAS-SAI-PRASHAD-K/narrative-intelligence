# Model card — narrative clustering

**Modules:** `modeling/text/embed.py`, `modeling/text/cluster.py`
**Versions:** embed `v0.1.0`, cluster `v0.1.0`
**Outputs:** `narratives`, `narrative_membership`

---

## Status: in use, unsupervised

This is the one module with no supervised metric, because there are no gold
narrative labels. **There is no F1 here and there will not be one.**

## Method

**Embeddings.** `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, L2-normalized.
Config-switchable to `BAAI/bge-base-en-v1.5` (768-dim, better, needs a GPU for
bulk). **The dimension is read from the model, never hardcoded** — Phase 4's
pgvector column width depends on it, and a hardcoded 384 that disagrees with a
768-dim checkpoint is a migration failure discovered in production.

Cached by `(model_name, model_version, sha256(text))`. Truncation is recorded,
not silent: over the token cap, the first sentence is kept whole and the
remainder filled from the head, because a news article's claim lives in its lede
and a naive head-truncation can cut it in half.

**Clustering.** HDBSCAN (`sklearn.cluster.HDBSCAN`) on Euclidean distance over
L2-normalized vectors, which is monotonically equivalent to cosine.

*Why not k-means.* We do not know how many narratives a corpus contains, and most
posts belong to none of them. k-means demands a k and assigns every point, so it
manufactures narratives out of background chatter. HDBSCAN discovers the count
and has a genuine noise label.

*Implementation note.* `sklearn`'s HDBSCAN rather than the standalone `hdbscan`
package — same algorithm, one fewer build-fragile dependency. The standalone
package's `approximate_predict` is not needed because cross-run identity is
handled by centroid matching, which is more robust here than soft-assigning new
points to an old model.

**Pre-cluster dedupe.** Near-duplicates (simhash Hamming ≤ 3) collapse to one
representative *for the clustering decision only*; the full member list is
restored afterwards, so `size` and `author_count` reflect reality. Without this,
one viral repost swarm forms its own dense cluster and dominates every quality
metric.

## Cross-run narrative identity

A product requirement, not a nicety: the UI shows "Generated 48 days ago / Update
Now", narrative ids are user-visible, and a user may have renamed one. Minting
fresh ids on every run would orphan all of that.

On rerun, new clusters are matched to previous ones by centroid cosine ≥ 0.85 and
the id is carried forward; only genuinely new clusters get a new id. Matching is
greedy over all pairs by descending similarity, and each old id is claimed at
most once — two new clusters matching one old id means the narrative **split**,
the better match keeps the id, and the split is logged. Ids nobody matches are
logged as **deaths**.

Verified by `tests/test_cluster.py`, including the case that matters: identical
ids across two runs on *overlapping* (grown) data.

## Derived metrics

**`coherence`** — mean pairwise cosine within the cluster, computed exactly in
O(n·d) via the identity `sum_{i≠j} v_i·v_j = ‖Σv‖² − n` for unit vectors. Read
this before `size`: a large cluster with low coherence is a bag of loosely
related posts.

**`velocity`** — **peak** posts-per-hour in any one-hour window, not the mean. A
narrative that produced 200 posts in one hour then went quiet for a week is the
interesting case, and a lifetime mean erases it.

**`severity`** — the engagement-weighted mean of member `misinfo_prob` scores in
the **top quartile**: an expected-shortfall statistic, not a mean and not a
percentile.

*Why not the alternatives.* A **mean** is dominated by the tail — a narrative of
10 posts at 0.95 and 30 at 0.02 scores 0.25, so precisely the narratives worth
surfacing look mild. A **high percentile** only fires when the alarming fraction
exceeds `100 − p`; at the 75th percentile that same example scores 0.02, because
75% of its posts genuinely are neutral. The **maximum** is wrong in the other
direction: one confident false positive sets the whole narrative alight.
Averaging the top quartile takes the alarming group on its own terms — the
neutral tail cannot dilute it, and no single outlier can define it.

Engagement-weighting reflects that a claim seen 50,000 times matters more than
the same claim seen twice. Posts with unmeasurable engagement (Phase 1's `null`)
get weight 1, not 0 — otherwise ConvoKit Reddit, which exposes no engagement at
all, would silently contribute nothing.

**`severity` is null when no member has a `misinfo_prob`.** A severity computed
from nothing would be a fabrication.

**Representatives** — 3–5 per cluster, closest to the centroid and spread across
platforms. Spread matters: three representatives all from Mastodon make a
cross-platform narrative look single-platform and give the summarizer a one-sided
view of the claim.

## Evaluation — how unsupervised output is judged honestly

1. **Silhouette** on the clustered points (noise excluded — noise has no cluster
   to be far from), sampled at 2000 points above that size.
2. **Noise ratio.** A very low ratio usually means `min_cluster_size` is too
   permissive and the model is manufacturing narratives.
3. **Cluster size distribution.** One cluster holding most of the corpus is a
   failure however good the silhouette.
4. **A manual audit of 20 clusters**, rated coherent / mixed / junk by a person,
   written into `artifacts/error_analysis/cluster.md`.

**The audit is the evaluation. The other three are triage.** `modeling cluster
--audit 20` renders the sample.

## Known limitations

- **English-only.** Non-English text is embedded by an English model and will
  cluster on orthography rather than meaning; the language policy skips it.
- **GDELT short text.** Article metadata (median ~83 characters in this corpus)
  embeds to something that clusters on stopwords. Records below the configured
  length floor are excluded from clustering with a logged count.
- **A cluster is not a claim.** It is a region of embedding space. The label is
  a *description* of what its members say, and `label_source` records whether a
  model or a truncated quotation produced it.
