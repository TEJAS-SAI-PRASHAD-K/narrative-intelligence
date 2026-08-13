# Narrative clustering — audit

There are no gold narrative labels, so there is no F1 and no confusion matrix.
The evaluation is this audit plus the three triage diagnostics below.

Regenerate the sample with:

```bash
python -m modeling.cli cluster --audit 20
```

## Triage diagnostics

These come from the clustering run and are recorded in
`data/scored/manifest.json` under `narratives.diagnostics`.

| diagnostic | how to read it |
|---|---|
| silhouette | separation of the clustered points only (noise excluded — noise has no cluster to be far from). Sampled at 2000 points above that size. |
| noise ratio | HDBSCAN's willingness to say "this belongs to nothing" is a feature. A *very low* ratio usually means `min_cluster_size` is too permissive and the model is manufacturing narratives out of chatter. |
| size distribution | one cluster holding most of the corpus is a failure however good the silhouette looks. |
| coherence | mean intra-cluster cosine. Read this *before* size: a large cluster with low coherence is a bag of loosely-related posts. |

## The audit — to be completed by hand

Rate 20 clusters, largest first:

- **coherent** — the members share one claim
- **mixed** — two or three distinct claims got merged
- **junk** — no shared claim; the cluster is an artefact

| rating | count | share |
|---|---|---|
| coherent | _to be filled_ | |
| mixed | _to be filled_ | |
| junk | _to be filled_ | |

**A junk rate above roughly a fifth means `min_cluster_size` is too permissive
for this corpus.** Raise it in `configs/models.yaml`, re-cluster, and re-audit.

## Failure modes seen so far

Filled in as the audit is performed. Candidates known in advance from the shape
of this corpus:

- **Topic clusters, not claim clusters.** Embedding similarity is topical. Posts
  about the same *subject* taking opposite positions land together, because
  MiniLM has no notion of stance. This is the expected dominant "mixed" case, and
  it is what the stance module (currently null) would separate.
- **Platform clusters.** Register differs enough between a Reddit comment and a
  news headline that a cluster can form around *how* people write rather than
  what they claim. Check the `platforms` column: a single-platform cluster in a
  multi-platform corpus is a candidate.
- **Short-text clusters.** GDELT article metadata is excluded below the length
  floor, but short Mastodon toots are not, and they cluster on stopwords.

## Written analysis

_To be written after completing the audit. State which failure modes are
systematic rather than incidental, and whether the fix is a parameter change, a
better embedding model, or an acknowledgement that a cluster is a region of
embedding space rather than a claim._
