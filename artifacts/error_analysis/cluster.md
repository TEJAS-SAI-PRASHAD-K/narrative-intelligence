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

Run of 2026-08-14 (`4188` records / `1906` authors / 5 sources):

```
narratives 21 · records_after_dedupe 4096 · noise_ratio 0.729
size min/median/max 15 / 26 / 537 · coherence_mean 0.419
clustered_fraction 0.267 · silhouette 0.257 · ids carried 21, new 0
```

The size distribution is the diagnostic that fires here: `size_max` of 537 against
a median of 26 means one cluster holds 13% of the corpus and 44% of everything
that got clustered at all. Its coherence (0.354) is below the corpus mean.

## The audit — completed by hand

Rated 20 clusters, largest first:

- **coherent** — the members share one claim
- **mixed** — two or three distinct claims got merged
- **junk** — no shared claim; the cluster is an artefact

| rating | count | share |
|---|---|---|
| coherent | 13 | 65% |
| mixed | 3 | 15% |
| junk | 4 | 20% |

**A junk rate above roughly a fifth means `min_cluster_size` is too permissive
for this corpus.** Raise it in `configs/models.yaml`, re-cluster, and re-audit.

Junk sits *at* the threshold, not above it — and see the analysis below for why
raising `min_cluster_size` is the wrong lever here regardless.

### Per-cluster ratings

| narrative | size | coh. | platforms | rating | note |
|---|---|---|---|---|---|
| nar-3efb136df79c | 537 | 0.354 | all 5 | mixed | Trump vaccine EO. Fact-checks + paediatrician alarm + unrelated personal vaccine anecdotes. A topic, not a claim. |
| nar-5496c75a618b | 85 | 0.257 | reddit | mixed | Exam-reuse incident. Three separable claims: professorial self-plagiarism, university overreach, who owns textbook questions. Lowest coherence in the sample. |
| nar-22c099c219ff | 48 | 0.380 | reddit | mixed | Singapore social discourse: 377A repeal *and* race/religion education *and* how to persuade the apolitical. |
| nar-0bdb9bae470a | 40 | 0.345 | reddit | coherent | One shared proposition, crudely put. Clusters correctly. |
| nar-8cfddb7cd8d8 | 36 | 0.316 | news, reddit | coherent | European total eclipse. One stray Longyearbyen/weather comment bridged in on Arctic vocabulary. |
| nar-a7bb5e30e6d1 | 33 | 0.417 | reddit | coherent | "I cut myself handling a knife" anecdotes. Experience cluster rather than narrative, but internally consistent. |
| nar-16aae95b6008 | 32 | 0.544 | news | coherent | Minnesota primary night. Borderline — the Flanagan Senate result is a distinct race from Demuth/Lindell, but it is one news event as covered. |
| nar-3332f16d1291 | 31 | 0.277 | reddit, youtube | **junk** | "Yea / Yes / yep / Absolutely." Bare agreement tokens. |
| nar-19df506f175f | 29 | 0.513 | reddit | coherent | Why people knock instead of ringing the doorbell. |
| nar-8125fdb75b95 | 27 | 0.325 | reddit | coherent | Maggoty bin after 18lb of meat in Texas heat; remedies. One thread. |
| nar-16b4d2b801d1 | 26 | 0.446 | reddit | coherent | NS and gendered discrimination, perception vs treatment. Two stances, one disputed proposition. |
| nar-1f8df74014eb | 26 | 0.940 | reddit | **junk** | r/singapore daily boilerplate threads, Sept 2018, 3 authors. Highest coherence in the sample and zero content. |
| nar-1f2cd790bdbb | 25 | 0.309 | news | coherent | Colombia earthquake, Cali, death toll and rescue. |
| nar-957d23127fc4 | 24 | 0.493 | reddit | coherent | Standard of Mandarin in Singapore, rote teaching, Nantah. |
| nar-3927a1b4dc30 | 24 | 0.413 | reddit | coherent | GrabHitch late-night availability and commissions. |
| nar-4e9578a323d7 | 22 | 0.336 | reddit | **junk** | "Thanks" / "good luck for your N levels". Politeness tokens. |
| nar-1737b395c349 | 22 | 0.373 | reddit, youtube | **junk** | Emoji-only reactions. |
| nar-bceeb5b07c3b | 20 | 0.357 | reddit | coherent | Beeturia — red urine mistaken for blood. |
| nar-80242a27724f | 19 | 0.412 | reddit | coherent | Singapore cat welfare: CWS mesh requirements, falls from height, vet costs. |
| nar-16ccfb38ed71 | 16 | 0.472 | reddit, youtube | coherent | Victim in a well could not swim. One shared factual claim. |

## Failure modes seen so far

- **Topic clusters, not claim clusters.** Confirmed, and it accounts for all
  three `mixed` ratings. Embedding similarity is topical; MiniLM has no notion of
  stance. nar-3efb136df79c is the pure case — a 537-member vaccine *subject*
  containing the executive order, the fact-checks of it, professional reaction to
  it, and idle personal vaccination stories. This is what the stance module
  (currently null) would separate. Note it is not *disagreement* that produces
  the mixed rating in these three: nar-16b4d2b801d1 has two opposed stances and
  rates coherent, because both sides are arguing the same proposition. The mixed
  clusters are ones holding several unrelated propositions.
- **Short-text clusters.** Confirmed, and this is the systematic failure. Three
  of the four junk clusters (nar-3332f16d1291, nar-4e9578a323d7,
  nar-1737b395c349, ~75 records) are agreement tokens, thanks, and emoji. The
  length floor is applied to GDELT article metadata only; Reddit and YouTube
  comments pass through it. At 256-token truncation these documents carry almost
  no signal, so they collapse onto each other and HDBSCAN finds them dense.
- **Boilerplate clusters.** Not anticipated. nar-1f8df74014eb is 26 near-identical
  r/singapore daily-thread stubs from 3 authors, coherence 0.940 — the highest in
  the sample. Near-duplicate collapse ran (92 records into 4096 representatives)
  and did not catch these, because the date in each title makes them similar
  rather than identical. Worth noting for the diagnostics table: high coherence
  is evidence of *nothing* on its own.
- **Platform clusters.** Not confirmed. 13 of 20 clusters are Reddit-only, but
  that reflects corpus composition, not register-driven grouping — each of those
  is a recognisable single conversation. Deprioritise this hypothesis.

## Written analysis

Junk is 4/20, at the threshold rather than over it, so `min_cluster_size` stays
where it is — and the size distribution says it should stay there regardless.
The junk clusters are sizes 22, 22, 26, 31, sitting in the middle of the
distribution, not the bottom. Raising `min_cluster_size` to 27 would take out
three of the four junk clusters and seven of the thirteen coherent ones
(sizes 16, 19, 20, 24, 24, 25, 26). The parameter cannot separate these
populations because size is not what distinguishes them. Document length is.

The fix for the dominant failure mode is a config change, but not that one:
extend the length floor from GDELT metadata to every platform. A floor of roughly
15–20 tokens of body text removes the agreement/emoji/thanks population without
touching a single cluster rated coherent, and should also cut the noise ratio,
since most of those short documents are presumably sitting in the 73% noise
already. Re-audit after, because removing ~75 documents from three clusters will
redraw density elsewhere.

The second failure mode is not a parameter problem at all. nar-3efb136df79c is a
region of embedding space labelled "vaccines", and no `min_cluster_size` splits
it into the executive order, the fact-checking response, and the paediatric
professional reaction, because MiniLM places all three in the same place by
construction. Two candidate fixes, in order of cost: a stronger embedding model
with more room (the 256-token cap truncated 175 records, and MiniLM-L6 is the
weakest sentence encoder in common use), or the stance module, which is the
actual answer and is currently null. Until one of those exists, the honest
description of the largest cluster in this corpus is topical, not narrative, and
anything downstream that treats it as a single claim will be wrong.

Third, on what the numbers are worth. Silhouette 0.257 and coherence_mean 0.419
are computed over the clustered 26.7% only, and the sample above shows those
numbers are not tracking claim quality: the highest-coherence cluster in the
audit is junk, and the largest is mixed. Read the size distribution and the audit;
treat silhouette as a regression check between runs rather than a quality score.

### Housekeeping from this run

- The embedding cache was discarded — `(4188, 384)` array against 4126 keys for a
  384-dim model. The array and the key list disagree on row count, so the cache is
  being written inconsistently somewhere; every run currently re-embeds from
  scratch (~18s here, worse as the corpus grows).
- Two record ids appear under more than one `date=` partition and Phase 1's
  per-partition dedupe cannot see them. Small now, but it is a cross-partition
  dedupe pass that doesn't exist.
- `narratives` and `narrative_membership` were unchanged and nothing was written,
  so re-running `cluster` without a config change is a no-op. Any re-audit needs
  the config edited first.