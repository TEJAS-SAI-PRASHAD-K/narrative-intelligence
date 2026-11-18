# Model card — coordination detection

**Module:** `modeling/accounts/coordination.py` · **Version:** `v0.2.1`
**Outputs:** `coordination_edges`, `author_scores.community_id`,
`author_scores.coordination_score`

---

## Status: in use. Not a model — a graph and a transparent formula.

There is no trained estimator here and there is no F1. What replaces a metric is
a **null-model comparison**, and that comparison is the finding.

## Method

An undirected co-behaviour graph. Two accounts are linked when they do the same
thing inside a time window (default 60 minutes), with the **evidence type stored
on the edge** so the UI can say *why* two nodes are linked rather than merely
asserting that they are:

| evidence | trigger | weight |
|---|---|---|
| `near_dup` | near-identical content (simhash Hamming ≤ 3) | 1.0 |
| `cotweet` | the same URL or domain | 0.8 |
| `hashtag_seq` | the same **ordered** hashtag sequence (≥ 2 tags) | 0.5 |
| `temporal` | replies to the same parent | 0.3 |

Hashtag *ordering* is part of the signature: a shared ordering is much stronger
evidence of a shared template than a shared vocabulary. Sequences of one tag are
ignored — one common hashtag links half a corpus.

Edge weight is the evidence weight times `log1p(observations)`. Saturating rather
than linear: two accounts sharing a URL fifty times are more suspicious than
sharing it twice, but not twenty-five times more, and a linear count lets one
prolific pair dominate the whole partition.

Communities via Louvain (`networkx.community.louvain_communities`), seeded.

## The null model — this is the finding

**Any graph has communities.** Louvain will happily partition random noise and
report a modularity above zero, so "we found communities" is not a result.

The corpus is re-run with timestamps shuffled **within each author**. That
destroys cross-account timing coincidences — the thing coordination detection
claims to find — while preserving every author's own volume, burstiness and
diurnal rhythm. A *global* shuffle would destroy those too and would be far too
easy a null to beat.

Observed modularity must exceed the shuffled mean by at least one standard
deviation. Below that bar, `exceeds_null` is `false`, the log warns, and the
report states plainly that the communities are not evidence of coordination.

**On the demo fixture the observed modularity (0.498) does not exceed the null
(0.497 ± 0.002)** — the fixture is too small and its Reddit thread generates
`temporal` edges that the shuffle preserves. That is reported rather than
suppressed, and it is exactly the outcome the null model exists to surface.

## Per-source null

The report gives observed and null modularity for all eligible records and for each
source alone. A source is a separate graph, so an edge never joins two sources. Each null
is a set of within-author time shuffles that starts from the settings seed (20260813),
the same seed as the observed partition. A shuffle with no edges has no modularity. It is
counted in `null_empty_shuffles` and left out of the null. When more than half of the
shuffles are empty the row status is `null_degenerate`. Each row also gives the mean edge
count of the null graphs, the null mean, the 2.5th to 97.5th percentile interval and the
empirical p value `(1 + k) / (1 + n)`. Here n counts the non-empty shuffles and k counts
those at or above the observed value. The column `p_floor` is `1 / (1 + n)`. The default
is 200 shuffles. A source with fewer than `min_source_authors` authors, no edges, or fewer
than `min_source_edges` edges has a reason code (`too_few_authors`, `no_edges`,
`too_few_edges`) and no number.

A sparse shuffled graph can have a high modularity, so a corpus with few edges does not
beat its null. Read the null mean edge count beside each row.

Run it with `python -m modeling.cli coordination-report --sources ...`. The option
`--sources` may name a source that the production eligible set leaves out. That run is a
research measurement. It never writes to the scored tables.

### Telegram

Telegram is not in `eligible_sources`, so it does not reach `author_scores.coordination_score`
or `coordinated_author_ratio`. The research report may include it. A Telegram author is a
channel, so the measure is channel co-posting and not account behaviour. The report drops
posts with `raw.forwarded_from` and counts them as `telegram_forward`, so a forward does
not create an edge. The seed list in `configs/sources.yaml` has 6 channels. That is below
`min_source_authors` (10), so the Telegram row reports `too_few_authors` until the list
grows.

Result: not measured. No Telegram session credentials exist. Telegram stays research only,
as stated above.

### Result per source

The run covers the eligible records of the corpus. Timestamps are shuffled within each
author, with 200 shuffles and the seed above.

| scope | authors | edges | observed | null mean | null 95% interval | shuffles | empty | null edges | p | p floor | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all eligible | 71 | 350 | 0.5334 | 0.5331 | [0.5285, 0.5380] | 200 | 0 | 349.31 | 0.7512 | 0.0050 | measured |
| youtube | 71 | 350 | 0.5334 | 0.5331 | [0.5285, 0.5380] | 200 | 0 | 349.31 | 0.7512 | 0.0050 | measured |

- **YouTube:** 87 records, 71 channels, 350 edges, 5 communities. 47 of the 71 channels are
  graph nodes. The largest community has
  19 channels. The evidence is 348 `temporal` edges and 2 `cotweet` edges. The null standard
  deviation is 0.0028. The observed modularity is inside the null interval, so
  `exceeds_null` is `false`. The communities are not evidence of coordination.
- **News and GDELT:** excluded with the reason `no_threading`. The counts are 3184 for news
  and 3598 for GDELT.
- **Telegram:** not measured, for the reason above.
- All eligible records are the YouTube records, so the first row equals the second row.

## `coordination_score` — a formula, not a model

The mean of three interpretable terms in [0, 1]:

1. **Community size**, log-scaled against the largest community. A pair is weak
   evidence; a synchronized bloc of forty is not.
2. **Mean edge weight**, normalized against the strongest edge. How *much*
   evidence links this account to its neighbours, not merely that some does.
3. **Participation rate** — the share of this account's own posts that
   contributed to a coordinated edge. One coincidental match among 500 posts is
   not coordination.

Kept transparent on purpose. **This number appears next to a person's account in
a dashboard, and "the model said so" is not an acceptable answer to "why".**

## Complexity

Naive pairwise comparison is O(n²) — at 200k records, 2×10¹⁰ comparisons. Two
bucketing passes avoid it:

1. **Time bucketing** into half-open windows, with each record also placed in the
   next window so a pair straddling a boundary is not missed.
2. **Content bucketing** inside each time bucket — an LSH band on the simhash
   prefix before any Hamming distance is computed; exact-key grouping for URL and
   hashtag evidence.

Complexity becomes O(Σ k²) over bucket occupancies — linear in the corpus for a
fixed posting rate — and is capped at 200,000 pairs per bucket for the
pathological case of one enormous burst, since a swarm that large is already one
obvious component.

## Input filtering

Only sources with real threading and stable author identity, filtered on
**threading availability rather than source name** so a future flat dump is
excluded automatically:

- **Included:** ConvoKit Reddit, Mastodon, YouTube comments. Mastodon is in
  `threading_optional_sources`, so it stays without reply threads.
- **Excluded:** GDELT (its "authors" are outlet domains, not people), news RSS
  (same), Kaggle-flat Reddit (`parent_id = None` honestly), and any record whose
  author is the `__deleted__` sentinel — keeping those would merge every
  tombstoned post into one enormous pseudo-account.

Exclusions are counted by reason and reported.

## Out-of-scope use

- **A community is not a conspiracy.** Fans of one outlet co-share its links all
  day; that is a community and not coordination.
- **`coordination_score` is not evidence about an individual.** It is a property
  of an account's position in a graph, and the graph is built from public
  behaviour that has innocent explanations.
- Not usable where `exceeds_null` is false. Report the null comparison alongside
  any claim built on these communities.

## Known limitations

- **The window is a guess.** 60 minutes is configurable and unvalidated; genuine
  campaigns operate at many timescales.
- **Popularity looks like coordination.** A viral URL links everyone who shared
  it. The participation-rate term dampens this; it does not remove it.
- **Cross-platform edges are weak.** Author identity does not survive across
  platforms, so a person coordinating from two accounts on two services appears
  as two unlinked nodes.
