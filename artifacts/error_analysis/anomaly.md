# Post-level anomaly — audit

Unsupervised. **There is no F1 here and there will not be one** — reporting a
supervised metric for an unsupervised model against labels that do not exist is
precisely the kind of number this project exists to avoid.

The evaluation is a score distribution plus a hand-audit of the top 20.

## What the score is

`anomaly_score` is the **within-corpus percentile rank** of the IsolationForest's
anomaly score. 0.95 means "more anomalous than 95% of the posts in this scoring
run". It is deliberately *not* a probability: there is nothing to calibrate
against, and a number that looks like a probability invites being multiplied by
one.

Because it is a percentile, it is only meaningful relative to the whole record
set. The batch scorer recomputes it over the full corpus whenever anything is
pending rather than over the resumed subset — otherwise a record's score would
depend on how the previous run happened to die.

## Coverage

Authors with fewer than 3 posts are skipped with reason code
`not_enough_history`. "Unusual for this author" is undefined against a sample of
one, and scoring it anyway would flag every one-off poster. On the current corpus
this is why `anomaly_score` covers roughly half the records — most of the news
and GDELT "authors" are outlet domains with one or two articles each.

## The top-20 audit — completed by hand

```python
from modeling.aux.anomaly import AnomalyScorer
from modeling.io import CorpusReader

records = CorpusReader().records()
scorer = AnomalyScorer()
scorer.score(records)
scorer.audit_frame(20)
```

For each of the top 20, record whether the flag is:

- **explicable** — the features that drove it describe something genuinely
  unusual (a burst at 4am from an account that never posts at night)
- **artefact** — the flag is driven by a data property, not behaviour (a missing
  engagement indicator, an author whose first post has no gap to a previous one)
- **uninteresting** — unusual but not meaningful (a long post from someone who
  usually writes short ones)

| verdict | count |
|---|---|
| explicable | 0 |
| artefact | 10 |
| uninteresting | 10 |

**A high artefact rate is the signal to act on.** The likely culprits are the
missingness indicators: `gap_is_missing` fires on every author's first post, and
`engagement_is_missing` fires for entire platforms at once. If the top of the
distribution is dominated by those, the indicator features are being read as
anomaly signal rather than as context, and they should be excluded from the
forest while remaining available to downstream consumers.

The prediction was right about the mechanism and wrong about which indicator.
`engagement_is_missing` is `0.0` for all twenty. The indicator doing the work is
`engagement_per_follower_is_missing`, which is `1.0` for all twenty, with its
paired value column `log_engagement_per_follower` pinned at a constant `0.0`.
`gap_is_missing` fires on only 4 of 20.

### What the top 20 actually is

Every record in the top 20 is Mastodon. Not a majority — all of them. Four
platforms are absent from the head of the distribution entirely.

Grouping by `log_author_post_count` and `self_duplicate_rate`, the twenty posts
come from about **four accounts**:

| group | rows | author post count | self-dup rate | hashtags | verdict |
|---|---|---|---|---|---|
| A | 1–4 | 5 (`ln 5 = 1.609`) | 0.250 = 1/4 | 2–3 | artefact |
| B | 5–10 | 4 (`ln 4 = 1.386`) — ≥2 accounts, since one 4-post author cannot supply 6 rows | 0.333 = 1/3 | 4–15 | artefact |
| C | 11–20 | 19 (`ln 19 = 2.944`) | 0.056 = 1/18 | 7–17 | uninteresting |

Group C is one account occupying half the top 20. The detector did not find ten
anomalies there; it found one account and reported it ten times.

### Reading the driving features

- `engagement_per_follower_is_missing = 1.0`, 20/20. Follower counts are absent
  for Mastodon, so this is constant across the platform. **Artefact, corpus-wide.**
- `log_engagement_per_follower = 0.0`, 20/20. The fill value, constant within the
  same subpopulation. Redundant with the indicator and equally splittable.
- `hours_from_own_median` — 18 of 20 are within 4 hours of the author's own median
  posting hour; only two exceed it (10.5 and 7.0). The doc's own example of an
  explicable flag is *the 4am burst from an account that never posts at night*,
  and that feature is contributing almost nothing at the top of the distribution.
- `gap_is_missing = 1.0` on 4 rows, each carrying `log_gap_seconds = 8.217978`
  exactly — the fill constant, ~1.03 hours. Same fill-as-signal pattern.
- `self_duplicate_rate` — 0.250 and 0.333 look high but are 1/4 and 1/3 from
  authors with four and five posts. At that sample size the feature is noise.
  The stable value in the table (0.056, from 19 posts) is the *low* one.
- `hashtag_count` — 2 to 17, and the genuinely distinctive column. But it is
  absolute, not author-relative: nothing in the feature set asks whether *this*
  post is hashtag-heavy for *this* account.
- `mention_count = 0.0`, 20/20. Contributing nothing here.

### Why no verdict is `explicable`

Group A/B are artefact: authors with four or five posts, where every
within-author feature (`self_duplicate_rate`, `hours_from_own_median`, the gap
distribution) is estimated from three or four observations. The 3-post coverage
floor admits them, but a floor that permissive produces per-author statistics
that are noise, and noise isolates easily.

Group C is uninteresting rather than explicable, on the evidence available. The
account posts long, heavily-hashtagged toots — consistently, at its usual hours
(`hours_from_own_median` 0–4), with low self-duplication. The flag says "this
account is unlike other accounts", not "this post is unlike this account's other
posts". That is a stable trait, not an event.

It is the one verdict I would revisit with more information. A Mastodon account
posting 19 times with 7–17 hashtags per post is a plausible amplifier, and if the
text turns out to be coordinated or repetitive, group C becomes explicable and
the detector deserves credit for it. What would settle it: the post text, the
account handle, and whether the 19 posts cluster into one narrative.

## Written analysis

The audit answers the question it was set — *are the top anomalies just first
posts by new authors?* — with a no, and then fails the model on a wider charge.
Only 4 of 20 have `gap_is_missing`. But all 20 are one platform, selected by a
missingness indicator that is constant for that platform.

The mechanism is structural, not a tuning problem. IsolationForest scores by how
few splits it takes to separate a point. A binary indicator that is 1 for one
platform and 0 for the rest is the cheapest split available in the entire feature
space, and every record on the minority side of it gets isolated immediately.
Missingness indicators are close to worst-case inputs for this estimator. The
same applies to their paired fill values: `log_engagement_per_follower` is a
constant `0.0` wherever the indicator is 1, and `log_gap_seconds` is a constant
`8.217978` wherever `gap_is_missing` is 1. Dropping the indicators while leaving
the fill columns in just relocates the same split one column over. Both halves
have to come out of the forest together.

The second problem is that the module's stated contract is not what the features
implement. The coverage note frames the score as *unusual for this author*, and
three features are author-relative (`hours_from_own_median`, `log_gap_seconds`,
`self_duplicate_rate`). The rest — `log_text_length`, `url_count`,
`hashtag_count`, `mention_count`, `log_engagement` — are absolute. So the forest
compares accounts against the corpus, not posts against their author's baseline,
and the author-relative features are visibly not driving the head of the
distribution: 18 of 20 flagged posts are within four hours of their own author's
median hour. Group C follows from this directly. One account with an unusual
absolute profile has *all* of its posts flagged, because nothing in the feature
vector varies enough between them to separate one post from the next.

Three changes, in order:

1. **Drop the missingness indicators and their fill columns from the forest.**
   Keep both in the output contract for downstream consumers, as the doc already
   proposes. This is what unpins the top 20 from Mastodon.
2. **Standardise the absolute features within platform** before fitting, or fit
   per platform. Hashtag counts, text lengths, and engagement are not comparable
   across a Mastodon toot, a Reddit comment, and a GDELT article stub, and a
   single forest over the union will keep rediscovering the platform boundary
   through whichever column survives step 1.
3. **Raise the coverage floor above 3 posts** for the author-relative features —
   10 is a defensible starting point — and deduplicate the audit frame by author,
   so twenty rows means twenty findings rather than four.

Re-audit after step 1 alone. That change should move the composition of the top 20
more than the other two combined, and it is worth seeing what surfaces once the
platform indicator is no longer available as a free split.

### Gaps in the audit frame

`audit_frame` returns features only. Four columns would make the next audit
considerably less inferential:

- `anomaly_score` — absent, so there is no way to see whether the top 20 is a
  cliff or a flat plateau, or where these sit against the rest of the corpus.
- `platform` — inferable from the id prefix here, but only because the answer
  turned out to be uniform.
- `author_id` — author identity had to be reconstructed by looking for collisions
  in `log_author_post_count` and `self_duplicate_rate`. That worked, but it is
  guesswork, and it is the single most important grouping in the frame.
- per-record feature attribution, or at least the top contributing splits. Every
  causal claim above is inferred from which columns are constant or extreme, not
  from the model.

### Housekeeping

The same 2 cross-partition duplicate record ids reported by the clustering run
appear here. Still a cross-partition dedupe pass that does not exist.