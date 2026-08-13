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

## The top-20 audit — to be completed by hand

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
| explicable | _to be filled_ |
| artefact | _to be filled_ |
| uninteresting | _to be filled_ |

**A high artefact rate is the signal to act on.** The likely culprits are the
missingness indicators: `gap_is_missing` fires on every author's first post, and
`engagement_is_missing` fires for entire platforms at once. If the top of the
distribution is dominated by those, the indicator features are being read as
anomaly signal rather than as context, and they should be excluded from the
forest while remaining available to downstream consumers.

## Written analysis

_To be written after the audit._
