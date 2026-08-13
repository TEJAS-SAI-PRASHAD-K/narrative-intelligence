# Model card — auxiliary scorers (toxicity, sentiment, emotion, anomaly)

**Module:** `modeling/aux/`
**Version:** toxicity `v0.1.0`, sentiment `v0.1.0`, emotion `v0.1.0`, anomaly `v0.1.0`
**Status:** in use — these fill four of the six scorecard fields in `record_scores`.

These four are **not fine-tuned**. Three are pretrained checkpoints used exactly
as their authors shipped them; one is unsupervised. That is a deliberate choice:
fine-tuning any of them would create a metric this project would then have to
defend, and no measured problem justifies one. The trade is stated plainly
below — off-the-shelf means inheriting someone else's domain and someone else's
biases.

---

## 1. Toxicity — `unitary/toxic-bert`

| | |
|---|---|
| Task | binary toxicity probability, `record_scores.toxicity` |
| Training data | Jigsaw Toxic Comment Classification (Wikipedia talk-page comments) |
| Head used | `toxic` only |
| Calibration | **none claimed** — see below |
| CPU inference | batched, 256 tokens, runs in the Phase 4 worker |
| Precomputed | yes, into Parquet; on-demand scoring is also viable on CPU |

**Why only the `toxic` head.** The model is multi-head (`toxic`, `severe_toxic`,
`obscene`, `threat`, `insult`, `identity_hate`). The heads overlap heavily and
summing them pushes almost every non-neutral comment toward 1.0. The product
wants one number and `toxic` is the head that means what the product means.

**Known demographic bias — this belongs in writing.** Classifiers in this family,
trained on Jigsaw-style annotations, systematically over-flag:

- **African-American English.** Sap et al. (2019, ACL) measured this directly on
  the corpora these models learn from: tweets in AAE are annotated as toxic at
  substantially higher rates than semantically equivalent "mainstream" English,
  and the classifier reproduces the annotation bias faithfully.
- **Identity terms in non-pejorative use.** Sentences that merely *mention* a
  marginalized group score higher than neutral sentences, because those terms
  are over-represented in abusive contexts in training.
- **Profanity without a target.** Swearing is not toxicity; this model does not
  reliably separate them.

**Consequence for this product.** `toxicity` must never be read as "this account
is abusive", and no view should rank accounts by it alone. It is one component
of a scorecard, and it carries a demographic error pattern the score itself
cannot express. The corpus-level distribution across language and topical slices
is reported in `artifacts/eval/aux/` per the fairness requirement.

**On calibration.** The score is a sigmoid output, so it is already in [0, 1],
but it is *not* calibrated against this corpus and no reliability curve is
claimed. Unlike `misinfo_prob` and `bot_prob` — which are calibrated on a
held-out split with a Brier score and a reliability diagram — this one is used
as the vendor shipped it. Phase 4 should treat it as an ordinal signal.

---

## 2. Sentiment — `cardiffnlp/twitter-roberta-base-sentiment-latest`

| | |
|---|---|
| Task | 3-way sentiment → `record_scores.sentiment` + `sentiment_score` |
| Training data | ~124M tweets, fine-tuned on TweetEval |
| Output | argmax label, plus a signed score in [-1, 1] |
| CPU inference | batched, 256 tokens |

**Why this and not an SST-2 model.** SST-2 is movie reviews: long, well-formed,
written to be evaluative. This corpus is Reddit comments, Mastodon toots and
news headlines — short, elliptical, full of mentions and URLs. The Cardiff model
is the closest available domain and ships a real three-way head rather than a
forced binary.

**`sentiment_score` is `p(positive) − p(negative)`, not the argmax probability.**
A post at p(pos)=0.45, p(neu)=0.10, p(neg)=0.45 is genuinely ambivalent and
should read ~0.0, not "positive, confidence 0.45". Phase 4 aggregates this per
author and per narrative, and aggregating argmax confidences would be
meaningless.

**Known failure mode: sarcasm.** "great, another study" reads as positive to
every model in this family. Not fixable here; it is counted in
`artifacts/error_analysis/aux.md` rather than papered over.

---

## 3. Emotion — `j-hartmann/emotion-english-distilroberta-base`

| | |
|---|---|
| Task | 7-way emotion distribution → `record_scores.emotion` |
| Labels | fear, anger, disgust, joy, surprise, sadness, neutral |
| Bucket coverage | **7/7 — no bucket is synthesized** |

The product's seven buckets and the model's seven labels coincide exactly, so
the mapping is one-to-one. This matters because the alternative was tempting:
several emotion models ship five or six labels, and the obvious move is to split
one label across two buckets or route "anticipation" into "surprise". Every such
move invents a number the model never produced. **The rule enforced in
`modeling/aux/emotion.py`: a bucket the model does not cover is emitted as `0.0`
and the gap is logged — never back-filled from a neighbouring label.** If the
configured model is ever swapped for one with fewer labels, the loader warns at
startup and the uncovered buckets are listed.

The full distribution is stored, not just the argmax, because "fearful *and*
angry" is what distinguishes coordinated outrage from ordinary complaint, and an
argmax loses it.

**Known limitation.** The `neutral` class absorbs anything the model cannot
place. A high neutral score means "no strong signal", not "calm".

---

## 4. Anomaly — IsolationForest over behavioural features

| | |
|---|---|
| Task | post-level behavioural anomaly → `record_scores.anomaly_score` |
| Method | `sklearn.ensemble.IsolationForest`, 200 trees, contamination 0.05 |
| Supervision | **none** — there are no anomaly labels |
| Output | within-corpus percentile rank in [0, 1] |

**What the score is, precisely.** 0.95 means "more anomalous than 95% of the
posts in this scoring run". It is deliberately **not** a probability. There is
nothing to calibrate against, and a number that looks like a probability invites
being multiplied by one. Phase 4 must treat it as a rank.

**Corpus-relative, therefore all-or-nothing.** Because the score is a percentile,
it is only meaningful relative to the whole record set. The batch scorer
re-computes it over the full corpus whenever anything is pending, rather than
over the resumed subset — otherwise a record's score would depend on how the
previous run happened to die.

**Features** (13; every one has a stated signal in `modeling/aux/anomaly.py`):
gap to the author's previous post and its missingness, hours from the author's
own median posting hour on a circular clock, total engagement and its
missingness, engagement-per-follower and its missingness, text length,
self-duplication rate, URL/hashtag/mention counts, author post count.

**Null discipline.** Phase 1's rule carries through: `engagement` of `None` means
"not measurable on this platform", `0` means "measured zero". Every
engagement-derived feature ships with an explicit `*_is_missing` indicator, and
missing values are filled with the feature's own median rather than 0. Filling
with 0 would make an unmeasurable metric look like a measured absence — exactly
the conflation Phase 1 refused to make at ingest.

**Authors with fewer than 3 posts are skipped** with reason code
`not_enough_history`. "Unusual for this author" is undefined against a sample of
one, and scoring it anyway would flag every one-off poster.

**How it is evaluated.** Score distribution plus a hand-audit of the top 20,
written into `artifacts/error_analysis/anomaly.md`. **There is no F1 here and
there will not be one** — reporting a supervised metric for an unsupervised model
against labels that do not exist is precisely the kind of number this project
exists to avoid.

---

## Shared properties

**Language policy.** All four are English-only. Non-English text is skipped with
reason code `unsupported_language` and written as `null`, never scored. Records
where Phase 1 left `lang` as `None` (common for short text, where it declined to
guess) are admitted under the `score_unknown_language` setting, and that
assumption is recorded rather than hidden.

**Null discipline.** `null` means "this model did not run on this row" and Phase 4
should render it as *not assessed*. Every null carries a reason code in
`record_scores.skip_reasons`, formatted `<module>:<reason>`. A fabricated `0.0`
would be indistinguishable from a confident negative.

**Per-row provenance.** `record_scores.model_versions` claims only the scorers
that actually produced a value for that row. A row skipped by toxicity does not
claim a toxicity version, so a retrain correctly treats it as unscored.

**Caching.** Keyed by `(model_name, model_version, sha256(text))` under
`data/cache/aux/`. A version bump invalidates the cache, which is what stops a
retrain from silently reusing the previous model's scores.

## Intended use

Components of a research scorecard for studying narrative dynamics across
platforms. Aggregated per narrative and per author by Phase 4.

## Out-of-scope use

- Moderation decisions about individual accounts or posts.
- Any determination that a person is abusive, dishonest, or automated.
- Ranking or enforcement based on `toxicity` alone — see the bias section.
- Non-English content of any kind.
- Treating `anomaly_score` as a probability, or multiplying it into a
  probability product.
