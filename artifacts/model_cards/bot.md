# Model card — bot classifier

**Module:** `modeling/accounts/bot_clf.py` · **Version:** `v0.1.0`
**Output:** `author_scores.bot_prob` (calibrated) + `author_scores.bot_top_features`

---

## Status

**Not trained on real benchmark data in this repository.** TwiBot-22 and
Cresci-2017 are both behind request forms and neither is present. The full path
— load → campaign-grouped 5-fold CV → out-of-fold calibration →
precision-targeted threshold → SHAP → report → registry — is implemented and
runs on the committed fixtures.

```bash
# after an accepted request form:
python -m modeling.cli train bot
```

## Intended use

One component of a research scorecard, indicating that an account's *posting
behaviour* resembles patterns labelled automated in a Twitter benchmark.

## Out-of-scope use

- **It is not a determination that an account is automated or that a person is a
  bot.** The README states this as a project-level limitation.
- Not for enforcement, suspension, or any action against an account.
- Not on Reddit accounts: the feature intersection is empty there and the scorer
  writes null with reason code `bot:feature_intersection_empty`.

---

## Training data

| dataset | what it is | access |
|---|---|---|
| Cresci-2017 | genuine accounts + 7 distinct labelled bot campaigns | Bot Repository request |
| TwiBot-22 | large graph-based Twitter bot benchmark | request form |

Cresci is preferred when both are present, because it carries campaign structure
and TwiBot does not.

## Split strategy — grouped by campaign, not by account

**This is the decision that determines whether the F1 means anything.** Each
`social_spambots_N` directory in Cresci-2017 is one botnet running one content
template; accounts inside it are near-identical by construction. Grouping by
account puts siblings from the same botnet in train and test, and the model
reports near-perfect F1 for having memorized seven signatures.

So the group key is the **campaign directory**. That is strictly stronger than
grouping by account, and it is the difference between an F1 that means "detects
bots" and one that means "recognizes these seven botnets".

5-fold `StratifiedGroupKFold`. Per-fold numbers are reported alongside the mean,
never instead of it: a mean that hides one catastrophic fold is worse than no
summary.

## Features — the intersection discipline

`modeling/accounts/features.py` declares three tiers:

- **universal** — computable from posts alone, on every platform (16 features:
  posting rate, inter-post interval entropy, hour-of-day entropy, burstiness,
  longest streak, type-token ratio, self-similarity, duplicate rate, URL/hashtag/
  mention rates, …)
- **social_graph** — needs follower/following counts (Mastodon and the Twitter
  benchmarks; **not** ConvoKit Reddit)
- **threading** — needs `parent_id` (ConvoKit Reddit and YouTube; not the
  Kaggle-flat Reddit dump)

**A model trained on forty Twitter features and scored on the twelve this corpus
can compute is not a model — it is a lookup table for a platform we do not
have.** `intersection_features()` computes what both sides support and refuses
outright when the intersection is empty. The tier actually used is recorded here
and in the eval report.

The bot benchmarks ship *profile rows*, not post histories, so only the
social-graph tier is computable on them. That is deliberately all the classifier
uses: a feature the corpus cannot supply is a feature that will not exist at
inference time.

**Missingness is a feature, not a fill.** Every feature whose source can be
absent ships an `*_is_missing` indicator. ConvoKit Reddit has no follower concept
at all, and "no follower data" must not look like "zero followers".

**Leakage warning for whoever adds a weak-label path:** Mastodon exposes a `bot`
flag on accounts. If it is ever used as a weak label, it must be excluded from
the training features, or the model learns to read the flag.

## Operating point — not 0.5

The threshold is chosen from the **precision-recall curve at a stated precision
target** (0.85 by default), and the recall it costs is reported next to it.

**Why.** A false "bot" flag is an accusation about a person. In this product
that costs more than a miss, so the operating point buys precision with recall
and says how much recall it spent. When the target is unreachable on the data,
the report says so and the model is treated as not yet deployable at the
intended precision.

## Calibration

Isotonic (Platt fallback below 200 rows), fitted on the **out-of-fold**
predictions — every score there was produced by a model that did not see that
row, which is the closest thing to a held-out validation set when the labelled
set is too small to carve one.

## Explanations

`author_scores.bot_top_features` carries the top 5 **signed, per-account** SHAP
contributions, and the dashboard's "why is this account flagged" panel reads that
column directly.

If `shap` is unavailable the column falls back to *global* gain importances and
each entry is suffixed `(global)`. That is a different question — "what does the
model use in general" versus "why this account" — and conflating them silently
in a panel labelled "why" would be a quiet lie.

## Known domain shift

**TwiBot-22 and Cresci-2017 are Twitter. This corpus is Mastodon, Reddit and
YouTube. Cross-platform transfer is UNMEASURED and should be assumed degraded.**

Restricting to the social-graph tier bounds the shift without removing it:
follower counts mean different things on a follow-graph platform and on a
federated one, and Reddit has no follower concept at all.

To measure it: hand-label 50 Mastodon accounts, score them, and report the
result. Until that exists, `bot_prob` on non-Twitter accounts is a research
signal and not a finding.

## Error analysis

`artifacts/error_analysis/bot.md`. The taxonomy encodes the false positives that
matter most because they are *people*: low-follower humans with lopsided follow
ratios, genuinely new accounts, prolific humans (journalists, moderators,
hobbyists) whose posting rate alone does not distinguish them from a scheduler,
and organisational accounts that are automated and entirely legitimate.

## Deployment

| | |
|---|---|
| Inference | CPU, milliseconds per account (tree ensemble) |
| Primary | XGBoost; RandomForest reported alongside as a comparison |
| Checkpoint | `estimator.pkl` + `model.json` (feature names, threshold, calibrator) |
