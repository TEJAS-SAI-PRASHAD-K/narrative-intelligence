# Model card — bot classifier

**Module:** `modeling/accounts/bot_clf.py` · **Version:** `v0.1.0`
**Output:** `author_scores.bot_prob` (calibrated) + `author_scores.bot_top_features`

---

## Status

**Trained on real Cresci-2017 data.** 14,368 labelled accounts, 5-fold grouped
cross-validation, isotonic calibration on out-of-fold predictions.

## Headline numbers — read the second one

| | macro F1 | PR-AUC |
|---|---|---|
| pooled out-of-fold | 0.705 [0.696, 0.713] | 0.908 [0.903, 0.914] |
| **mean across folds** | **0.416 ± 0.187** | 0.865 ± 0.137 |
| worst fold | 0.249 | 0.628 |

Per-fold macro F1: `[0.249, 0.296, 0.679, 0.311, 0.546]`

**The gap between 0.705 and 0.416 is the finding, not a rounding artefact.**

The pooled figure is computed over every out-of-fold prediction at once, so
folds where the held-out botnet happened to resemble the training campaigns
average away the folds where it did not. The per-fold mean keeps them separate,
and separate is the honest view: **this model does not reliably generalise to a
botnet it has never seen.** Two of five held-out campaigns scored below 0.32.

Quote the per-fold mean. The pooled number describes an averaging artefact, not
a capability.

## Intended use

One component of a research scorecard, indicating that an account's *posting
behaviour* resembles patterns labelled automated in a Twitter benchmark.

## Out-of-scope use

- **Not a determination that an account is automated or that a person is a
  bot.** With a worst fold of 0.249 this is nowhere near a basis for any claim
  about an individual.
- Not for enforcement, suspension, or ranking anyone's account.
- Not on Reddit accounts: the feature intersection is empty there and the scorer
  writes null with reason code `bot:feature_intersection_empty`.

---

## Training data

Cresci-2017: 14,368 accounts across 9 campaign directories — 3,474 genuine and
10,894 bot across 8 distinct campaigns (`fake_followers`, `social_spambots_1-3`,
`traditional_spambots_1-4`). Requested through the Bot Repository.

TwiBot-22 is supported as a fallback but was not used: only its `label.csv` was
obtainable, and `user.json` runs to tens of gigabytes.

## Split strategy — the hybrid, and why neither extreme works

**Grouping by account is wrong.** Each spambot directory is one botnet running
one content template, so its accounts are near-identical by construction.
Account-level grouping puts siblings in train and test, and the model reports
near-perfect F1 for having memorised eight signatures.

**Grouping by campaign alone is impossible.** In this dataset the label is a
*deterministic function of the campaign*: one genuine directory, eight bot
directories, every group 100% one class. `StratifiedGroupKFold` then cannot
balance a fold, and some fold's training set arrives single-class — which
aborted XGBoost with `Invalid classes inferred from unique values of y`, and via
the CLI took the interpreter down with SIGSEGV and no message at all.

So the key is `split_group`: **campaign for bots, account for humans.** The
leakage worth preventing is the shared bot template; genuine accounts are
independent individuals with no template to share. Each test fold then holds out
one or two entire botnets plus a random sample of humans:

| fold | held-out campaigns | train / test |
|---|---|---|
| 0 | social_spambots_2 | 10217 / 4151 |
| 1 | fake_followers | 10322 / 4046 |
| 2 | traditional_spambots_2, _4 | 12445 / 1923 |
| 3 | traditional_spambots_1, _3 | 12270 / 2098 |
| 4 | social_spambots_1, _3 | 12218 / 2150 |

That structure is what produces the fold variance above, and it is the point:
each fold asks "does this detect a campaign it has never seen?", and the answer
is often no.

## Features — the intersection discipline

8 features, all from the social-graph tier: `followers`, `following`,
`follower_following_ratio`, `post_count`, `posts_per_account_day`,
`account_age_days`, plus `followers_is_missing` and `account_age_is_missing`.

The bot benchmarks ship *profile rows*, not post histories, so the behavioural
tier (interval entropy, burstiness, self-similarity) cannot be computed on them.
That tier exists in `features.py` and is used for corpus-side scoring, but a
model cannot be trained on features its benchmark lacks.

**A known data defect that did not become a leak.** Cresci's
`genuine_accounts/users.csv` has a different column order from the other
campaigns, so `created_at` fails to parse for all 3,474 genuine accounts and
`account_age_is_missing` is 1 for exactly the human class. That is a textbook
label proxy. It did **not** dominate: `account_age_is_missing` appears in only
588 of 14,368 accounts' top-5 SHAP contributions (4%), against `post_count` at
14,159 (99%). The model is keying on behaviour, not on the missingness flag.
Worth re-checking after any change to the loader.

## Calibration

Isotonic on the out-of-fold predictions — every score there came from a model
that did not see that row. Brier **0.4345 → 0.1402**, a large improvement,
which is expected: raw tree-ensemble scores are badly calibrated by default.

## Operating point

Threshold **0.500** → precision **0.868**, recall **0.825**. The 0.85 precision
target is met.

The threshold is chosen from the PR curve at a stated precision target, not
fixed at 0.5 by convention; that it landed on 0.5 here is a coincidence of this
data. A false "bot" flag is an accusation about a person, so the operating point
buys precision with recall and reports what recall it spent.

## Baselines

| baseline | macro F1 | delta | verdict |
|---|---|---|---|
| majority class | 0.431 [0.429, 0.433] | +0.274 | beats |
| logreg on `follower_following_ratio` alone | 0.647 [0.624, 0.668] | +0.058 | beats |

The model clears both on the pooled metric. Note that a **single feature** gets
to 0.647 — the ensemble buys 0.058 macro-F1 over one logistic regression on the
follower/following ratio. That is a real gain and a modest one.

## Explanations

`author_scores.bot_top_features` carries the top 5 signed per-account SHAP
contributions. If an entry is suffixed `(global)`, SHAP was unavailable and the
values are global gain importances — a different question ("what does the model
use in general" vs "why this account"), and it must not be presented as
per-account in a panel labelled "why".

## Known domain shift

**Cresci-2017 is Twitter. This corpus is Mastodon, Reddit and YouTube.
Cross-platform transfer is UNMEASURED and should be assumed degraded.**

Restricting to the social-graph tier bounds the shift without removing it:
follower counts mean different things on a follow-graph platform and a federated
one, and Reddit has no follower concept at all.

To measure it: hand-label 50 Mastodon accounts, score them, report the result.
Until then, `bot_prob` on non-Twitter accounts is a research signal, not a
finding — and given the fold variance above, a weak one.

## Error analysis

`artifacts/error_analysis/bot.md`. The false positives that matter are *people*:
low-follower humans with lopsided follow ratios, genuinely new accounts,
prolific humans (journalists, moderators, hobbyists) whose posting rate alone
does not distinguish them from a scheduler, and organisational accounts that are
automated and entirely legitimate.

Note the per-class numbers: **human recall is 0.608** — the model misclassifies
39% of genuine accounts as bots at the chosen threshold. In a product that
surfaces this to users, that is the number to design around.

## Deployment

| | |
|---|---|
| Inference | CPU, milliseconds per account |
| Primary | XGBoost; RandomForest reported alongside |
| Checkpoint | `estimator.pkl` + `model.json` (feature names, threshold, calibrator) |
| Training time | ~14 seconds, 5 folds, CPU |

**Import-order constraint:** `xgboost` must be imported before `torch`. On macOS
both ship their own OpenMP runtime and loading torch first makes the first
`fit()` segfault with no traceback. `modeling/__init__.py` claims the runtime at
package import; do not remove that.
