# bot — evaluation report (v0.1.0)

**Headline:** macro-F1 0.705 [0.696, 0.713], PR-AUC 0.908 [0.903, 0.914], out-of-fold, grouped by split_group (3482 groups; train/test = 10217/0/4151; seed=20260813)

- Split: `out-of-fold, grouped by split_group (3482 groups; train/test = 10217/0/4151; seed=20260813)`
- Test rows: 14368
- Positive rate in test: 0.758

## Training data

- **cresci**: {'dataset': 'cresci', 'rows': 14368, 'groups': 3482, 'group_col': 'split_group', 'is_demo': False, 'dropped': {'user_without_id': 0, 'unparseable_created_at': 1000}, 'label_balance': {1: 0.758, 0: 0.242}, 'domains': ['fake_followers', 'genuine', 'social_spambots', 'traditional_spambots']}

## Metrics

Accuracy is deliberately absent. On an imbalanced target it rewards predicting the majority class, and PR-AUC is the number that degrades when the model starts crying wolf.

| metric | value | 95% CI |
|---|---|---|
| macro F1 | 0.705 | [0.696, 0.713] |
| PR-AUC | 0.908 | [0.903, 0.914] |
| ROC-AUC | 0.801 | [0.793, 0.808] |
| Brier | 0.1402 | — |

### Per class

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| human | 0.525 | 0.608 | 0.564 | 3474 |
| bot | 0.868 | 0.825 | 0.846 | 10894 |

### Confusion matrix

| actual \ predicted | human | bot |
|---|---|---|
| **human** | 2112 | 1362 |
| **bot** | 1910 | 8984 |

## Baselines

A baseline exists to answer *what did the expensive model buy*. Overlapping confidence intervals are reported as 'not separable', never as a win.

| baseline | macro F1 | delta | verdict |
|---|---|---|---|
| majority class | 0.431 [0.429, 0.433] | +0.274 | beats |
| logreg on follower_following_ratio only | 0.647 [0.624, 0.668] | +0.058 | beats |

## Calibration

Phase 4 multiplies these scores together, so they must be probabilities rather than arbitrary decision values.

- isotonic calibration on 14368 rows: Brier 0.4345 -> 0.1402 (improved)

| predicted | observed | n |
|---|---|---|
| 0.000 | 0.000 | 1 |
| 0.475 | 0.475 | 4021 |
| 0.529 | 0.529 | 735 |
| 0.666 | 0.666 | 920 |
| 0.746 | 0.746 | 358 |
| 0.841 | 0.841 | 1377 |
| 0.943 | 0.943 | 6956 |

## Cross-validation

5-fold grouped cross-validation, grouped by split_group (3482 groups; train/test = 10217/0/4151; seed=20260813).

- macro F1: **0.4162 ± 0.1869** across folds
- PR-AUC: **0.8648 ± 0.1372**
- per fold: [0.2486, 0.2958, 0.6791, 0.3113, 0.5464]
- worst fold: 0.2486

The per-fold numbers are given alongside the mean deliberately: a mean that hides one catastrophic fold is worse than no summary at all.

## Operating point

Threshold **0.500**, chosen from the precision-recall curve at a precision target of 0.85 — not 0.5.

- precision at this threshold: 0.868
- recall at this threshold: 0.825

**Why a precision target.** A false 'bot' flag is an accusation about a person. In this product that costs more than a miss, so the operating point buys precision with recall, and the recall it costs is stated rather than buried.

## Comparison estimator (random_forest)

macro-F1 0.216 [0.204, 0.228], PR-AUC 0.939 [0.927, 0.951], fold 0, grouped by split_group (3482 groups; train/test = 10217/0/4151; seed=20260813)

Reported so the choice of XGBoost as primary is an observation rather than an assumption.

## Feature explanations

Per-account SHAP contributions, top 5 per account, written into `author_scores.bot_top_features`. The dashboard's "why is this account flagged" panel reads that column directly.

How often each feature appears in an account's top-5:

| feature | accounts |
|---|---|
| `post_count` | 14159 |
| `account_age_days` | 12649 |
| `following` | 11573 |
| `posts_per_account_day` | 11546 |
| `followers` | 10743 |
| `follower_following_ratio` | 10582 |
| `account_age_is_missing` | 588 |

## Cross-platform transfer

Trained on **cresci**, which is Twitter data. This project's corpus is Mastodon, Reddit and YouTube.

**Cross-platform transfer is unmeasured and should be assumed degraded.** The feature set is deliberately restricted to the social-graph tier that Mastodon also supplies, which bounds the shift but does not remove it: follower counts mean different things on a follow-graph platform and on a federated one, and Reddit has no follower concept at all.

To measure it: hand-label 50 Mastodon accounts, score them with this model, and report the result here. Until that exists, `bot_prob` on non-Twitter accounts is a research signal and not a finding.

## Reproducibility

- seed: `20260813`
- device: `mps`
- input manifest hash: `b6ba18d6285f05da`
- languages: `['en']`

<details><summary>library versions</summary>

```json
{
  "python": "3.13.7",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "numpy": "2.5.2",
  "scipy": "1.18.0",
  "pandas": "3.0.5",
  "pyarrow": "25.0.1",
  "scikit-learn": "1.9.0",
  "xgboost": "3.4.0",
  "shap": "0.52.0",
  "torch": "2.13.0",
  "transformers": "5.15.0",
  "sentence-transformers": "5.7.0",
  "timm": "1.0.28",
  "networkx": "3.6.1",
  "anthropic": "0.121.0"
}
```

</details>

Regenerate from saved predictions with `python -m modeling.cli report bot`.
