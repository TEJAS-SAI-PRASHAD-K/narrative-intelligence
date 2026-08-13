# misinfo — evaluation report (v0.1.0)

> **These numbers are not a result.** They were computed on the committed demo fixture, which is shape-faithful and value-meaningless. They demonstrate that the training and evaluation path executes end to end. Reproduce with a real benchmark on disk before citing anything below.

**Headline:** [DEMO FIXTURE -- NOT A RESULT] macro-F1 0.908 [0.821, 0.978], PR-AUC 0.988 [0.959, 1.000], grouped by group_id (243 groups; train/val/test = 224/31/55; seed=20260813)

- Split: `grouped by group_id (243 groups; train/val/test = 224/31/55; seed=20260813)`
- Test rows: 55
- Positive rate in test: 0.418

## Training data

- **liar**: {'dataset': 'liar', 'rows': 75, 'groups': 8, 'group_col': 'speaker', 'is_demo': True, 'dropped': {'half_true_dropped': 15}, 'label_balance': {1: 0.6, 0: 0.4}, 'domains': ['democrat', 'independent', 'none', 'republican']}
- **fakenewsnet**: {'dataset': 'fakenewsnet', 'rows': 95, 'groups': 95, 'group_col': 'claim_id', 'is_demo': True, 'dropped': {}, 'label_balance': {0: 0.558, 1: 0.442}, 'domains': ['gossipcop', 'politifact']}
- **coaid**: {'dataset': 'coaid', 'rows': 140, 'groups': 140, 'group_col': 'claim_id', 'is_demo': True, 'dropped': {'cross_wave_duplicate': 0}, 'label_balance': {0: 0.614, 1: 0.386}, 'domains': ['claim', 'news']}

## Metrics

Accuracy is deliberately absent. On an imbalanced target it rewards predicting the majority class, and PR-AUC is the number that degrades when the model starts crying wolf.

| metric | value | 95% CI |
|---|---|---|
| macro F1 | 0.908 | [0.821, 0.978] |
| PR-AUC | 0.988 | [0.959, 1.000] |
| ROC-AUC | 0.990 | [0.968, 1.000] |
| Brier | 0.0381 | — |

### Per class

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| not-misinformation | 0.966 | 0.875 | 0.918 | 32 |
| misinformation-like | 0.846 | 0.957 | 0.898 | 23 |

### Confusion matrix

| actual \ predicted | not-misinformation | misinformation-like |
|---|---|---|
| **not-misinformation** | 28 | 4 |
| **misinformation-like** | 1 | 22 |

## Baselines

A baseline exists to answer *what did the expensive model buy*. Overlapping confidence intervals are reported as 'not separable', never as a win.

| baseline | macro F1 | delta | verdict |
|---|---|---|---|
| majority class | 0.368 [0.312, 0.415] | +0.540 | beats |
| tf-idf + logreg | 0.927 [0.852, 0.982] | -0.019 | not separable at this test size |

> **The model does not cleanly clear every baseline.** That is the finding, reported here rather than tuned away.

## Calibration

Phase 4 multiplies these scores together, so they must be probabilities rather than arbitrary decision values.

- platt calibration on 31 rows: Brier 0.0756 -> 0.0570 (improved)
- Note: fell back to Platt: 31 validation rows is below the 200-row floor for isotonic regression

| predicted | observed | n |
|---|---|---|
| 0.000 | 0.000 | 15 |
| 0.495 | 0.000 | 1 |
| 0.538 | 0.667 | 3 |
| 0.643 | 0.667 | 3 |
| 0.996 | 1.000 | 9 |

## Per-benchmark breakdown

The three benchmarks are three different problems: politicians' statements, news headlines, and COVID-era health claims. A single averaged F1 over their union hides which one the model actually learned.

| benchmark | n | macro F1 | positive rate |
|---|---|---|---|
| coaid | 25 | 1.000 [1.000, 1.000] | 0.400 |
| fakenewsnet | 23 | 1.000 [1.000, 1.000] | 0.435 |
| liar | 7 | too small to report | — |

## Cross-domain transfer (PolitiFact -> GossipCop)

Trained on PolitiFact (political fact-checks), tested on GossipCop (celebrity gossip) — a genuine domain shift inside one benchmark.

Reported here with the TF-IDF baseline rather than the fine-tune, because re-fine-tuning for one table costs a full training run; the baseline's drop measures the shift itself, which is the quantity of interest.

- in-domain reference: see the main table above
- PolitiFact -> GossipCop, TF-IDF baseline: 1.000 [1.000, 1.000]
- test rows: 50

**Expect the fine-tune to drop similarly.** House style is memorizable; the underlying task is not.

## Corpus transfer

**Not yet measured.** Benchmark F1 is not production accuracy and must not be quoted as if it were: LIAR is politicians' statements, FakeNewsNet is news headlines, and this project's corpus is Reddit comments, Mastodon toots and GDELT article metadata.

To measure the gap:

```bash
python -m modeling.cli sample-for-labelling misinfo --n 100
```

That writes `artifacts/hand_labels/misinfo_corpus_sample.csv` with a blank `label` column. Fill it in by hand, rerun `modeling evaluate misinfo`, and this section becomes a table. Until then, the honest statement is that the transfer gap is **unmeasured and expected to be large**.

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

Regenerate from saved predictions with `python -m modeling.cli report misinfo`.
