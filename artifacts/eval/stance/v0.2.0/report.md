# stance — evaluation report (v0.2.0)

> **These numbers are not a result.** They were computed on the committed demo fixture, which is shape-faithful and value-meaningless. They demonstrate that the training and evaluation path executes end to end. Reproduce with a real benchmark on disk before citing anything below.

**Headline:** [DEMO FIXTURE -- NOT A RESULT] macro-F1 0.100 [0.038, 0.147], FNC-1's own competition test set (bodies disjoint from train); train/val grouped by body_id; sizes = 50/10/24; seed=20260813

- Split: `FNC-1's own competition test set (bodies disjoint from train); train/val grouped by body_id; sizes = 50/10/24; seed=20260813`
- Test rows: 24
- Positive rate in test: 0.250

## Training data

- **fnc1**: {'dataset': 'fnc1', 'rows': 84, 'groups': 18, 'group_col': 'body_id', 'is_demo': True, 'dropped': {}, 'label_balance': {'unrelated': 0.595, 'discuss': 0.202, 'support': 0.119, 'deny': 0.083}, 'domains': ['competition_test', 'train']}

## Metrics

Accuracy is deliberately absent. On an imbalanced target it rewards predicting the majority class, and PR-AUC is the number that degrades when the model starts crying wolf.

| metric | value | 95% CI |
|---|---|---|
| macro F1 | 0.100 | [0.038, 0.147] |

### Per class

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| support | 0.000 | 0.000 | 0.000 | 6 |
| deny | 0.000 | 0.000 | 0.000 | 6 |
| discuss | 0.250 | 1.000 | 0.400 | 6 |
| unrelated | 0.000 | 0.000 | 0.000 | 6 |

### Confusion matrix

| actual \ predicted | support | deny | discuss | unrelated |
|---|---|---|---|---|
| **support** | 0 | 0 | 6 | 0 |
| **deny** | 0 | 0 | 6 | 0 |
| **discuss** | 0 | 0 | 6 | 0 |
| **unrelated** | 0 | 0 | 6 | 0 |

## Baselines

A baseline exists to answer *what did the expensive model buy*. Overlapping confidence intervals are reported as 'not separable', never as a win.

| baseline | macro F1 | delta | verdict |
|---|---|---|---|
| majority class | 0.100 [0.038, 0.147] | +0.000 | not separable at this test size |
| tf-idf + logreg | 0.100 [0.038, 0.147] | +0.000 | not separable at this test size |
| zero-shot NLI (same base model) | 0.219 [0.098, 0.323] | -0.119 | not separable at this test size |

> **The model does not cleanly clear every baseline.** That is the finding, reported here rather than tuned away.

## Zero-shot NLI baseline

The base model **MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli**, run with its own NLI head and no fine-tuning: macro-F1 **0.2190**.

entailment -> support, contradiction -> deny, neutral -> discuss. The NLI head has no `unrelated` class, because `unrelated` is a retrieval judgement rather than an entailment one, so it is assigned by thresholding the directional mass (`zero_shot.unrelated_below`); that fired on 9 of 24 test pairs. This is the number the fine-tune has to beat to justify its cost.

## Calibration

**`stance_conf` is not calibrated.** `modeling/eval/calibrate.py` fits a binary isotonic or Platt curve and stance is 4-class, so there is nothing honest to fit here without building per-class one-vs-rest calibration first. Nothing in `configs/fusion.yaml` multiplies `stance_conf`, so shipping it uncalibrated costs nothing downstream -- but it must be read as a ranking signal, the way the README's table says to read `toxicity`, and never as a probability.

## Class imbalance

| label | train | test |
|---|---|---|
| `deny` | 1 (2.0%) | 6 (25.0%) |
| `discuss` | 9 (18.0%) | 6 (25.0%) |
| `support` | 4 (8.0%) | 6 (25.0%) |
| `unrelated` | 36 (72.0%) | 6 (25.0%) |

FNC-1 is ~73% `unrelated` and ~1.7% `deny`. A model that never predicts `deny` still scores high accuracy, which is why accuracy is absent from this report. `deny` is also the class the product cares most about -- a post pushing back on a circulating claim -- so read its per-class recall first.

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

Regenerate from saved predictions with `python -m modeling.cli report stance`.
