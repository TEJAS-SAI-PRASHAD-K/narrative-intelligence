# deepfake — evaluation report (v0.2.0-pretrained)

**Headline:** macro-F1 0.564 [0.485, 0.638], PR-AUC 0.885 [0.826, 0.932], DFDC as a pure test set at video grain, split in half by source_video: 189 videos fit the calibrator and the operating point, 191 held-out videos are reported (from 3745 face crops; seed=20260813). Nothing was trained.

- Split: `DFDC as a pure test set at video grain, split in half by source_video: 189 videos fit the calibrator and the operating point, 191 held-out videos are reported (from 3745 face crops; seed=20260813). Nothing was trained.`
- Test rows: 191
- Positive rate in test: 0.791

## Training data

- **dfdc**: {'dataset': 'dfdc', 'rows': 3745, 'groups': 381, 'group_col': 'source_video', 'is_demo': False, 'dropped': {}, 'label_balance': {1: 0.797, 0: 0.203}, 'domains': ['crops_fake', 'crops_real']}

## Metrics

Accuracy is deliberately absent. On an imbalanced target it rewards predicting the majority class, and PR-AUC is the number that degrades when the model starts crying wolf.

| metric | value | 95% CI |
|---|---|---|
| macro F1 | 0.564 | [0.485, 0.638] |
| PR-AUC | 0.885 | [0.826, 0.932] |
| ROC-AUC | 0.665 | [0.567, 0.756] |
| Brier | 0.2025 | — |

### Per class

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| authentic | 0.297 | 0.475 | 0.365 | 40 |
| manipulated | 0.835 | 0.702 | 0.763 | 151 |

### Confusion matrix

| actual \ predicted | authentic | manipulated |
|---|---|---|
| **authentic** | 19 | 21 |
| **manipulated** | 45 | 106 |

## Baselines

A baseline exists to answer *what did the expensive model buy*. Overlapping confidence intervals are reported as 'not separable', never as a win.

| baseline | macro F1 | delta | verdict |
|---|---|---|---|
| majority class | 0.442 [0.423, 0.457] | +0.122 | beats |

## Calibration

Phase 4 multiplies these scores together, so they must be probabilities rather than arbitrary decision values.

- none calibration on 189 rows: Brier 0.1823 -> 0.1823 (worsened)
- Note: fell back to Platt: 189 validation rows is below the 200-row floor for isotonic regression; platt calibration was REJECTED: it mapped higher raw scores to lower probabilities, inverting the model. Raw scores are served uncalibrated instead. This is a signal-strength problem, not a calibration one -- the fit had too little to orient itself by.
- **Calibration made the Brier score worse.** Reported rather than hidden; the usual cause is a validation split too small for the chosen method.

| predicted | observed | n |
|---|---|---|
| 0.415 | 1.000 | 1 |
| 0.607 | 1.000 | 1 |
| 0.868 | 1.000 | 4 |
| 0.989 | 0.809 | 183 |

## Operating point

Threshold **0.990**, chosen from the precision-recall curve at a precision target of 0.85 — not 0.5.

- precision at this threshold: 0.850
- recall at this threshold: 0.773

**Why a precision target.** A false 'bot' flag is an accusation about a person. In this product that costs more than a miss, so the operating point buys precision with recall, and the recall it costs is stated rather than buried.

## This is cross-dataset transfer, and that is the point

**dima806/deepfake_vs_real_image_detection** was trained by someone else, on someone else's data, and is evaluated here on DFDC with no adaptation. That is cross-dataset transfer, which is the single hardest regime in deepfake detection: published detectors routinely lose most of their headline accuracy when the manipulation method, codec and capture conditions change. A weak number here is the expected result, not a bug in this pipeline.

Test set: 191 videos, 151 manipulated / 40 authentic (79% positive), decision threshold 0.9900317430496216. Macro-F1 **0.5640**.

**How to read this.** If the detector does not clear the majority-class baseline, it has measured nothing and the module should stay off rather than ship a `deepfake_prob` that is noise -- `configs/fusion.yaml` thresholds that column at 0.50 and a noisy value propagates into the risk score. The detector also targets a different manipulation family than DFDC's face swaps; the FF++-trained alternative named in `configs/models.yaml` is the closer domain match, at the cost of `trust_remote_code`.

## Why no training happened

FaceForensics++ is 17 GB behind a signed agreement and is not on disk. The DFDC copy that is on disk is pre-extracted face crops with the `original` field dropped, so a fake cannot be tied to the real clip it came from; training on it risks identity leakage, because DFDC swaps faces between actors recorded in the same sessions. Mounting someone else's trained detector and testing it here sidesteps both problems: nothing is fitted, so there is no split to leak across, and the module's stated role -- a cross-dataset generalisation check -- is exactly what this measures.

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

Regenerate from saved predictions with `python -m modeling.cli report deepfake`.
