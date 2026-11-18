# Model card — Laya misinformation scorer

**Module:** `modeling/text/laya_clf.py` · **Version:** `v0.2.1`
**Status:** fine-tuned and compared. The Hindi gate passes. The Hindi scoring stage is built. It scores titles only.

## Result

The comparison run of 2026-11-09 fine-tuned two checkpoints of
`convaiinnovations/laya`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
Training used the same LIAR, FakeNewsNet and CoAID split as the `misinfo` module. The
full report is [docs/results/factcheck-comparison.md](../../docs/results/factcheck-comparison.md).

| model | threshold | validation recall | test macro F1 | difference to TF-IDF | decision |
|---|---|---|---|---|---|
| Laya English fine-tune | 0.9734 | 0.318 | 0.683 [0.670, 0.696] | +0.079 [0.065, 0.095] | passes the baseline rule |
| Laya multilingual fine-tune | 0.9770 | 0.165 | 0.565 [0.552, 0.578] | -0.039 [-0.054, -0.022] | do not ship |
| Laya multilingual base (control) | 0.9041 | 0.004 | 0.414 [0.410, 0.419] | -0.189 [-0.202, -0.176] | control, never ships |

The base checkpoint did not reach precision 0.85 on validation. Its best precision
was 0.444.

**Decision for English: Laya does not enter the scoring pipeline.** The English fine-tune passes
the rule against TF-IDF. But `misinfo` v0.2.0 has a higher macro F1 of 0.753 and a
higher fact-check lift. The two candidates cover the same language, so a second
scorer adds cost and no measured gain.

## Hindi

Hindi is the one language that only Laya can score here. The gate needs at least 50
Hindi claims and a title lift interval with a lower limit above 1. The set grew from
13 to 1016 to 4180 Hindi claims. The sitemap backfill read the whole Vishvas News
archive inside the 2022 window, and then we froze the manifest again.

| measure, Hindi titles | Laya multilingual fine-tune |
|---|---|
| claims scored | 4180 |
| recall on known-false claims | 0.122 [0.113, 0.132] |
| flag rate on 431 corpus titles | 0.070 [0.046, 0.093] |
| lift | 1.760 [1.286, 2.630] |

**The gate passes.** The result has clear limits:

- The lower limit is 1.286. With 1016 claims it was 1.116. The interval is much
  narrower now, but the corpus has only 431 Hindi titles.
- On full text, the lift is 0.476 [0.401, 0.573] and the upper limit is below 1. The
  model flags Hindi corpus articles (0.258) more often than debunked claims (0.122).
  The full-text figures are not like for like with the title figures. The stage scores
  titles only. Do not use the score on full text.
- Recall is 0.122, so the model does not flag most debunked claims at this threshold.
- Almost all Hindi claims come from one publisher. Vishvas News gives 4173 of the 4180
  claims, which is 99.8 percent. The result reflects the choice of claims by that publisher.
- The claims are short fact-check statements, and the corpus has news titles. A part
  of the lift can come from this difference in text form.
- The English recall on the fact-check set is 0.060 [0.020, 0.110] for `laya_en` and
  0.050 [0.010, 0.100] for `laya_multi`. The English result does not prove any transfer to Hindi.

## Real run of the stage

On 2026-11-10 the stage scored 156 Hindi titles on the GPU host. No raw value fell
outside the fitted range, and the stage wrote 24 distinct calibrated values. The mean
of the calibrated Hindi values was 0.458, and the mean of the English `misinfo` values
was 0.451. These means do not measure accuracy, because the corpus has no labels.

The calibrator had a raw ROC-AUC of 0.854 on 3,558 English validation rows. Brier went
from 0.173 to 0.134. The calibration error of 0 after the fit is an in-sample value.

## Calibration of the raw model

The vendor fit of the temperature ends on its upper clamp of 5 for both fine-tunes. Issue 33
asked why, and whether the clamp is right. This section gives the answer.

**Decision: the clamp at 5 stays and `laya.temperature_refit` stays `false`.** The isotonic
calibrator does the calibration. On the stored scores, ECE prob falls from 0.159 to 0.009
for the multilingual fine-tune, and from 0.151 to 0.009 for the English fine-tune. Both values
are out of fold. The guard difference is below zero for both, with an interval that excludes
zero (see "The calibrator guard"). ECE does not rise after calibration.

The raw score is not a probability. The mean confidence of the raw score is 0.966 and 0.962
(multilingual and English, diagnosis run). The raw ECE prob is 0.159 and 0.151 on the stored
scores. Use the calibrated value of the stage.

### Why the optimum lies past 5

The temperature that minimizes the validation NLL is inside the wide search of 0.05 to 100.
It is not on a bound. The cause is overconfident logits. It is not the fit objective, and it
is not the grouping by question type. The checkpoint holds `per_type` temperatures of
`[1.0, 1.0, 5.0]`, and only the `noul` type is on the clamp.

| measure | multilingual | English |
|---|---|---|
| optimum temperature on validation | 11.846 | 11.408 |
| 5th to 95th percentile of fits on 400 rows | 10.472 to 13.678 | 10.119 to 13.066 |
| share of 400-row fits above the vendor clamp of 5 | 1.000 | 1.000 |
| NLL at the optimum | 0.460 | 0.469 |
| NLL at the checkpoint temperature of 5 | 0.657 | 0.648 |
| logit gap, 5th percentile | -22.467 | -18.250 |
| logit gap, 95th percentile | 18.938 | 18.125 |

Every fit on a 400-row slice passes the clamp. One small slice is enough to reach it.
The logit gaps are far from zero on both sides, which is the sign of overconfident logits.
The gap at the 5th and the 95th percentile is 18 or more in absolute value for both models.

`scripts/diagnose_laya_calibration.py` measures where the optimum lies on the validation
rows. It searches 0.05 to 100 and reports the argmin, the NLL curve, a fit with an
intercept, and the spread of fits on 400-row slices. It leaves out texts that the scorer
reads with `predict_long`, and it counts them. The report is marked invalid when the agent
has a binning map or a language temperature, or when more than 2 percent of the rebuilt
scores differ from the score file.

The temperature fit is part of `laya.train.finetune`. This repository does not own it. The
fit uses the negative log likelihood of the two-option softmax, which is the right
objective for a `noul`. The clamp is `[0.5, 5]`, and the fit uses a calibration slice of at
most 400 rows.

### ECE before and after

Two ECE definitions appear in the report. "ECE prob" is the 10-bin ECE of P(true). "ECE top1"
is the 15-bin ECE of the top-1 confidence, which is the number in the vendor report. Every row
has both, with a 95 percent group bootstrap interval (seed 20260813). Rows that use a
fit are out of fold. The validation set has 3596 rows.

| fine-tune | row | what it is | ECE prob (10 bins) | ECE top1 (15 bins) |
|---|---|---|---|---|
| multilingual | raw, checkpoint temperature | P(true) as the checkpoint serves it, diagnosis run | 0.159 [0.143, 0.176] | 0.159 [0.143, 0.176] |
| multilingual | temperature only | wide-range temperature, no isotonic | 0.063 [0.050, 0.077] | 0.034 [0.029, 0.047] |
| multilingual | isotonic only | isotonic map on the gap | 0.008 [0.010, 0.026] | 0.012 [0.010, 0.027] |
| multilingual | temperature and isotonic | same map as the row above in exact arithmetic | 0.008 [0.010, 0.026] | 0.012 [0.010, 0.027] |
| multilingual | `isotonic_served` | isotonic map on the 4-decimal score that the scorer serves now | 0.008 [0.010, 0.026] | 0.012 [0.010, 0.027] |
| multilingual | raw, stored score file | P(true) in the score file of the score stage | 0.159 [0.144, 0.175] | 0.159 [0.143, 0.175] |
| multilingual | isotonic, stored score file | out-of-fold isotonic map on the stored scores, mean of five fold seeds | 0.009 [0.009, 0.027] | 0.016 [0.011, 0.030] |
| English | raw, checkpoint temperature | P(true) as the checkpoint serves it, diagnosis run | 0.150 [0.135, 0.167] | 0.152 [0.137, 0.169] |
| English | temperature only | wide-range temperature, no isotonic | 0.058 [0.044, 0.075] | 0.018 [0.012, 0.034] |
| English | isotonic only | isotonic map on the gap | 0.009 [0.010, 0.029] | 0.013 [0.010, 0.028] |
| English | temperature and isotonic | same map as the row above in exact arithmetic | 0.009 [0.010, 0.029] | 0.013 [0.010, 0.028] |
| English | `isotonic_served` | isotonic map on the 4-decimal score that the scorer serves now | 0.009 [0.010, 0.029] | 0.013 [0.010, 0.028] |
| English | raw, stored score file | P(true) in the score file of the score stage | 0.151 [0.133, 0.170] | 0.152 [0.134, 0.169] |
| English | isotonic, stored score file | out-of-fold isotonic map on the stored scores, mean of five fold seeds | 0.009 [0.011, 0.029] | 0.019 [0.015, 0.033] |

For the isotonic rows, the point value can be below the lower end of its percentile interval.
ECE has an upward bias under resampling, so the bootstrap values sit above the point value.

### Validity of the diagnosis run

Both diagnosis reports are marked `valid: false`. The rebuilt scores differ from the score
file by more than 0.00015 on 50.6 percent of rows for the multilingual fine-tune and on
36.0 percent for the English fine-tune. The largest difference is 0.505 and 0.773. For the
multilingual fine-tune, 18.2 percent of rows differ by more than 0.0005, 4.3 percent by more
than 0.01, and 1.5 percent by more than 0.05. The Spearman rank correlation of the rebuilt
and the stored scores is 0.9997. The checkpoint sha256 is the same in the score provenance
and in the diagnosis, so the weights are the same.

A probable cause is the batch shape, because the score stage runs padded batches of 16.
This cause is not proven. Because of the difference, this card bases its decision on the
rows "stored score file", which use the scores that the pipeline serves. The other rows
describe the logits of the diagnosis run.

### Does the clamp change the served score

No, not on validation. The vendor rounds P(true) to 4 decimals. At a temperature of 5, a
logit gap above about 46 gives exactly 1.0, and a gap below about -46 gives exactly 0.0. Every
such row ties with the rows beyond it, and an isotonic map cannot split a tie. The largest
absolute logit gap on validation is 31.875 for the multilingual fine-tune and 18.438 for the
English fine-tune, so no validation row reaches a tie. The `isotonic_served` row is the same
as the exact `isotonic` row to the third decimal for both fine-tunes. The clamp does not change
the calibrated value.

### The temperature refit

The config key `laya.temperature_refit` is `false`, so served scores do not change. When it
is `true`, `calibrate-laya --logits <gap file>` refits the temperature on validation gaps
over 0.05 to 100 and refuses a fit on a bound. It stores the value in the calibrator state.
The scorer then sets that temperature on the agent. The agent divides the logits by it
before the vendor rounds, so no rounded value is re-scaled. The isotonic map is fitted on
the scores that the scorer gives at that temperature. The calibrator stamp in
`model_versions` then holds `+t<value>`. A calibrator without a refit is refused while the
key is on, and a calibrator with a refit is refused while the key is off.

The English fine-tune has no isotonic calibrator in the served path. Its raw score is
uncalibrated. If the refit is on, it applies to the English fine-tune too, when
`calibrate-laya --model laya_en` wrote the state beside its checkpoint.

### The calibrator guard

The fit refuses an isotonic map when it makes the scores worse on rows it did not see. The
check averages `ECE(out of fold) - ECE(raw)` over five fold seeds, with a group bootstrap
(seed 20260813). It refuses only when the lower end of the 95 percent interval is above
zero. A fixed margin was removed because it rejected calibrated scores by chance at 400 rows.
A real fit needs a group id for every validation row, and the stage fails without them.

## Get the weights on a fresh clone

Weights are not in git. A fresh clone pulls the two fine-tunes from a private Hugging Face
model repo. The folders are `laya/multi-v0.2.1` (Hindi, used by the stage) and
`laya/en-v0.2.1` (English, a comparison only). Each folder holds the calibrator files
`misinfo_prob_calibrator.json` and `misinfo_prob_calibration_report.json`.

```bash
HF_REPO=<owner>/<repo>
HF_TOKEN=<read token>
LAYA_HI_CHECKPOINT=laya/multi-v0.2.1
```

Put the three lines in `.env`. The first `laya` stage run pulls the folder into
`models/laya/multi-v0.2.1`. To fetch a folder by hand, run
`python -m modeling.cli checkpoint-pull laya --version multi-v0.2.1`, and the same for
`en-v0.2.1`. The registry checks each folder against the `sha256` map in
`configs/models.yaml`. A different hash gives `checkpoint_sha256_mismatch`. The stage then
writes no score and gives the reason `laya:checkpoint_sha256_mismatch`. A pulled copy that
fails the check is deleted. A pulled copy with no `sha256` value is refused with
`checkpoint_unverified_pull`. The pull leaves out `checkpoint_latest/`. No training run is
needed.

On the stored scores, the guard difference `ECE(out of fold) - ECE(raw)` is -0.147
[-0.153, -0.122] for the multilingual fine-tune and -0.138 [-0.145, -0.112] for the English
fine-tune. Both intervals are below zero, so the guard accepts both maps.

## Out-of-scope use

Laya scores are not a determination that a claim is false. Do not use them for
moderation or enforcement.

## Scoring stage

The `laya` stage in `modeling/scoring.py` scores Hindi records only. It runs after the `misinfo` stage. The English slot belongs to the `misinfo` module, and the stage never writes for English.

The stage scores the title of a record, not the full text. The Hindi gate passed on `corpus_title` with a lift of 1.760 [1.286, 2.630]. On Hindi full text, the same model has a flag rate of 0.258 and a lift of 0.476 [0.401, 0.573]. The upper limit is below 1. Full text is outside what the gate measured.

The stage reads the title as the comparison reads it. A record with no title gets the reason `laya:no_title`. The rule `text_too_short` applies to the title. A record from a fact-checker in `configs/factcheck.yaml` gets the reason `laya:fact_check_publisher` and no value.

The stage writes `record_scores.misinfo_prob` from a calibrator, never from the raw Laya score. The calibrator files are in the checkpoint directory. Without a usable calibrator, the stage writes no score. It gives each Hindi record the reason `laya:calibrator_missing`, or another `laya:` reason.

The fit refuses raw scores with a ROC-AUC of 0.60 or less on the fit rows. The fit and the load both refuse a map with fewer than 5 distinct values. The calibrator file records the identity of the checkpoint that it was fitted for. The stage refuses a calibrator that does not match the served directory, and a calibrator that the demo flow made.

The calibrator is fitted on the validation labels of the English benchmarks. No Hindi label took part. The calibration on Hindi is not measured. Use a Hindi value as a rank within Hindi records. Do not read it as a measured probability.

A Hindi value and an English value come from different models and different calibrations. The two values are not comparable. Fusion mixes them, and the narrative score inherits this limit.

The stage never overwrites a non-null `misinfo_prob` from another module. The stamp `model_versions["laya"]` names the checkpoint directory, the calibrator and the checkpoint identity. An example is `v0.2.1+finetuned:multi-v0.2.1+isotonic@1a2b3c4d`.
