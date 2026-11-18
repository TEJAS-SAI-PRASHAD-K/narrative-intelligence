# Model card — Laya misinformation scorer

**Module:** `modeling/text/laya_clf.py` · **Version:** `v0.2.1`
**Status:** fine-tuned and compared. The Hindi gate passes. The Hindi scoring stage is built. It scores titles only.

## Result

The comparison run of 2026-11-09 fine-tuned two checkpoints of
`convaiinnovations/laya`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
Training used the same LIAR, FakeNewsNet and CoAID split as the `misinfo` module. The
full report is [docs/results/factcheck-comparison.md](../../docs/results/factcheck-comparison.md).
The report was rerun on 2026-11-18 with the same score files and seed, on 4428 claims and with
all eight fact-checker domains left out of the corpus. The macro F1 values and the decisions
below are the same in both runs. The Hindi numbers below are from the rerun.

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
13 to 1016 (2026-11-11) to 4180 (2026-11-14) to 4334 (2026-11-18) Hindi claims. The sitemap backfill read the whole Vishvas News
archive inside the 2022 window. A later backfill added BOOM Hindi and Fact Crescendo. We froze
the manifest again each time.

| measure, Hindi titles | Laya multilingual fine-tune |
|---|---|
| claims scored | 4334 |
| recall on known-false claims | 0.119 [0.109, 0.128] |
| flag rate on 717 scored corpus titles (721 rows, 4 without a title) | 0.068 [0.050, 0.086] |
| lift | 1.735 [1.331, 2.368] |

The report of 2026-11-18 leaves all eight fact-checker domains out of the corpus (265 rows
out of the title and text counts). The earlier report (freeze of 2026-11-14, 4180 claims) left
out four. It gave a lift of 1.760 [1.286, 2.630] on 431 Hindi titles.

**The gate passes.** The result has clear limits:

- The lower limit is 1.331. With 1016 claims (freeze of 2026-11-11) it was 1.116. The interval is much
  narrower now, but the corpus has only 717 scored Hindi titles.
- On full text, the lift is 0.503 [0.432, 0.590] and the upper limit is below 1. The
  model flags Hindi corpus articles (0.236) more often than debunked claims (0.119).
  The full-text figures are not like for like with the title figures. The stage scores
  titles only. Do not use the score on full text.
- Recall is 0.119, so the model does not flag most debunked claims at this threshold.
- Most Hindi claims come from one publisher. Vishvas News gives 3789 of the 4334
  claims (87.4 percent), BOOM gives 524 (12.1 percent) and Fact Crescendo gives 21
  (0.5 percent). The result reflects mostly the choice of claims by Vishvas News. In the
  freeze of 2026-11-14, Vishvas News gave 4173 of 4180 claims (99.8 percent).
- The claim dedup pass with embeddings collapsed 421 rows, and the simhash pass collapsed 29.
  The precision of the embedding collapses is 0.390 (78 of 200 labelled pairs), Wilson
  [0.325, 0.459]. It is provisional. The labeller is a model, and no person reviewed the labels.
- The claims are short fact-check statements, and the corpus has news titles. A part
  of the lift can come from this difference in text form.
- The English recall on the fact-check set (93 claims) is 0.065 [0.022, 0.118] for `laya_en` and
  0.054 [0.011, 0.108] for `laya_multi`. The English result does not prove any transfer to Hindi.

### Transfer on Hindi corpus titles (#31)

Status: provisional, needs person labels. Every label here is from models, and no person
reviewed it. Each number carries the note "labeller: model, unreviewed by a person".

The sample has 150 Hindi rows, drawn with `modeling.cli sample-for-labelling misinfo --lang hi
--n 150` and the seed 20260813, from outlets that are not fact-checkers. The record
embeddings come from the claim dedup model. In the claim-match stratum, 2 records match a
false claim. The Reddit, Mastodon and Telegram strata are empty, because the corpus has no
such records. Laya scores the title, so the labels apply to the title. Two independent model
labellers labelled each row. They did not see the scores or each other. They agree on every
row: raw agreement 1.000 for the label and for `reports_on_false_claim`, with 0
disagreements. Cohen kappa of the label is undefined, because both used one class. The run
used `relabel-eval --allow-model-labels --n-bootstrap 2000` on 2026-11-18 (UTC) with the
model v0.2.1.

| measure on the 150 Hindi titles | value |
|---|---|
| positive rate | 0 of 150, 0.000 [0.000, 0.025] (Wilson 95%) |
| ROC-AUC, precision, recall | not measured (reason `one_class_only`) |
| mean calibrated probability | 0.446 [0.401, 0.491] |
| mean calibrated probability, reweighted | 0.432 [0.388, 0.474] |
| ECE | 0.446 [0.401, 0.491] |
| ECE, reweighted | 0.432 [0.388, 0.474] (n_eff 137.5) |
| flag rate | not measured, because `laya.decision_threshold` is null |

All 150 labels are 0, so the sample holds one class. With one class, the ECE equals the mean
predicted probability. With n below 200 the ECE is noisy, so read the interval and not the
point. The mean calibrated value of 0.446 is far above the observed positive rate, whose
upper limit is 0.025. By this sample, the calibration on Hindi corpus titles is poor. The
calibrator saw only English labels, and this result is the first measurement on Hindi. Issue
#76 tracks a refit on Hindi labels. The result is provisional until a person labels a
sample that has positives.

## Real run of the stage

On 2026-11-10 the stage scored 156 Hindi titles on the GPU host. No raw value fell
outside the fitted range, and the stage wrote 24 distinct calibrated values. The mean
of the calibrated Hindi values was 0.458, and the mean of the English `misinfo` values
was 0.451. These means do not measure accuracy, because the corpus has no labels.

The calibrator had a raw ROC-AUC of 0.854 on 3,558 English validation rows. Brier went
from 0.173 to 0.134. The calibration error of 0 after the fit is an in-sample value.

## Calibration of the raw model

The vendor fit of the temperature ends on its upper clamp of 5 for both fine-tunes. Issue 33
asked why, and whether the clamp is right. This section gives the answer. The vendor fit does
not converge on these fine-tunes, because of the clamp. A fit that ends on the clamp has not
converged.

**Decision: the clamp at 5 stays and `laya.temperature_refit` stays `false`.** The isotonic
calibrator does the calibration. On the stored scores, ECE prob falls from 0.159 to 0.009 out
of fold for the multilingual fine-tune. The guard difference is below zero, and its interval
excludes zero (see "The calibrator guard"). ECE does not rise after calibration.

The English fine-tune has no isotonic calibrator in the served path. Its figure of 0.151 to
0.009 shows what an isotonic map can do on its validation rows. The pipeline does not serve it.

The raw score is not a probability. The mean confidence of the raw score is 0.966 and 0.962
(multilingual and English, diagnosis run). The raw ECE prob is 0.159 and 0.151 on the stored
scores. Use the calibrated value of the stage for the multilingual fine-tune only.

All ECE values in this card come from validation rows. Those rows are English benchmark rows.
The Hindi ECE of the model-labelled sample is in "Transfer on Hindi corpus titles (#31)".

### History

The vendor report of the fit gave an ECE of 0.068 before the temperature fit and 0.159 after
it. These values are the 15-bin top-1 ECE on held-out rows (`run_summary.before.ece` and
`run_summary.after.ece`). The raw ECE at temperature 1 in this run is 0.191. The reason that
the vendor value of 0.068 differs from 0.191 is not known. This repository does not reproduce
the vendor protocol.

### Why the optimum lies past 5

The temperature that minimizes the validation NLL is inside the wide search of 0.05 to 100.
It is not on a bound. The cause is overconfident logits. It is not the fit objective, and it
is not the per-type groups. The checkpoint holds `per_type` temperatures of `[1.0, 1.0, 5.0]`.
Only the `noul` type is on the clamp.

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
The optimum is near 11.8 for the multilingual fine-tune, while the accuracy at a gap of zero is
0.807. A model that is right about 81 percent of the time and needs a temperature of 11.8 is
overconfident. The English fine-tune has the optimum 11.408 and an accuracy of 0.813 at a gap
of zero.

The English gap quantiles are nearly flat at both ends, and its largest stored tie has 430
rows. This also fits saturated or quantized logits.

Temperature alone at the optimum improves the NLL (0.460 against 0.657) and the ECE prob
(0.063 against 0.159, multilingual). The clamp stays because the isotonic map does better, and
the served value goes through the map.

`scripts/diagnose_laya_calibration.py` measures where the optimum lies on the validation
rows. It searches 0.05 to 100. It reports the argmin, the NLL curve, a fit with an
intercept, and the spread of fits on 400-row slices. It leaves out and counts the texts that the scorer
reads with `predict_long`.

The report is marked invalid in two cases. In the first case, the agent has a binning map or
a language temperature. In the second case, more than 2 percent of the rebuilt scores differ
from the score file.

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
| multilingual | raw, temperature 1 | P(true) with no temperature | 0.191 [0.174, 0.208] | 0.190 [0.174, 0.208] |
| multilingual | temperature only | wide-range temperature, no isotonic | 0.063 [0.050, 0.077] | 0.034 [0.029, 0.047] |
| multilingual | isotonic only | isotonic map on the gap | 0.008 [0.010, 0.026] | 0.012 [0.010, 0.027] |
| multilingual | temperature and isotonic | same map as the row above in exact arithmetic | 0.008 [0.010, 0.026] | 0.012 [0.010, 0.027] |
| multilingual | `isotonic_served` | isotonic map on the 4-decimal score that the scorer serves now | 0.008 [0.010, 0.026] | 0.012 [0.010, 0.027] |
| multilingual | raw, stored score file | P(true) in the score file of the score stage | 0.159 [0.144, 0.175] | 0.159 [0.143, 0.175] |
| multilingual | isotonic, stored score file | out-of-fold isotonic map on the stored scores, mean of five fold seeds | 0.009 [0.009, 0.027] | 0.016 [0.011, 0.030] |
| English | raw, checkpoint temperature | P(true) as the checkpoint serves it, diagnosis run | 0.150 [0.135, 0.167] | 0.152 [0.137, 0.169] |
| English | raw, temperature 1 | P(true) with no temperature | 0.183 [0.168, 0.202] | 0.184 [0.169, 0.202] |
| English | temperature only | wide-range temperature, no isotonic | 0.058 [0.044, 0.075] | 0.018 [0.012, 0.034] |
| English | isotonic only | isotonic map on the gap | 0.009 [0.010, 0.029] | 0.013 [0.010, 0.028] |
| English | temperature and isotonic | same map as the row above in exact arithmetic | 0.009 [0.010, 0.029] | 0.013 [0.010, 0.028] |
| English | `isotonic_served` | isotonic map on the 4-decimal score that the scorer serves now | 0.009 [0.010, 0.029] | 0.013 [0.010, 0.028] |
| English | raw, stored score file | P(true) in the score file of the score stage | 0.151 [0.133, 0.170] | 0.152 [0.134, 0.169] |
| English | isotonic, stored score file | out-of-fold isotonic map on the stored scores, mean of five fold seeds | 0.009 [0.011, 0.029] | 0.019 [0.015, 0.033] |

For the isotonic rows, the point value can be below the lower end of its percentile interval.
ECE has an upward bias when the bootstrap resamples the rows. The bootstrap values sit above
the point value.

### Validity of the diagnosis run

Both diagnosis reports are marked `valid: false`. On 50.6 percent of rows, the rebuilt multilingual score differs from the
score file by over 0.00015. The share is 36.0 percent for English.
The largest difference is 0.505 and 0.773.

| rebuilt against stored | multilingual | English |
|---|---|---|
| share of rows that differ by more than 0.0005 | 18.2 percent | 11.2 percent |
| share of rows that differ by more than 0.01 | 4.3 percent | 3.0 percent |
| share of rows that differ by more than 0.05 | 1.5 percent | 1.35 percent |
| Spearman rank correlation | 0.9997 | 0.994 |

The checkpoint sha256 is the same in the score provenance and in the diagnosis for both
fine-tunes, so the weights are the same. A probable cause is the batch shape, because the score
stage runs padded batches of 16. This cause is not proven.

Because of the difference, this card bases its decision on the rows "stored score file". Those
rows use the scores that the pipeline serves. The other rows describe the logits of the
diagnosis run.

### Does the clamp change the served score

No. This answer comes from the stored scores of the score stage, not from the diagnosis run.
The vendor rounds P(true) to 4 decimals, and rows with the same rounded value tie. An isotonic
map cannot split a tie. At a temperature of 5, the rounded value is exactly 1.0 for a logit gap
above 5 × ln(0.99995 / 0.00005), which is about 49.5. It is exactly 0.0 below about -49.5.

On the stored scores, no row is at 0.0 or at 1.0 for either fine-tune. The multilingual scores
have 811 distinct values, and the largest tie has 21 rows. The English scores have 342 distinct
values, and the largest tie has 430 rows. Near-saturated rows share a rounded value
before the logit gap reaches 49.5.

| stored scores, ROC-AUC | multilingual | English |
|---|---|---|
| raw | 0.854 [0.839, 0.869] | 0.875 [0.861, 0.890] |
| isotonic, out of fold | 0.847 [0.831, 0.862] | 0.868 [0.854, 0.883] |

The isotonic map lowers the ROC-AUC a little, because a step map creates ties. The intervals
overlap. The `isotonic_served` row is the same as the exact `isotonic` row to the third decimal
for both fine-tunes. The clamp does not change the calibrated value.

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
[-0.153, -0.122] for the multilingual fine-tune. It is -0.138 [-0.145, -0.112] for the English
fine-tune. Both intervals are below zero, so the guard accepts both maps.

The guard compares the raw ECE with its own `ece_after_oof`. This value is 0.012 for the
multilingual fine-tune and 0.013 for the English fine-tune. The differences are -0.147 and
-0.138. The 0.009 figure is the mean over the five fold seeds, which is a different protocol.

## Out-of-scope use

Laya scores are not a determination that a claim is false. Do not use them for
moderation or enforcement.

## Scoring stage

The `laya` stage in `modeling/scoring.py` scores Hindi records only. It runs after the `misinfo` stage. The English slot belongs to the `misinfo` module, and the stage never writes for English.

The stage scores the title of a record, not the full text. The Hindi gate passed on `corpus_title` with a lift of 1.735 [1.331, 2.368]. On Hindi full text, the same model has a flag rate of 0.236 and a lift of 0.503 [0.432, 0.590]. The upper limit is below 1. Full text is outside what the gate measured.

The stage reads the title as the comparison reads it. A record with no title gets the reason `laya:no_title`. The rule `text_too_short` applies to the title. A record from a fact-checker in `configs/factcheck.yaml` gets the reason `laya:fact_check_publisher` and no value.

The stage writes `record_scores.misinfo_prob` from a calibrator, never from the raw Laya score. The calibrator files are in the checkpoint directory. Without a usable calibrator, the stage writes no score. It gives each Hindi record the reason `laya:calibrator_missing`, or another `laya:` reason.

The fit refuses raw scores with a ROC-AUC of 0.60 or less on the fit rows. The fit and the load both refuse a map with fewer than 5 distinct values. The calibrator file records the identity of the checkpoint that it was fitted for. The stage refuses a calibrator that does not match the served directory, and a calibrator that the demo flow made.

The calibrator is fitted on the validation labels of the English benchmarks. No Hindi label took part. The calibration on Hindi corpus titles is poor by the model-labelled sample (see "Transfer on Hindi corpus titles (#31)"). Use a Hindi value as a rank within Hindi records. Do not read it as a measured probability.

A Hindi value and an English value come from different models and different calibrations. The two values are not comparable. Fusion mixes them, and the narrative score inherits this limit.

The stage never overwrites a non-null `misinfo_prob` from another module. The stamp `model_versions["laya"]` names the checkpoint directory, the calibrator and the checkpoint identity. An example is `v0.2.1+finetuned:multi-v0.2.1+isotonic@1a2b3c4d`.
