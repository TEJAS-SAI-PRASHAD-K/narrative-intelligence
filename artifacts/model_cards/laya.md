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

The temperature fit reached the upper clamp of 5 on both fine-tunes, and the report
records this issue. Expected calibration error on validation is 0.152 for the English
fine-tune. For the multilingual fine-tune, it increased from 0.068 to 0.159. Do not
read a Laya score as a calibrated probability.

## Out-of-scope use

Laya scores are not a determination that a claim is false. Do not use them for
moderation or enforcement.

## Scoring stage

The `laya` stage in `modeling/scoring.py` scores Hindi records only. It runs after the `misinfo` stage. The English slot belongs to the `misinfo` module, and the stage never writes for English.

The stage scores the title of a record, not the full text. The Hindi gate passed on `corpus_title` with a lift of 1.760 [1.286, 2.630]. On Hindi full text, the same model has a flag rate of 0.258 and a lift of 0.476 [0.401, 0.573]. The upper limit is below 1. Full text is outside what the gate measured.

The stage reads the title as the comparison reads it. A record with no title gets the reason `laya:no_title`. The rule `text_too_short` applies to the title.

The stage writes `record_scores.misinfo_prob` from a calibrator, never from the raw Laya score. The calibrator files are in the checkpoint directory. Without a usable calibrator, the stage writes no score. It gives each Hindi record the reason `laya:calibrator_missing`, or another `laya:` reason.

The fit refuses raw scores with a ROC-AUC of 0.60 or less on the fit rows. The fit and the load both refuse a map with fewer than 5 distinct values. The calibrator file records the identity of the checkpoint that it was fitted for. The stage refuses a calibrator that does not match the served directory, and a calibrator that the demo flow made.

The calibrator is fitted on the validation labels of the English benchmarks. No Hindi label took part. The calibration on Hindi is not measured. Use a Hindi value as a rank within Hindi records. Do not read it as a measured probability.

A Hindi value and an English value come from different models and different calibrations. The two values are not comparable. Fusion mixes them, and the narrative score inherits this limit.

The stage never overwrites a non-null `misinfo_prob` from another module. The stamp `model_versions["laya"]` names the checkpoint directory, the calibrator and the checkpoint identity. An example is `v0.2.1+finetuned:multi-v0.2.1+isotonic@1a2b3c4d`.
