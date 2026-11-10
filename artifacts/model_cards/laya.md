# Model card — Laya misinformation scorer

**Module:** `modeling/text/laya_clf.py` · **Version:** `v0.2.1`
**Status:** fine-tuned and compared. The Hindi gate passes. The scoring stage is not built yet.

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
13 to 261 Hindi claims when the sitemap backfill added Vishvas News archive pages.

| measure, Hindi titles | Laya multilingual fine-tune |
|---|---|
| claims scored | 261 |
| recall on known-false claims | 0.115 [0.077, 0.157] |
| flag rate on 136 corpus titles | 0.051 [0.015, 0.088] |
| lift | 2.233 [1.146, 6.605] |

**The gate passes.** The result has clear limits:

- The lower limit is 1.146, close to 1. The corpus has only 136 Hindi titles.
- Recall is 0.115, so the model does not flag most debunked claims at this threshold.
- Almost all Hindi claims come from one publisher. The result reflects the choice of
  claims by that publisher.
- The claims are short fact-check statements, and the corpus has news titles. A part
  of the lift can come from this difference in text form.
- The English recall on the fact-check set is 0.045 [0.011, 0.091] for both Laya
  fine-tunes. The English result does not prove any transfer to Hindi.

## Calibration

The temperature fit reached the upper clamp of 5 on both fine-tunes, and the report
records this issue. Expected calibration error on validation is 0.152 for the English
fine-tune. For the multilingual fine-tune, it increased from 0.068 to 0.159. Do not
read a Laya score as a calibrated probability.

## Out-of-scope use

Laya scores are not a determination that a claim is false. Do not use them for
moderation or enforcement.

## Scoring stage

The `laya` stage in `modeling/scoring.py` scores Hindi records only. It runs after the `misinfo` stage. The English slot belongs to the `misinfo` module, and the stage never writes for English.

The stage writes `record_scores.misinfo_prob` from a calibrator, never from the raw Laya score. The calibrator file `misinfo_prob_calibrator.json` must be in the checkpoint directory. Without it, the stage writes no score. It gives each Hindi record the reason `laya:calibrator_missing`, or another `laya:` reason.

The calibrator is fitted on the validation labels of the English benchmarks. No Hindi label took part. The calibration on Hindi is not measured. Use a Hindi value as a rank within Hindi records. Do not read it as a measured probability.

A Hindi value and an English value come from different models and different calibrations. The two values are not comparable. Fusion mixes them, and the narrative score inherits this limit.

The stage never overwrites a non-null `misinfo_prob` from another module. The stamp `model_versions["laya"]` names the checkpoint directory and the calibrator. An example is `v0.2.1+finetuned:multi-v0.2.1+isotonic@1a2b3c4d`.
