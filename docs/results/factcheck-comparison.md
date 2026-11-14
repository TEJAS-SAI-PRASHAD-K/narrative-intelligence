# Fact-check evaluation: three-way comparison

> The fact-check set contains only the claims that publishers chose to debunk. It is not a sample of false claims in general, and every claim in it is false. Recall on it measures detection of publicly debunked claims and is never reported without the flag rate on the unlabelled corpus.

- Baseline: `tfidf`. Precision target: 0.85. Seed: `20260813`. Resamples: 2000. Intervals are 95% percentile bootstrap intervals.
- Split: grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)
- Evaluation claims: 4281 (frozen, 4281 manifest rows)

## Notices

- laya_multi_base: control model, not a candidate: its numbers are a baseline for the fine-tuned models and it is never shipped
- laya_multi_base: no validation threshold reaches precision 0.85; the best available is 0.444. The flags below use that threshold.
- laya_en: 130 test row(s) were not scored by both laya_en and tfidf; the paired comparison uses the 6955 rows both scored
- laya_multi: 130 test row(s) were not scored by both laya_multi and tfidf; the paired comparison uses the 6955 rows both scored
- laya_multi_base: 130 test row(s) were not scored by both laya_multi_base and tfidf; the paired comparison uses the 6955 rows both scored

## Models and operating points

The threshold of each model comes from its own validation rows. Nothing is shared.

| model | version | threshold | val precision | val recall | target met | val scored/rows | test scored/rows |
|---|---|---|---|---|---|---|---|
| laya_en | v0.2.1+finetuned:en-v0.2.1 | 0.9734 | 0.880 | 0.318 | yes | 3558/3596 | 6955/7085 |
| laya_multi | v0.2.1+finetuned:multi-v0.2.1 | 0.9770 | 0.851 | 0.165 | yes | 3558/3596 | 6955/7085 |
| laya_multi_base (control) | v0.2.1+base@55cf4c4ebb4e | 0.9041 | 0.444 | 0.004 | **no** | 3558/3596 | 6955/7085 |
| misinfo | v0.2.0 | 0.7389 | 0.851 | 0.482 | yes | 3596/3596 | 7085/7085 |
| tfidf | tfidf-logreg+sklearn-1.9.1 | 0.8573 | 0.850 | 0.244 | yes | 3596/3596 | 7085/7085 |

## Shared test split: macro-F1 at each model's own threshold

| model | macro-F1 [95% CI] | scored/rows | flagged | skips |
|---|---|---|---|---|
| laya_en | 0.683 [0.670, 0.696] | 6955/7085 | 800 | text_too_short=130 |
| laya_multi | 0.565 [0.552, 0.578] | 6955/7085 | 434 | text_too_short=130 |
| laya_multi_base | 0.414 [0.410, 0.419] | 6955/7085 | 18 | text_too_short=130 |
| misinfo | 0.753 [0.741, 0.764] | 7085/7085 | 1263 | none |
| tfidf | 0.603 [0.590, 0.616] | 7085/7085 | 570 | none |

### Ship decision against the baseline

A transformer ships only if the lower limit of the 95% paired-bootstrap interval of (its macro-F1 minus the baseline macro-F1) is above 0.

| model | macro-F1 difference [95% CI] | paired rows | left out | decision | reason |
|---|---|---|---|---|---|
| laya_en | 0.079 [0.065, 0.095] | 6955 | 130 | **ship** | lower limit 0.0646 is above 0 |
| laya_multi | -0.039 [-0.054, -0.022] | 6955 | 130 | **do not ship** | lower limit -0.0536 is not above 0 |
| laya_multi_base | -0.189 [-0.202, -0.176] | 6955 | 130 | **do not ship (control)** | control model, not a candidate: its numbers are a baseline for the fine-tuned models and it is never shipped. By the numbers alone: lower limit -0.2023 is not above 0 |
| misinfo | 0.150 [0.135, 0.164] | 7085 | 0 | **ship** | lower limit 0.1354 is above 0 |

The difference is the candidate minus the baseline, on test rows scored by both, with the same resampled row indices for both models.

## Recall, flag rate and lift: corpus titles (primary)

Lift = recall on known-false claims / flag rate on the unlabelled corpus, at the same threshold. Fact-check publishers are excluded from the corpus.

| model | language | claims scored/rows | recall [95% CI] | corpus scored/rows | flag rate [95% CI] | lift [95% CI] | zero-flag resamples |
|---|---|---|---|---|---|---|---|
| laya_en | en | 100/100 | 0.060 [0.020, 0.110] | 1350/1350 | 0.030 [0.021, 0.040] | 1.976 [0.587, 3.971] | 0 of 2000 |
| laya_en | hi | 0/4180 | no score: no valid score for any of 4180 claim rows (skip_reason: unrouted_language=4180) | 0/431 | no score: no valid score for any of 431 corpus title rows (skip_reason: unrouted_language=431) | no score: recall: no valid score for any of 4180 claim rows (skip_reason: unrouted_language=4180); flag rate: no valid score for any of 431 corpus title rows (skip_reason: unrouted_language=431) | n/a of 2000 |
| laya_en | all | 100/4281 | 0.060 [0.020, 0.110] | 1350/1781 | 0.030 [0.021, 0.040] | 1.976 [0.587, 3.971] | 0 of 2000 |
| laya_multi | en | 100/100 | 0.050 [0.010, 0.100] | 1350/1350 | 0.030 [0.021, 0.039] | 1.688 [0.365, 3.750] | 0 of 2000 |
| laya_multi | hi | 4180/4180 | 0.122 [0.113, 0.132] | 431/431 | 0.070 [0.046, 0.093] | 1.760 [1.286, 2.630] | 0 of 2000 |
| laya_multi | all | 4280/4281 | 0.121 [0.111, 0.130] | 1781/1781 | 0.039 [0.030, 0.048] | 3.073 [2.445, 3.975] | 0 of 2000 |
| laya_multi_base | en | 100/100 | 0.000 [0.000, 0.000] | 1350/1350 | 0.004 [0.001, 0.008] | 0.000 [0.000, 0.000] (5 of 2000 resamples had flag rate 0; they are counted in the percentiles as undefined and are not dropped) | 5 of 2000 |
| laya_multi_base | hi | 4180/4180 | 0.008 [0.005, 0.011] | 431/431 | 0.000 [0.000, 0.000] | undefined: flag rate is 0 at the validation threshold (0 of 431 corpus rows flagged), so the ratio is undefined | 2000 of 2000 |
| laya_multi_base | all | 4280/4281 | 0.008 [0.005, 0.010] | 1781/1781 | 0.003 [0.001, 0.006] | 2.289 [1.109, 8.322] (5 of 2000 resamples had flag rate 0; they are counted in the percentiles as undefined and are not dropped) | 5 of 2000 |
| misinfo | en | 100/100 | 0.520 [0.420, 0.610] | 1350/1350 | 0.081 [0.067, 0.096] | 6.382 [4.950, 8.297] | 0 of 2000 |
| misinfo | hi | 0/4180 | no score: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180) | 0/431 | no score: no valid score for any of 431 corpus title rows (skip_reason: unsupported_language=431) | no score: recall: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180); flag rate: no valid score for any of 431 corpus title rows (skip_reason: unsupported_language=431) | n/a of 2000 |
| misinfo | all | 100/4281 | 0.520 [0.420, 0.610] | 1350/1781 | 0.081 [0.067, 0.096] | 6.382 [4.950, 8.297] | 0 of 2000 |
| tfidf | en | 100/100 | 0.030 [0.000, 0.070] | 1350/1350 | 0.004 [0.001, 0.008] | 6.750 [0.000, 31.556] (5 of 2000 resamples had flag rate 0; they are counted in the percentiles as undefined and are not dropped) | 5 of 2000 |
| tfidf | hi | 0/4180 | no score: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180) | 0/431 | no score: no valid score for any of 431 corpus title rows (skip_reason: unsupported_language=431) | no score: recall: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180); flag rate: no valid score for any of 431 corpus title rows (skip_reason: unsupported_language=431) | n/a of 2000 |
| tfidf | all | 100/4281 | 0.030 [0.000, 0.070] | 1350/1781 | 0.004 [0.001, 0.008] | 6.750 [0.000, 31.556] (5 of 2000 resamples had flag rate 0; they are counted in the percentiles as undefined and are not dropped) | 5 of 2000 |

## Recall, flag rate and lift: corpus full text (second value)

Lift = recall on known-false claims / flag rate on the unlabelled corpus, at the same threshold. Fact-check publishers are excluded from the corpus.

> **Not like for like.** Full-text flag rates and lifts are not like for like across model families. Laya scores long text with predict_long, which takes the maximum over windows of the whole text. The roberta classifier truncates the text at 256 tokens. A model that reads more of a long article has more chances to cross its threshold. Compare models on corpus_title, the primary figure, and read corpus_text as a second value.

| model | language | claims scored/rows | recall [95% CI] | corpus scored/rows | flag rate [95% CI] | lift [95% CI] | zero-flag resamples |
|---|---|---|---|---|---|---|---|
| laya_en | en | 100/100 | 0.060 [0.020, 0.110] | 1350/1350 | 0.032 [0.023, 0.041] (not like for like) | 1.884 [0.600, 3.857] (not like for like) | 0 of 2000 |
| laya_en | hi | 0/4180 | no score: no valid score for any of 4180 claim rows (skip_reason: unrouted_language=4180) | 0/431 | no score: no valid score for any of 431 corpus text rows (skip_reason: unrouted_language=431) (not like for like) | no score: recall: no valid score for any of 4180 claim rows (skip_reason: unrouted_language=4180); flag rate: no valid score for any of 431 corpus text rows (skip_reason: unrouted_language=431) (not like for like) | n/a of 2000 |
| laya_en | all | 100/4281 | 0.060 [0.020, 0.110] | 1350/1781 | 0.032 [0.023, 0.041] (not like for like) | 1.884 [0.600, 3.857] (not like for like) | 0 of 2000 |
| laya_multi | en | 100/100 | 0.050 [0.010, 0.100] | 1350/1350 | 0.053 [0.041, 0.064] (not like for like) | 0.951 [0.196, 1.969] (not like for like) | 0 of 2000 |
| laya_multi | hi | 4180/4180 | 0.122 [0.113, 0.132] | 431/431 | 0.258 [0.216, 0.299] (not like for like) | 0.476 [0.401, 0.573] (not like for like) | 0 of 2000 |
| laya_multi | all | 4280/4281 | 0.121 [0.111, 0.130] | 1781/1781 | 0.102 [0.088, 0.117] (not like for like) | 1.182 [1.012, 1.416] (not like for like) | 0 of 2000 |
| laya_multi_base | en | 100/100 | 0.000 [0.000, 0.000] | 1350/1350 | 0.049 [0.039, 0.061] (not like for like) | 0.000 [0.000, 0.000] (not like for like) | 0 of 2000 |
| laya_multi_base | hi | 4180/4180 | 0.008 [0.005, 0.011] | 431/431 | 0.193 [0.155, 0.230] (not like for like) | 0.041 [0.027, 0.058] (not like for like) | 0 of 2000 |
| laya_multi_base | all | 4280/4281 | 0.008 [0.005, 0.010] | 1781/1781 | 0.084 [0.071, 0.097] (not like for like) | 0.092 [0.061, 0.129] (not like for like) | 0 of 2000 |
| misinfo | en | 100/100 | 0.520 [0.420, 0.610] | 1350/1350 | 0.067 [0.053, 0.080] (not like for like) | 7.800 [5.951, 10.212] (not like for like) | 0 of 2000 |
| misinfo | hi | 0/4180 | no score: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180) | 0/431 | no score: no valid score for any of 431 corpus text rows (skip_reason: unsupported_language=431) (not like for like) | no score: recall: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180); flag rate: no valid score for any of 431 corpus text rows (skip_reason: unsupported_language=431) (not like for like) | n/a of 2000 |
| misinfo | all | 100/4281 | 0.520 [0.420, 0.610] | 1350/1781 | 0.067 [0.053, 0.080] (not like for like) | 7.800 [5.951, 10.212] (not like for like) | 0 of 2000 |
| tfidf | en | 100/100 | 0.030 [0.000, 0.070] | 1350/1350 | 0.102 [0.086, 0.119] (not like for like) | 0.293 [0.000, 0.675] (not like for like) | 0 of 2000 |
| tfidf | hi | 0/4180 | no score: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180) | 0/431 | no score: no valid score for any of 431 corpus text rows (skip_reason: unsupported_language=431) (not like for like) | no score: recall: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180); flag rate: no valid score for any of 431 corpus text rows (skip_reason: unsupported_language=431) (not like for like) | n/a of 2000 |
| tfidf | all | 100/4281 | 0.030 [0.000, 0.070] | 1350/1781 | 0.102 [0.086, 0.119] (not like for like) | 0.293 [0.000, 0.675] (not like for like) | 0 of 2000 |

## Hindi gate

Hindi scores enter fusion only with at least 50 Hindi claims and a title lift interval whose lower limit is above 1.

| model | Hindi claims scored | lift [95% CI] | result | reason |
|---|---|---|---|---|
| laya_en | 0 of 4180 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 4180 claim rows (skip_reason: unrouted_language=4180); flag rate: no valid score for any of 431 corpus title rows (skip_reason: unrouted_language=431) |
| laya_multi | 4180 of 4180 | 1.760 [1.286, 2.630] | **PASS** | 4180 Hindi claims and lift lower limit 1.286 above 1 |
| laya_multi_base | 4180 of 4180 | no value | **FAIL (control)** | control model, not a candidate: its numbers are a baseline for the fine-tuned models and it is never shipped. By the numbers alone: no lift interval: flag rate is 0 at the validation threshold (0 of 431 corpus rows flagged), so the ratio is undefined |
| misinfo | 0 of 4180 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180); flag rate: no valid score for any of 431 corpus title rows (skip_reason: unsupported_language=431) |
| tfidf | 0 of 4180 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 4180 claim rows (skip_reason: unsupported_language=4180); flag rate: no valid score for any of 431 corpus title rows (skip_reason: unsupported_language=431) |

## Skipped and excluded rows

A null score is never counted as not flagged. Each one is left out of the denominator and counted here.

| model | part | rows | no score | skip_reason counts |
|---|---|---|---|---|
| laya_en | claims | 4281 | 4181 | unrouted_language=4181 |
| laya_en | corpus_title | 1781 | 431 | unrouted_language=431 |
| laya_en | corpus_text | 1781 | 431 | unrouted_language=431 |
| laya_multi | claims | 4281 | 1 | unrouted_language=1 |
| laya_multi | corpus_title | 1781 | 0 | none |
| laya_multi | corpus_text | 1781 | 0 | none |
| laya_multi_base | claims | 4281 | 1 | unrouted_language=1 |
| laya_multi_base | corpus_title | 1781 | 0 | none |
| laya_multi_base | corpus_text | 1781 | 0 | none |
| misinfo | claims | 4281 | 4181 | unsupported_language=4181 |
| misinfo | corpus_title | 1781 | 431 | unsupported_language=431 |
| misinfo | corpus_text | 1781 | 431 | unsupported_language=431 |
| tfidf | claims | 4281 | 4181 | unsupported_language=4181 |
| tfidf | corpus_title | 1781 | 431 | unsupported_language=431 |
| tfidf | corpus_text | 1781 | 431 | unsupported_language=431 |

| model | claim ids excluded (reason=count) | kept claims absent from scores | corpus rows from fact-check publishers (title/text) | corpus rows without source_detail |
|---|---|---|---|---|
| laya_en | label_not_false=1 | 0 | 151/151 | 0/0 |
| laya_multi | label_not_false=1 | 0 | 151/151 | 0/0 |
| laya_multi_base | label_not_false=1 | 0 | 151/151 | 0/0 |
| misinfo | label_not_false=1 | 0 | 151/151 | 0/0 |
| tfidf | label_not_false=1 | 0 | 151/151 | 0/0 |

## Inputs

| model | file | rows | sha256 |
|---|---|---|---|
| laya_en | laya_en.parquet | 18827 | `bf83d28f3d0467bd52837b98120662439241114e668353a139d3b41742229462` |
| laya_multi | laya_multi.parquet | 18827 | `befb5dd47d633540aee9348bcb26bec0493a341a0a5fcc20c4159d757389e48f` |
| laya_multi_base | laya_multi_base.parquet | 18827 | `b35f877880ee076749a502b7c2d227ff05d3f45498ae1c7e2f3841b82afbc370` |
| misinfo | misinfo.parquet | 18827 | `386f4047a5fb05014ebe68af5008a03287e1117f6266662621cb9753436eaa3a` |
| tfidf | tfidf.parquet | 18827 | `a9570b67313a14c70f1ef0e4808333a19d80e4518504176f6ae76e23202adbf8` |

## Reproducibility

- seed: `20260813`
- input manifest hash: `c066fe12b0de6739`
- regenerate by rerunning `python -m modeling.cli evaluate-factchecks` on the same score files and the same seed
