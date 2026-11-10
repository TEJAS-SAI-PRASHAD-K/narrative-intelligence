# Fact-check evaluation: three-way comparison

> The fact-check set contains only the claims that publishers chose to debunk. It is not a sample of false claims in general, and every claim in it is false. Recall on it measures detection of publicly debunked claims and is never reported without the flag rate on the unlabelled corpus.

- Baseline: `tfidf`. Precision target: 0.85. Seed: `20260813`. Resamples: 2000. Intervals are 95% percentile bootstrap intervals.
- Split: grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)
- Evaluation claims: 101 (frozen, 101 manifest rows)

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
| laya_en | en | 88/88 | 0.045 [0.011, 0.091] | 374/374 | 0.040 [0.021, 0.061] | 1.133 [0.224, 2.975] | 0 of 2000 |
| laya_en | hi | 0/13 | no score: no valid score for any of 13 claim rows (skip_reason: unrouted_language=13) | 0/136 | no score: no valid score for any of 136 corpus title rows (skip_reason: unrouted_language=136) | no score: recall: no valid score for any of 13 claim rows (skip_reason: unrouted_language=13); flag rate: no valid score for any of 136 corpus title rows (skip_reason: unrouted_language=136) | n/a of 2000 |
| laya_en | all | 88/101 | 0.045 [0.011, 0.091] | 374/510 | 0.040 [0.021, 0.061] | 1.133 [0.224, 2.975] | 0 of 2000 |
| laya_multi | en | 88/88 | 0.045 [0.011, 0.091] | 374/374 | 0.035 [0.019, 0.053] | 1.308 [0.250, 3.542] | 0 of 2000 |
| laya_multi | hi | 13/13 | 0.077 [0.000, 0.231] | 136/136 | 0.051 [0.015, 0.088] | 1.495 [0.000, 6.974] (3 of 2000 resamples had flag rate 0; they are counted in the percentiles as undefined and are not dropped) | 3 of 2000 |
| laya_multi | all | 101/101 | 0.050 [0.010, 0.099] | 510/510 | 0.039 [0.024, 0.057] | 1.262 [0.281, 2.970] | 0 of 2000 |
| laya_multi_base | en | 88/88 | 0.000 [0.000, 0.000] | 374/374 | 0.005 [0.000, 0.013] | 0.000 [0.000, no value] (269 of 2000 resamples had flag rate 0, so the upper limit cannot be bounded) | 269 of 2000 |
| laya_multi_base | hi | 13/13 | 0.000 [0.000, 0.000] | 136/136 | 0.000 [0.000, 0.000] | undefined: flag rate is 0 at the validation threshold (0 of 136 corpus rows flagged), so the ratio is undefined | 2000 of 2000 |
| laya_multi_base | all | 101/101 | 0.000 [0.000, 0.000] | 510/510 | 0.004 [0.000, 0.010] | 0.000 [0.000, no value] (269 of 2000 resamples had flag rate 0, so the upper limit cannot be bounded) | 269 of 2000 |
| misinfo | en | 88/88 | 0.489 [0.386, 0.591] | 374/374 | 0.115 [0.083, 0.150] | 4.250 [2.994, 6.072] | 0 of 2000 |
| misinfo | hi | 0/13 | no score: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13) | 0/136 | no score: no valid score for any of 136 corpus title rows (skip_reason: unsupported_language=136) | no score: recall: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13); flag rate: no valid score for any of 136 corpus title rows (skip_reason: unsupported_language=136) | n/a of 2000 |
| misinfo | all | 88/101 | 0.489 [0.386, 0.591] | 374/510 | 0.115 [0.083, 0.150] | 4.250 [2.994, 6.072] | 0 of 2000 |
| tfidf | en | 88/88 | 0.034 [0.000, 0.080] | 374/374 | 0.003 [0.000, 0.008] | 12.750 [0.000, no value] (726 of 2000 resamples had flag rate 0, so the upper limit cannot be bounded) | 726 of 2000 |
| tfidf | hi | 0/13 | no score: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13) | 0/136 | no score: no valid score for any of 136 corpus title rows (skip_reason: unsupported_language=136) | no score: recall: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13); flag rate: no valid score for any of 136 corpus title rows (skip_reason: unsupported_language=136) | n/a of 2000 |
| tfidf | all | 88/101 | 0.034 [0.000, 0.080] | 374/510 | 0.003 [0.000, 0.008] | 12.750 [0.000, no value] (726 of 2000 resamples had flag rate 0, so the upper limit cannot be bounded) | 726 of 2000 |

## Recall, flag rate and lift: corpus full text (second value)

Lift = recall on known-false claims / flag rate on the unlabelled corpus, at the same threshold. Fact-check publishers are excluded from the corpus.

> **Not like for like.** Full-text flag rates and lifts are not like for like across model families. Laya scores long text with predict_long, which takes the maximum over windows of the whole text. The roberta classifier truncates the text at 256 tokens. A model that reads more of a long article has more chances to cross its threshold. Compare models on corpus_title, the primary figure, and read corpus_text as a second value.

| model | language | claims scored/rows | recall [95% CI] | corpus scored/rows | flag rate [95% CI] | lift [95% CI] | zero-flag resamples |
|---|---|---|---|---|---|---|---|
| laya_en | en | 88/88 | 0.045 [0.011, 0.091] | 374/374 | 0.051 [0.029, 0.072] (not like for like) | 0.895 [0.185, 2.125] (not like for like) | 0 of 2000 |
| laya_en | hi | 0/13 | no score: no valid score for any of 13 claim rows (skip_reason: unrouted_language=13) | 0/136 | no score: no valid score for any of 136 corpus text rows (skip_reason: unrouted_language=136) (not like for like) | no score: recall: no valid score for any of 13 claim rows (skip_reason: unrouted_language=13); flag rate: no valid score for any of 136 corpus text rows (skip_reason: unrouted_language=136) (not like for like) | n/a of 2000 |
| laya_en | all | 88/101 | 0.045 [0.011, 0.091] | 374/510 | 0.051 [0.029, 0.072] (not like for like) | 0.895 [0.185, 2.125] (not like for like) | 0 of 2000 |
| laya_multi | en | 88/88 | 0.045 [0.011, 0.091] | 374/374 | 0.075 [0.051, 0.102] (not like for like) | 0.607 [0.129, 1.417] (not like for like) | 0 of 2000 |
| laya_multi | hi | 13/13 | 0.077 [0.000, 0.231] | 136/136 | 0.331 [0.250, 0.412] (not like for like) | 0.232 [0.000, 0.805] (not like for like) | 0 of 2000 |
| laya_multi | all | 101/101 | 0.050 [0.010, 0.099] | 510/510 | 0.143 [0.114, 0.175] (not like for like) | 0.346 [0.072, 0.709] (not like for like) | 0 of 2000 |
| laya_multi_base | en | 88/88 | 0.000 [0.000, 0.000] | 374/374 | 0.061 [0.037, 0.086] (not like for like) | 0.000 [0.000, 0.000] (not like for like) | 0 of 2000 |
| laya_multi_base | hi | 13/13 | 0.000 [0.000, 0.000] | 136/136 | 0.191 [0.125, 0.257] (not like for like) | 0.000 [0.000, 0.000] (not like for like) | 0 of 2000 |
| laya_multi_base | all | 101/101 | 0.000 [0.000, 0.000] | 510/510 | 0.096 [0.071, 0.122] (not like for like) | 0.000 [0.000, 0.000] (not like for like) | 0 of 2000 |
| misinfo | en | 88/88 | 0.489 [0.386, 0.591] | 374/374 | 0.078 [0.051, 0.104] (not like for like) | 6.302 [4.365, 10.046] (not like for like) | 0 of 2000 |
| misinfo | hi | 0/13 | no score: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13) | 0/136 | no score: no valid score for any of 136 corpus text rows (skip_reason: unsupported_language=136) (not like for like) | no score: recall: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13); flag rate: no valid score for any of 136 corpus text rows (skip_reason: unsupported_language=136) (not like for like) | n/a of 2000 |
| misinfo | all | 88/101 | 0.489 [0.386, 0.591] | 374/510 | 0.078 [0.051, 0.104] (not like for like) | 6.302 [4.365, 10.046] (not like for like) | 0 of 2000 |
| tfidf | en | 88/88 | 0.034 [0.000, 0.080] | 374/374 | 0.174 [0.136, 0.211] (not like for like) | 0.196 [0.000, 0.451] (not like for like) | 0 of 2000 |
| tfidf | hi | 0/13 | no score: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13) | 0/136 | no score: no valid score for any of 136 corpus text rows (skip_reason: unsupported_language=136) (not like for like) | no score: recall: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13); flag rate: no valid score for any of 136 corpus text rows (skip_reason: unsupported_language=136) (not like for like) | n/a of 2000 |
| tfidf | all | 88/101 | 0.034 [0.000, 0.080] | 374/510 | 0.174 [0.136, 0.211] (not like for like) | 0.196 [0.000, 0.451] (not like for like) | 0 of 2000 |

## Hindi gate

Hindi scores enter fusion only with at least 50 Hindi claims and a title lift interval whose lower limit is above 1.

| model | Hindi claims scored | lift [95% CI] | result | reason |
|---|---|---|---|---|
| laya_en | 0 of 13 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 13 claim rows (skip_reason: unrouted_language=13); flag rate: no valid score for any of 136 corpus title rows (skip_reason: unrouted_language=136) |
| laya_multi | 13 of 13 | 1.495 [0.000, 6.974] | **FAIL** | 13 Hindi claims with a score, 50 needed; lift lower limit 0.000 is not above 1 |
| laya_multi_base | 13 of 13 | no value | **FAIL (control)** | control model, not a candidate: its numbers are a baseline for the fine-tuned models and it is never shipped. By the numbers alone: 13 Hindi claims with a score, 50 needed; no lift interval: flag rate is 0 at the validation threshold (0 of 136 corpus rows flagged), so the ratio is undefined |
| misinfo | 0 of 13 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13); flag rate: no valid score for any of 136 corpus title rows (skip_reason: unsupported_language=136) |
| tfidf | 0 of 13 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 13 claim rows (skip_reason: unsupported_language=13); flag rate: no valid score for any of 136 corpus title rows (skip_reason: unsupported_language=136) |

## Skipped and excluded rows

A null score is never counted as not flagged. Each one is left out of the denominator and counted here.

| model | part | rows | no score | skip_reason counts |
|---|---|---|---|---|
| laya_en | claims | 101 | 13 | unrouted_language=13 |
| laya_en | corpus_title | 510 | 136 | unrouted_language=136 |
| laya_en | corpus_text | 510 | 136 | unrouted_language=136 |
| laya_multi | claims | 101 | 0 | none |
| laya_multi | corpus_title | 510 | 0 | none |
| laya_multi | corpus_text | 510 | 0 | none |
| laya_multi_base | claims | 101 | 0 | none |
| laya_multi_base | corpus_title | 510 | 0 | none |
| laya_multi_base | corpus_text | 510 | 0 | none |
| misinfo | claims | 101 | 13 | unsupported_language=13 |
| misinfo | corpus_title | 510 | 136 | unsupported_language=136 |
| misinfo | corpus_text | 510 | 136 | unsupported_language=136 |
| tfidf | claims | 101 | 13 | unsupported_language=13 |
| tfidf | corpus_title | 510 | 136 | unsupported_language=136 |
| tfidf | corpus_text | 510 | 136 | unsupported_language=136 |

| model | claim ids excluded (reason=count) | kept claims absent from scores | corpus rows from fact-check publishers (title/text) | corpus rows without source_detail |
|---|---|---|---|---|
| laya_en | none | 0 | 120/120 | 0/0 |
| laya_multi | none | 0 | 120/120 | 0/0 |
| laya_multi_base | none | 0 | 120/120 | 0/0 |
| misinfo | none | 0 | 120/120 | 0/0 |
| tfidf | none | 0 | 120/120 | 0/0 |

## Inputs

| model | file | rows | sha256 |
|---|---|---|---|
| laya_en | laya_en.parquet | 12042 | `fd06a0ddd91eb91f43dc905c3f762e9e98f47a75872583559ad6a3cc28a719cd` |
| laya_multi | laya_multi.parquet | 12042 | `dd7139394bd1b7b6fe559853ff02fc2a2b62a38c83b5cdf291b0fc87e120a25a` |
| laya_multi_base | laya_multi_base.parquet | 12042 | `3bc8c8fe1f8d31b5b5cba500aed65c964ea8a58eb8100f14d5e0944208c10fe8` |
| misinfo | misinfo.parquet | 12042 | `7fb50a3f2cfdf2976f4608261c524a6293ef46b6d27e0b0fc1f5adb5a1870562` |
| tfidf | tfidf.parquet | 12042 | `c425c9c78ca4ca7ae6b304333e1b7e57230c84f81471b4f161c276925582111f` |

## Reproducibility

- seed: `20260813`
- input manifest hash: `d9c692247d0a301e`
- regenerate by rerunning `python -m modeling.cli evaluate-factchecks` on the same score files and the same seed
