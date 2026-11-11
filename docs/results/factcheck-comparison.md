# Fact-check evaluation: three-way comparison

> The fact-check set contains only the claims that publishers chose to debunk. It is not a sample of false claims in general, and every claim in it is false. Recall on it measures detection of publicly debunked claims and is never reported without the flag rate on the unlabelled corpus.

- Baseline: `tfidf`. Precision target: 0.85. Seed: `20260813`. Resamples: 2000. Intervals are 95% percentile bootstrap intervals.
- Split: grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)
- Evaluation claims: 1107 (frozen, 1107 manifest rows)

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
| laya_en | en | 90/90 | 0.056 [0.011, 0.111] | 612/612 | 0.039 [0.025, 0.054] | 1.417 [0.309, 3.238] | 0 of 2000 |
| laya_en | hi | 0/1016 | no score: no valid score for any of 1016 claim rows (skip_reason: unrouted_language=1016) | 0/211 | no score: no valid score for any of 211 corpus title rows (skip_reason: unrouted_language=211) | no score: recall: no valid score for any of 1016 claim rows (skip_reason: unrouted_language=1016); flag rate: no valid score for any of 211 corpus title rows (skip_reason: unrouted_language=211) | n/a of 2000 |
| laya_en | all | 90/1107 | 0.056 [0.011, 0.111] | 612/823 | 0.039 [0.025, 0.054] | 1.417 [0.309, 3.238] | 0 of 2000 |
| laya_multi | en | 90/90 | 0.044 [0.011, 0.089] | 612/612 | 0.031 [0.018, 0.044] | 1.432 [0.296, 3.549] | 0 of 2000 |
| laya_multi | hi | 1016/1016 | 0.104 [0.087, 0.123] | 211/211 | 0.057 [0.028, 0.090] | 1.834 [1.116, 3.981] | 0 of 2000 |
| laya_multi | all | 1106/1107 | 0.099 [0.082, 0.117] | 823/823 | 0.038 [0.026, 0.051] | 2.640 [1.819, 4.000] | 0 of 2000 |
| laya_multi_base | en | 90/90 | 0.000 [0.000, 0.000] | 612/612 | 0.007 [0.002, 0.013] | 0.000 [0.000, 0.000] (41 of 2000 resamples had flag rate 0; they are counted in the percentiles as undefined and are not dropped) | 41 of 2000 |
| laya_multi_base | hi | 1016/1016 | 0.005 [0.001, 0.010] | 211/211 | 0.000 [0.000, 0.000] | undefined: flag rate is 0 at the validation threshold (0 of 211 corpus rows flagged), so the ratio is undefined | 2000 of 2000 |
| laya_multi_base | all | 1106/1107 | 0.005 [0.001, 0.009] | 823/823 | 0.005 [0.001, 0.010] | 0.930 [0.186, 6.697] (41 of 2000 resamples had flag rate 0; they are counted in the percentiles as undefined and are not dropped) | 41 of 2000 |
| misinfo | en | 90/90 | 0.489 [0.389, 0.589] | 612/612 | 0.108 [0.083, 0.132] | 4.533 [3.352, 6.146] | 0 of 2000 |
| misinfo | hi | 0/1016 | no score: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016) | 0/211 | no score: no valid score for any of 211 corpus title rows (skip_reason: unsupported_language=211) | no score: recall: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016); flag rate: no valid score for any of 211 corpus title rows (skip_reason: unsupported_language=211) | n/a of 2000 |
| misinfo | all | 90/1107 | 0.489 [0.389, 0.589] | 612/823 | 0.108 [0.083, 0.132] | 4.533 [3.352, 6.146] | 0 of 2000 |
| tfidf | en | 90/90 | 0.033 [0.000, 0.078] | 612/612 | 0.002 [0.000, 0.005] | 20.400 [0.000, no value] (726 of 2000 resamples had flag rate 0, so the upper limit cannot be bounded) | 726 of 2000 |
| tfidf | hi | 0/1016 | no score: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016) | 0/211 | no score: no valid score for any of 211 corpus title rows (skip_reason: unsupported_language=211) | no score: recall: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016); flag rate: no valid score for any of 211 corpus title rows (skip_reason: unsupported_language=211) | n/a of 2000 |
| tfidf | all | 90/1107 | 0.033 [0.000, 0.078] | 612/823 | 0.002 [0.000, 0.005] | 20.400 [0.000, no value] (726 of 2000 resamples had flag rate 0, so the upper limit cannot be bounded) | 726 of 2000 |

## Recall, flag rate and lift: corpus full text (second value)

Lift = recall on known-false claims / flag rate on the unlabelled corpus, at the same threshold. Fact-check publishers are excluded from the corpus.

> **Not like for like.** Full-text flag rates and lifts are not like for like across model families. Laya scores long text with predict_long, which takes the maximum over windows of the whole text. The roberta classifier truncates the text at 256 tokens. A model that reads more of a long article has more chances to cross its threshold. Compare models on corpus_title, the primary figure, and read corpus_text as a second value.

| model | language | claims scored/rows | recall [95% CI] | corpus scored/rows | flag rate [95% CI] | lift [95% CI] | zero-flag resamples |
|---|---|---|---|---|---|---|---|
| laya_en | en | 90/90 | 0.056 [0.011, 0.111] | 612/612 | 0.041 [0.026, 0.057] (not like for like) | 1.360 [0.296, 3.092] (not like for like) | 0 of 2000 |
| laya_en | hi | 0/1016 | no score: no valid score for any of 1016 claim rows (skip_reason: unrouted_language=1016) | 0/211 | no score: no valid score for any of 211 corpus text rows (skip_reason: unrouted_language=211) (not like for like) | no score: recall: no valid score for any of 1016 claim rows (skip_reason: unrouted_language=1016); flag rate: no valid score for any of 211 corpus text rows (skip_reason: unrouted_language=211) (not like for like) | n/a of 2000 |
| laya_en | all | 90/1107 | 0.056 [0.011, 0.111] | 612/823 | 0.041 [0.026, 0.057] (not like for like) | 1.360 [0.296, 3.092] (not like for like) | 0 of 2000 |
| laya_multi | en | 90/90 | 0.044 [0.011, 0.089] | 612/612 | 0.067 [0.047, 0.088] (not like for like) | 0.663 [0.136, 1.488] (not like for like) | 0 of 2000 |
| laya_multi | hi | 1016/1016 | 0.104 [0.087, 0.123] | 211/211 | 0.284 [0.227, 0.341] (not like for like) | 0.367 [0.280, 0.487] (not like for like) | 0 of 2000 |
| laya_multi | all | 1106/1107 | 0.099 [0.082, 0.117] | 823/823 | 0.123 [0.101, 0.145] (not like for like) | 0.810 [0.632, 1.032] (not like for like) | 0 of 2000 |
| laya_multi_base | en | 90/90 | 0.000 [0.000, 0.000] | 612/612 | 0.056 [0.038, 0.075] (not like for like) | 0.000 [0.000, 0.000] (not like for like) | 0 of 2000 |
| laya_multi_base | hi | 1016/1016 | 0.005 [0.001, 0.010] | 211/211 | 0.185 [0.133, 0.242] (not like for like) | 0.027 [0.005, 0.059] (not like for like) | 0 of 2000 |
| laya_multi_base | all | 1106/1107 | 0.005 [0.001, 0.009] | 823/823 | 0.089 [0.070, 0.108] (not like for like) | 0.051 [0.010, 0.106] (not like for like) | 0 of 2000 |
| misinfo | en | 90/90 | 0.489 [0.389, 0.589] | 612/612 | 0.075 [0.056, 0.096] (not like for like) | 6.504 [4.610, 9.521] (not like for like) | 0 of 2000 |
| misinfo | hi | 0/1016 | no score: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016) | 0/211 | no score: no valid score for any of 211 corpus text rows (skip_reason: unsupported_language=211) (not like for like) | no score: recall: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016); flag rate: no valid score for any of 211 corpus text rows (skip_reason: unsupported_language=211) (not like for like) | n/a of 2000 |
| misinfo | all | 90/1107 | 0.489 [0.389, 0.589] | 612/823 | 0.075 [0.056, 0.096] (not like for like) | 6.504 [4.610, 9.521] (not like for like) | 0 of 2000 |
| tfidf | en | 90/90 | 0.033 [0.000, 0.078] | 612/612 | 0.134 [0.108, 0.162] (not like for like) | 0.249 [0.000, 0.591] (not like for like) | 0 of 2000 |
| tfidf | hi | 0/1016 | no score: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016) | 0/211 | no score: no valid score for any of 211 corpus text rows (skip_reason: unsupported_language=211) (not like for like) | no score: recall: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016); flag rate: no valid score for any of 211 corpus text rows (skip_reason: unsupported_language=211) (not like for like) | n/a of 2000 |
| tfidf | all | 90/1107 | 0.033 [0.000, 0.078] | 612/823 | 0.134 [0.108, 0.162] (not like for like) | 0.249 [0.000, 0.591] (not like for like) | 0 of 2000 |

## Hindi gate

Hindi scores enter fusion only with at least 50 Hindi claims and a title lift interval whose lower limit is above 1.

| model | Hindi claims scored | lift [95% CI] | result | reason |
|---|---|---|---|---|
| laya_en | 0 of 1016 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 1016 claim rows (skip_reason: unrouted_language=1016); flag rate: no valid score for any of 211 corpus title rows (skip_reason: unrouted_language=211) |
| laya_multi | 1016 of 1016 | 1.834 [1.116, 3.981] | **PASS** | 1016 Hindi claims and lift lower limit 1.116 above 1 |
| laya_multi_base | 1016 of 1016 | no value | **FAIL (control)** | control model, not a candidate: its numbers are a baseline for the fine-tuned models and it is never shipped. By the numbers alone: no lift interval: flag rate is 0 at the validation threshold (0 of 211 corpus rows flagged), so the ratio is undefined |
| misinfo | 0 of 1016 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016); flag rate: no valid score for any of 211 corpus title rows (skip_reason: unsupported_language=211) |
| tfidf | 0 of 1016 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 1016 claim rows (skip_reason: unsupported_language=1016); flag rate: no valid score for any of 211 corpus title rows (skip_reason: unsupported_language=211) |

## Skipped and excluded rows

A null score is never counted as not flagged. Each one is left out of the denominator and counted here.

| model | part | rows | no score | skip_reason counts |
|---|---|---|---|---|
| laya_en | claims | 1107 | 1017 | unrouted_language=1017 |
| laya_en | corpus_title | 823 | 211 | unrouted_language=211 |
| laya_en | corpus_text | 823 | 211 | unrouted_language=211 |
| laya_multi | claims | 1107 | 1 | unrouted_language=1 |
| laya_multi | corpus_title | 823 | 0 | none |
| laya_multi | corpus_text | 823 | 0 | none |
| laya_multi_base | claims | 1107 | 1 | unrouted_language=1 |
| laya_multi_base | corpus_title | 823 | 0 | none |
| laya_multi_base | corpus_text | 823 | 0 | none |
| misinfo | claims | 1107 | 1017 | unsupported_language=1017 |
| misinfo | corpus_title | 823 | 211 | unsupported_language=211 |
| misinfo | corpus_text | 823 | 211 | unsupported_language=211 |
| tfidf | claims | 1107 | 1017 | unsupported_language=1017 |
| tfidf | corpus_title | 823 | 211 | unsupported_language=211 |
| tfidf | corpus_text | 823 | 211 | unsupported_language=211 |

| model | claim ids excluded (reason=count) | kept claims absent from scores | corpus rows from fact-check publishers (title/text) | corpus rows without source_detail |
|---|---|---|---|---|
| laya_en | none | 0 | 126/126 | 0/0 |
| laya_multi | none | 0 | 126/126 | 0/0 |
| laya_multi_base | none | 0 | 126/126 | 0/0 |
| misinfo | none | 0 | 126/126 | 0/0 |
| tfidf | none | 0 | 126/126 | 0/0 |

## Inputs

| model | file | rows | sha256 |
|---|---|---|---|
| laya_en | laya_en.parquet | 13686 | `eaad8c50438d2788db066edeb049a1f429dc451bef211f51c1d9b64c551a1050` |
| laya_multi | laya_multi.parquet | 13686 | `2eef8c43dce91d74e4148c607217625ed34df656d732cff375ab1031ad1413be` |
| laya_multi_base | laya_multi_base.parquet | 13686 | `c2fcde9204d8fd55e8cec400922e9848b91ff6406df35737674ae35b647f477a` |
| misinfo | misinfo.parquet | 13686 | `f677946c62300e39b101f6caa687afadf2da73dc81f8e124685fc8a80658e175` |
| tfidf | tfidf.parquet | 13686 | `01b646b8091064d9ef0e1937972fda64a5e90978cc987d2f4814597d42b87550` |

## Reproducibility

- seed: `20260813`
- input manifest hash: `82396c7a3b50090e`
- regenerate by rerunning `python -m modeling.cli evaluate-factchecks` on the same score files and the same seed
