# Fact-check evaluation: three-way comparison

> The fact-check set contains only the claims that publishers chose to debunk. It is not a sample of false claims in general, and every claim in it is false. Recall on it measures detection of publicly debunked claims and is never reported without the flag rate on the unlabelled corpus.

- Baseline: `tfidf`. Precision target: 0.85. Seed: `20260813`. Resamples: 2000. Intervals are 95% percentile bootstrap intervals.
- Split: grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)
- Evaluation claims: 4428 (frozen, 4428 manifest rows)

- Fact-check rows per extraction tier: `claimreview` 4420, `rss_tag` 9

- Evaluation claims per extraction tier: `claimreview` 4419, `rss_tag` 9

## Claim deduplication (read side)

`factchecks.parquet` is never modified. Each collapsed row is counted under the pass that removed it. A pass that did not run has no count, which is not the same as a measured zero.

| pass | status | collapsed | detail |
|---|---|---|---|
| simhash | ok | 29 | 4879 rows considered; hamming radius 3 |
| embedding | ok | 421 | 4850 rows considered; cosine >= 0.85; `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; 4850 embedded |

embedding precision of collapses: 0.390 (78 of 200 labelled link pairs are the same claim), Wilson 95% interval [0.325, 0.459], kappa 0.927, agreement 0.965 on 200 double-labelled pairs. PROVISIONAL, not a measured precision. labeller: model, unreviewed by a person. Reason code labeller_not_human. Direct links (similarity to the representative at or above 0.85): 70 of 124 = 0.565 [0.477, 0.649]. Chain links (below 0.85, joined through other members): 8 of 76 = 0.105 [0.054, 0.194]. 0.103 of the same-claim pairs are chain links. The chain band pools links of different similarity, so it is heterogeneous. The sample takes evenly spaced ranks of all links, so each link has about the same chance of selection. With the full sample labelled, the headline is close to an unbiased precision over all links. Post-stratified cross-check, not a correction: 0.389 (each band's precision times the share of all 421 links in that band; no interval). Sample `data/factchecks/claim_dedup_sample.csv`.

| link type and similarity band | pairs | same claim | precision | Wilson 95% interval |
|---|---|---|---|---|
| direct [0.85, 0.9) | 79 | 33 | 0.418 | [0.315, 0.528] |
| direct [0.9, 0.95) | 34 | 27 | 0.794 | [0.632, 0.897] |
| direct [0.95, 1.0] | 11 | 10 | 0.909 | [0.623, 0.984] |
| chain [0.0, 0.85) | 76 | 8 | 0.105 | [0.054, 0.194] |

A measured precision needs labels from a person: `labeller_kind` `human`, two different labeller names on each row, at least one double-labelled pair, and pairs that are in the current links.

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
| laya_en | en | 93/93 | 0.065 [0.022, 0.118] | 2205/5706 | 0.017 [0.012, 0.022] | 3.845 [1.185, 7.903] | 0 of 2000 |
| laya_en | hi | 0/4334 | no score: no valid score for any of 4334 claim rows (skip_reason: unrouted_language=4334) | 0/721 | no score: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unrouted_language=717) | no score: recall: no valid score for any of 4334 claim rows (skip_reason: unrouted_language=4334); flag rate: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unrouted_language=717) | n/a of 2000 |
| laya_en | all | 93/4428 | 0.065 [0.022, 0.118] | 2205/6427 | 0.017 [0.012, 0.022] | 3.845 [1.185, 7.903] | 0 of 2000 |
| laya_multi | en | 93/93 | 0.054 [0.011, 0.108] | 2205/5706 | 0.024 [0.018, 0.031] | 2.195 [0.447, 4.639] | 0 of 2000 |
| laya_multi | hi | 4334/4334 | 0.119 [0.109, 0.128] | 717/721 | 0.068 [0.050, 0.086] | 1.735 [1.331, 2.368] | 0 of 2000 |
| laya_multi | all | 4427/4428 | 0.117 [0.108, 0.126] | 2922/6427 | 0.035 [0.029, 0.042] | 3.326 [2.759, 4.105] | 0 of 2000 |
| laya_multi_base | en | 93/93 | 0.000 [0.000, 0.000] | 2205/5706 | 0.005 [0.002, 0.007] | 0.000 [0.000, 0.000] | 0 of 2000 |
| laya_multi_base | hi | 4334/4334 | 0.007 [0.005, 0.010] | 717/721 | 0.004 [0.000, 0.010] | 1.765 [0.717, no value] (95 of 2000 resamples had flag rate 0, so the upper limit cannot be bounded) | 95 of 2000 |
| laya_multi_base | all | 4427/4428 | 0.007 [0.005, 0.010] | 2922/6427 | 0.004 [0.002, 0.007] | 1.625 [0.889, 3.394] | 0 of 2000 |
| misinfo | en | 93/93 | 0.516 [0.419, 0.613] | 2205/5706 | 0.063 [0.053, 0.074] | 8.188 [6.231, 10.538] | 0 of 2000 |
| misinfo | hi | 0/4334 | no score: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334) | 0/721 | no score: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unsupported_language=717) | no score: recall: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334); flag rate: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unsupported_language=717) | n/a of 2000 |
| misinfo | all | 93/4428 | 0.516 [0.419, 0.613] | 2205/6427 | 0.063 [0.053, 0.074] | 8.188 [6.231, 10.538] | 0 of 2000 |
| tfidf | en | 93/93 | 0.032 [0.000, 0.075] | 2205/5706 | 0.004 [0.002, 0.007] | 7.903 [0.000, 28.452] | 0 of 2000 |
| tfidf | hi | 0/4334 | no score: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334) | 0/721 | no score: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unsupported_language=717) | no score: recall: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334); flag rate: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unsupported_language=717) | n/a of 2000 |
| tfidf | all | 93/4428 | 0.032 [0.000, 0.075] | 2205/6427 | 0.004 [0.002, 0.007] | 7.903 [0.000, 28.452] | 0 of 2000 |

## Recall, flag rate and lift: corpus full text (second value)

Lift = recall on known-false claims / flag rate on the unlabelled corpus, at the same threshold. Fact-check publishers are excluded from the corpus.

> **Not like for like.** Full-text flag rates and lifts are not like for like across model families. Laya scores long text with predict_long, which takes the maximum over windows of the whole text. The roberta classifier truncates the text at 256 tokens. A model that reads more of a long article has more chances to cross its threshold. Compare models on corpus_title, the primary figure, and read corpus_text as a second value.

| model | language | claims scored/rows | recall [95% CI] | corpus scored/rows | flag rate [95% CI] | lift [95% CI] | zero-flag resamples |
|---|---|---|---|---|---|---|---|
| laya_en | en | 93/93 | 0.065 [0.022, 0.118] | 5706/5706 | 0.015 [0.012, 0.018] (not like for like) | 4.281 [1.278, 8.273] (not like for like) | 0 of 2000 |
| laya_en | hi | 0/4334 | no score: no valid score for any of 4334 claim rows (skip_reason: unrouted_language=4334) | 0/721 | no score: no valid score for any of 721 corpus text rows (skip_reason: unrouted_language=721) (not like for like) | no score: recall: no valid score for any of 4334 claim rows (skip_reason: unrouted_language=4334); flag rate: no valid score for any of 721 corpus text rows (skip_reason: unrouted_language=721) (not like for like) | n/a of 2000 |
| laya_en | all | 93/4428 | 0.065 [0.022, 0.118] | 5706/6427 | 0.015 [0.012, 0.018] (not like for like) | 4.281 [1.278, 8.273] (not like for like) | 0 of 2000 |
| laya_multi | en | 93/93 | 0.054 [0.011, 0.108] | 5706/5706 | 0.019 [0.015, 0.022] (not like for like) | 2.841 [0.584, 5.752] (not like for like) | 0 of 2000 |
| laya_multi | hi | 4334/4334 | 0.119 [0.109, 0.128] | 721/721 | 0.236 [0.204, 0.266] (not like for like) | 0.503 [0.432, 0.590] (not like for like) | 0 of 2000 |
| laya_multi | all | 4427/4428 | 0.117 [0.108, 0.126] | 6427/6427 | 0.043 [0.038, 0.049] (not like for like) | 2.710 [2.343, 3.111] (not like for like) | 0 of 2000 |
| laya_multi_base | en | 93/93 | 0.000 [0.000, 0.000] | 5706/5706 | 0.016 [0.013, 0.020] (not like for like) | 0.000 [0.000, 0.000] (not like for like) | 0 of 2000 |
| laya_multi_base | hi | 4334/4334 | 0.007 [0.005, 0.010] | 721/721 | 0.179 [0.150, 0.207] (not like for like) | 0.041 [0.027, 0.058] (not like for like) | 0 of 2000 |
| laya_multi_base | all | 4427/4428 | 0.007 [0.005, 0.010] | 6427/6427 | 0.035 [0.030, 0.039] (not like for like) | 0.208 [0.139, 0.292] (not like for like) | 0 of 2000 |
| misinfo | en | 93/93 | 0.516 [0.419, 0.613] | 5706/5706 | 0.047 [0.042, 0.053] (not like for like) | 10.948 [8.576, 13.552] (not like for like) | 0 of 2000 |
| misinfo | hi | 0/4334 | no score: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334) | 0/721 | no score: no valid score for any of 721 corpus text rows (skip_reason: unsupported_language=721) (not like for like) | no score: recall: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334); flag rate: no valid score for any of 721 corpus text rows (skip_reason: unsupported_language=721) (not like for like) | n/a of 2000 |
| misinfo | all | 93/4428 | 0.516 [0.419, 0.613] | 5706/6427 | 0.047 [0.042, 0.053] (not like for like) | 10.948 [8.576, 13.552] (not like for like) | 0 of 2000 |
| tfidf | en | 93/93 | 0.032 [0.000, 0.075] | 5706/5706 | 0.029 [0.024, 0.033] (not like for like) | 1.122 [0.000, 2.575] (not like for like) | 0 of 2000 |
| tfidf | hi | 0/4334 | no score: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334) | 0/721 | no score: no valid score for any of 721 corpus text rows (skip_reason: unsupported_language=721) (not like for like) | no score: recall: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334); flag rate: no valid score for any of 721 corpus text rows (skip_reason: unsupported_language=721) (not like for like) | n/a of 2000 |
| tfidf | all | 93/4428 | 0.032 [0.000, 0.075] | 5706/6427 | 0.029 [0.024, 0.033] (not like for like) | 1.122 [0.000, 2.575] (not like for like) | 0 of 2000 |

## Tier 4 claims read from headlines (separate cell)

> Tier 4 claims are headlines that a pattern reader turned into a verdict. They are inference, not a verdict that a publisher stated. The claim text of a tier 4 row is the debunk headline, for example 'Video Is AI-Generated', and not the false claim itself. The recall of a model on this cell therefore measures how well it detects debunk language, not how well it detects a false claim. The cell is reported apart, and tier 4 counts in the primary figures only when the config says so.

- In the primary evaluation: **no**.
- Tier 4 known-false claims: 0.
- Measured error rate: not measured (the label sheet is not labelled yet).

No tier 4 claim is present in the fact-check rows, so there is no cell.

## Hindi gate

Hindi scores enter fusion only with at least 50 Hindi claims and a title lift interval whose lower limit is above 1.

| model | Hindi claims scored | lift [95% CI] | result | reason |
|---|---|---|---|---|
| laya_en | 0 of 4334 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 4334 claim rows (skip_reason: unrouted_language=4334); flag rate: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unrouted_language=717) |
| laya_multi | 4334 of 4334 | 1.735 [1.331, 2.368] | **PASS** | 4334 Hindi claims and lift lower limit 1.331 above 1 |
| laya_multi_base | 4334 of 4334 | 1.765 [no interval] | **FAIL (control)** | control model, not a candidate: its numbers are a baseline for the fine-tuned models and it is never shipped. By the numbers alone: lift lower limit 0.717 is not above 1 |
| misinfo | 0 of 4334 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334); flag rate: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unsupported_language=717) |
| tfidf | 0 of 4334 | no value | **FAIL** | 0 Hindi claims with a score, 50 needed; no lift interval: recall: no valid score for any of 4334 claim rows (skip_reason: unsupported_language=4334); flag rate: no valid score for any of 721 corpus title rows (skip_reason: no_title=4, unsupported_language=717) |

## Hindi claims by publisher

4334 Hindi evaluation claims from 3 publisher(s). Share is the publisher's Hindi claims over all Hindi evaluation claims.

| publisher | Hindi claims | share |
|---|---|---|
| vishvasnews.com | 3789 | 87.4% |
| boomlive.in | 524 | 12.1% |
| factcrescendo.com | 21 | 0.5% |

## Skipped and excluded rows

A null score is never counted as not flagged. Each one is left out of the denominator and counted here.

| model | part | rows | no score | skip_reason counts |
|---|---|---|---|---|
| laya_en | claims | 4428 | 4335 | unrouted_language=4335 |
| laya_en | corpus_title | 6427 | 4222 | no_title=3505, unrouted_language=717 |
| laya_en | corpus_text | 6427 | 721 | unrouted_language=721 |
| laya_multi | claims | 4428 | 1 | unrouted_language=1 |
| laya_multi | corpus_title | 6427 | 3505 | no_title=3505 |
| laya_multi | corpus_text | 6427 | 0 | none |
| laya_multi_base | claims | 4428 | 1 | unrouted_language=1 |
| laya_multi_base | corpus_title | 6427 | 3505 | no_title=3505 |
| laya_multi_base | corpus_text | 6427 | 0 | none |
| misinfo | claims | 4428 | 4335 | unsupported_language=4335 |
| misinfo | corpus_title | 6427 | 4222 | no_title=3505, unsupported_language=717 |
| misinfo | corpus_text | 6427 | 721 | unsupported_language=721 |
| tfidf | claims | 4428 | 4335 | unsupported_language=4335 |
| tfidf | corpus_title | 6427 | 4222 | no_title=3505, unsupported_language=717 |
| tfidf | corpus_text | 6427 | 721 | unsupported_language=721 |

| model | claim ids excluded (reason=count) | kept claims absent from scores | corpus rows from fact-check publishers (title/text) | corpus rows without source_detail |
|---|---|---|---|---|
| laya_en | collapsed_duplicate_embedding=421, label_not_false=1 | 0 | 265/265 | 0/0 |
| laya_multi | collapsed_duplicate_embedding=421, label_not_false=1 | 0 | 265/265 | 0/0 |
| laya_multi_base | collapsed_duplicate_embedding=421, label_not_false=1 | 0 | 265/265 | 0/0 |
| misinfo | collapsed_duplicate_embedding=421, label_not_false=1 | 0 | 265/265 | 0/0 |
| tfidf | collapsed_duplicate_embedding=421, label_not_false=1 | 0 | 265/265 | 0/0 |

### Corpus rows excluded per fact-checker domain

Every listed domain appears, also with 0 rows.

| domain | corpus_title | corpus_text |
|---|---|---|
| altnews.in | 34 | 34 |
| boomlive.in | 66 | 66 |
| digiteye.in | 9 | 9 |
| factcrescendo.com | 24 | 24 |
| factly.in | 22 | 22 |
| newschecker.in | 29 | 29 |
| thequint.com | 55 | 55 |
| vishvasnews.com | 26 | 26 |
| total | 265 | 265 |

laya_en: 0/0 corpus_title/corpus_text rows are on a subdomain of a fact-checker domain (`fact_check_subdomain_not_excluded`). The exclusion rule does not match them, so they stay in the flag rate.

laya_multi: 0/0 corpus_title/corpus_text rows are on a subdomain of a fact-checker domain (`fact_check_subdomain_not_excluded`). The exclusion rule does not match them, so they stay in the flag rate.

laya_multi_base: 0/0 corpus_title/corpus_text rows are on a subdomain of a fact-checker domain (`fact_check_subdomain_not_excluded`). The exclusion rule does not match them, so they stay in the flag rate.

misinfo: 0/0 corpus_title/corpus_text rows are on a subdomain of a fact-checker domain (`fact_check_subdomain_not_excluded`). The exclusion rule does not match them, so they stay in the flag rate.

tfidf: 0/0 corpus_title/corpus_text rows are on a subdomain of a fact-checker domain (`fact_check_subdomain_not_excluded`). The exclusion rule does not match them, so they stay in the flag rate.

## Inputs

| model | file | rows | sha256 |
|---|---|---|---|
| laya_en | laya_en.parquet | 28915 | `74deaf9030429521961f37b005ac1b7a57bb3bd4e8a36afd9b7d73b3f8e40898` |
| laya_multi | laya_multi.parquet | 28915 | `3d01a33ef62173c4fb7178f4b6a885dd226f69f4e2d578116c43466c32e9d242` |
| laya_multi_base | laya_multi_base.parquet | 28915 | `f8b97889790b961b70be40deaf275e0c2a0686f93a74e86a1b08a8cb18464cb2` |
| misinfo | misinfo.parquet | 28915 | `de62e3c61e26e473f743230516db5326abefaa3a3bc9e4091743ebf75a90f9fc` |
| tfidf | tfidf.parquet | 28915 | `6f20c91cf809a42d5d2c5c457136f6604e04aee479453794b4d8ad182e011520` |

## Reproducibility

- seed: `20260813`
- input manifest hash: `865cc7215e6fbf9c`
- regenerate by rerunning `python -m modeling.cli evaluate-factchecks` on the same score files and the same seed
