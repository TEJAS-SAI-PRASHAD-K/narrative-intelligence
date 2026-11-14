# Fusion scoring

The single number the UI ranks narratives by, and everything behind it.

This document is the reference for `app/scoring/fusion.py`, `configs/fusion.yaml`
and `GET /api/v1/narratives/{id}/score`. If the code and this file disagree, the
code is right and this file is a bug.

---

## 1. Where the numbers live, and why not in `configs/scoring.yaml`

The Phase 4 brief asked for `configs/scoring.yaml`. That file already exists and
belongs to Phase 2: it declares which batch scoring stages run, in what order,
and in what chunk sizes. Overwriting it would break `modeling score --all`.

The fusion weights therefore live in **`configs/fusion.yaml`**. Nothing else
changed — weights and thresholds are still config, still versioned, still read
per run rather than baked into code.

## 2. The formula

```
fusion = 100 * (w1*narrative_severity + w2*coordination + w3*authenticity)
```

with, from `configs/fusion.yaml` at `version: fusion-v1.2.0`:

| component | weight | what it measures |
|---|---:|---|
| `narrative_severity` | 0.45 | how harmful the content is |
| `coordination` | 0.35 | how inauthentically it spread |
| `authenticity` | 0.20 | whether its sources are what they claim to be |

Each component is itself a weighted mean of named sub-signals:

**narrative_severity**

| sub-signal | weight | definition |
|---|---:|---|
| `misinfo_likelihood_agg` | 0.40 | engagement-weighted mean of member posts' misinformation likelihood |
| `compass_risk` | 0.25 | Compass verification status mapped to [0,1] |
| `toxicity` | 0.20 | mean toxicity over scored member posts |
| `negative_sentiment` | 0.15 | share of scored member posts classified negative |

**coordination**

| sub-signal | weight | definition |
|---|---:|---|
| `coordinated_author_ratio` | 0.35 | share of member authors with a coordination score above 0.50, over authors that have a score |
| `co_post_similarity_density` | 0.25 | internal edge density against a complete graph on the same author set |
| `temporal_burstiness` | 0.25 | share of volume in buckets above 3× the narrative's own mean |
| `cohort_concentration` | 0.15 | share of member posts from the single largest cohort |

**authenticity**

| sub-signal | weight | definition |
|---|---:|---|
| `domain_risk_agg` | 0.5833 | mean risk score of linked domains |
| `anonymous_author_ratio` | 0.4167 | share of member authors deleted or without a handle |

The two weights keep the 0.35 to 0.25 proportion that these signals had in
`fusion-v1.1.0`. In that version, a third signal (`deepfake_hits`) shared the
component. A narrative that had no value for that signal gets the same score
in `fusion-v1.2.0`, apart from rounding below 0.01 points.

The explanation payload changes for that narrative. The `missing` list of
`components[authenticity]` no longer names `deepfake_hits`. The value of
`weights_renormalized` is now false. The `deepfake_hits` input key is gone.

`misinfo_likelihood_agg` is engagement-weighted because a false claim nobody saw
is not the same problem as one that reached a hundred thousand people. The weight
is `1 + likes + shares`, so a zero-engagement post still contributes its own
weight rather than vanishing from the average.

`coordinated_author_ratio` divides by the authors that have a coordination score.
An author that the detector saw but linked to nobody has a score of 0.0. That is a
measured zero and it stays in the denominator. An author that the detector did not
see has no score. Examples are an author from a source that is not eligible, such
as news, and any author when the coordination stage did not run. Those authors are
excluded. If no member author has a score, the signal is `null`. The fusion then
treats it like each other missing signal and shares its weight among the other
coordination signals.

The score of an author is the mean of three terms in [0, 1]. The terms are the size
of the community of the author, the mean edge weight of the author, and the share
of the posts of the author in an edge. The threshold is
`thresholds.coordination_score_threshold` in `configs/fusion.yaml`. Its value is 0.50.
The value is a placeholder. It is not tuned. It has not been measured on a real
score distribution. An author with an edge has a floor from the size term and the
participation term, so most authors with an edge can exceed 0.50. A score equal to
the threshold is not above it.

**Why the bot ratio left the formula.** The bot classifier does not generalize.
Its mean-fold macro-F1 is 0.416, and it flags 39 percent of genuine accounts
(`docs/roadmap.md`). A signal with that error rate is a poor input to a rank.
`bot_like_ratio` is still collected. The system stores it in `sc.bot_like` and the
alert metric `bot_like_ratio` still reads it. The fusion formula does not read it.

**Limit of the new signal.** The coordination score comes from a detector whose
modularity did not exceed the time-shuffled null on this corpus (0.912 against
0.979). The signal is weak evidence. The change improves the input of the
formula. It does not show that coordination exists in this corpus.

## 3. Normalization

**Percentile-rank within the project.** Every sub-signal is ranked against the
distribution of that same signal across the project's other narratives.

The alternative is min-max against a fixed reference range. Both are defensible
and mixing them is not — a codebase with some signals percentile-ranked and
others min-maxed produces scores that are individually reasonable and mutually
incomparable, and nobody notices until two numbers that should agree do not.

Percentile is the choice because this is a **within-case triage tool**. The
question an analyst asks is "which of these narratives do I look at first", not
"is this narrative worse than one in a different investigation". A fixed
reference range is a guess that ages badly and that one extreme outlier flattens.

The cost is real and worth stating: a project where nothing is coordinated still
has a top decile. The percentile says "highest here", not "high in absolute
terms", and the UI must not imply otherwise.

**Below 20 scored narratives** (`normalization.min_sample`) there is no
meaningful distribution to rank against. The raw value is clamped to [0,1] and
used directly, and the component records that it fell back.

## 4. Missing inputs

The rule, enforced in `app/scoring/normalize.py:weighted_mean`:

> **A missing input is not a zero. The weights renormalize over what is present.**

If `authenticity` has no input, it is `null`, it appears in
`missing`, and `narrative_severity` and `coordination` are scaled from 0.45/0.35
to 0.5625/0.4375 so they still sum to 1.

Treating absence as zero would systematically under-flag every narrative nobody
has analysed yet — the ones most likely to need attention. `tests/test_scoring.py`
asserts the renormalized score is strictly higher than the same narrative with
authenticity forced to 0.0, and covers all six missing-component combinations.

Two distinctions the schema preserves that are easy to collapse:

* **No Compass context** (`null`) versus **a context that found nothing**
  (`insufficient_evidence`, worth 0.5). Failing to source a claim is not evidence
  that the claim is harmless.
* **A narrative with no signals at all** scores `null`, not 0. A score computed
  from no inputs is not a low score; it is not a score. Its priority is `low`
  rather than `high` — escalating unmeasured narratives would flood the queue —
  and `missing` names all three components so the UI can say why.

## 5. Priority

Derived from `fusion_score` by the bounds in `configs/fusion.yaml`, never
assigned by hand, so two narratives with the same score cannot carry different
priorities.

| priority | fusion score |
|---|---|
| `high` | ≥ 70 |
| `medium` | ≥ 40 |
| `low` | < 40, or unscorable |

## 6. Worked example

Narrative `6cd364c8-9412-5c99-995c-a281a23cad0d` from the loaded corpus, scored
by `score.fusion` at `fusion-v1.0.0`. That version still used `bot_like_ratio` in
the coordination component. The arithmetic of the combination is the same at
`fusion-v1.2.0`.

Component values as returned by `GET /api/v1/narratives/{id}/score`:

| component | value | applied weight | contribution |
|---|---:|---:|---:|
| `narrative_severity` | 0.7333 | 0.45 | 0.3300 |
| `coordination` | 0.9250 | 0.35 | 0.3237 |
| `authenticity` | 0.0000 | 0.20 | 0.0000 |

By hand:

```
0.7333 * 0.45 = 0.32999
0.9250 * 0.35 = 0.32375
0.0000 * 0.20 = 0.00000
                --------
                0.65374
100 * 0.65374 = 65.37  ->  API returns 65.38 (component values are rounded to
                           4dp in the response; the score is computed from the
                           unrounded values)
```

`priority` is `medium`: 65.38 is above 40 and below 70. `missing` is empty and
`weights_renormalized` is `false`, so the applied weights equal the configured
ones.

Note `authenticity = 0.0`, not `null`. That is a **measured** zero: its linked
domains scored low. A null would mean that nothing was measured. The response
distinguishes them and so must any reader.

## 7. Changing the weights

```bash
# edit configs/fusion.yaml: bump `version`, change a weight
python -m app.etl.cli load --project <slug>     # if the corpus changed
celery -A app.tasks.celery_app.celery call score.fusion --kwargs '{"project_id":"<slug>"}'
```

`load_config()` is deliberately **not** cached, so a rerun picks up an edit
without restarting the worker. Every scorecard the run writes carries the new
`scoring_version`, which is how a stale score is detectable rather than merely
suspected: rows with two different versions are not comparable and the API says
which version produced each.

## 8. What this score is not

* It is **not a model**. No training data, no fitted parameters, no confidence
  interval. It is a weighted sum of other models' outputs with hand-chosen
  weights, and the weights encode an editorial judgement about what matters.
* It is **not calibrated**. A 70 does not mean "70% likely to be a coordinated
  influence operation". It means "ranks in the upper range of this project on a
  weighted combination of these eleven signals".
* It **inherits every upstream limitation**. The coordination detector did not
  beat its time-shuffled null on this corpus, so `coordination` is the weakest of
  the three components. The bot classifier (per-fold macro-F1 0.416 ± 0.187 against
  a pooled 0.705) no longer feeds the formula. Its caveat stays on
  `/authors/{id}/score`.

The score is a triage aid. The drilldown is the product.


## 9. Scoring stage

The `laya` batch stage in `modeling/scoring.py` adds `record_scores.misinfo_prob` for Hindi records. It runs after the `misinfo` stage and before `cluster`. The `scoring_languages` key of the `laya` module in `configs/models.yaml` sets its languages. The value is `[hi]`.

The stage scores a record only when its language is in `scoring_languages` and `misinfo` did not score it. It never writes for English. It never overwrites a non-null `misinfo_prob`.

The stage scores the title of the record, not the full text. The Hindi gate passed on titles with a lift of 1.760 [1.286, 2.630]. On Hindi full text, the same model has a flag rate of 0.258 and a lift of 0.476 [0.401, 0.573]. The upper limit is below 1. Full text is outside what the gate measured.

The stage reads the title as the comparison script reads `corpus_title`. A record with no title gets a null value and the reason `laya:no_title`. The rule `text_too_short` applies to the title.

The stage writes the output of a calibrator, not the raw Laya score. The raw score has a mean confidence of about 0.97 and an expected calibration error of 0.159. The calibrator files are in the checkpoint directory. The command `python scripts/run_comparison.py calibrate-laya` fits them on the validation rows of the comparison.

The fit refuses raw scores with a ROC-AUC of 0.60 or less. The fit and the load both refuse a map with fewer than 5 distinct values. The calibrator file records the demo flag and the identity of the checkpoint. The fit refuses a directory that does not hold the comparison checkpoint.

The stage refuses to score in these cases. `laya` is not installed (`model_unavailable`). The checkpoint is missing (`checkpoint_missing`).

The calibrator can be missing, unreadable or rejected (`calibrator_missing`, `calibrator_unreadable`, `calibrator_rejected`). The calibrator has the demo mark (`calibrator_demo`). The calibrator was fitted for other weights (`calibrator_checkpoint_mismatch`). Each Hindi record then gets a `laya:<reason>` skip reason, a null `misinfo_prob`, and no `laya` stamp.

The stage reports, and does not refuse, when the values collapse. The summary and the log give the count of raw values outside the fitted range and the count of distinct calibrated values. The log warns when fewer than 5 values are distinct, or when more than half of the raw values are out of range.

The calibrator is fitted on English benchmark validation labels. Its calibration on Hindi is not measured. A Hindi value and an English value come from different models and different calibrations. Fusion mixes both values in `misinfo_likelihood_agg`. A narrative with many Hindi posts has a less certain severity value.

A Hindi row keeps its `misinfo:unsupported_language` skip reason when `laya` writes the value. The field `skip_reasons` records what each module did. The field `model_versions` says which model wrote the value.
