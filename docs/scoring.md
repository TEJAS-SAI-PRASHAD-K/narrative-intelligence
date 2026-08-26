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

with, from `configs/fusion.yaml` at `version: fusion-v1.0.0`:

| component | weight | what it measures |
|---|---:|---|
| `narrative_severity` | 0.45 | how harmful the content is |
| `coordination` | 0.35 | how inauthentically it spread |
| `authenticity` | 0.20 | whether its evidence is what it claims to be |

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
| `bot_like_ratio` | 0.35 | share of member authors above the bot threshold, over *scorable* authors |
| `co_post_similarity_density` | 0.25 | internal edge density against a complete graph on the same author set |
| `temporal_burstiness` | 0.25 | share of volume in buckets above 3× the narrative's own mean |
| `cohort_concentration` | 0.15 | share of member posts from the single largest cohort |

**authenticity**

| sub-signal | weight | definition |
|---|---:|---|
| `deepfake_hits` | 0.40 | share of checked media flagged as manipulated |
| `domain_risk_agg` | 0.35 | mean risk score of linked domains |
| `anonymous_author_ratio` | 0.25 | share of member authors deleted or without a handle |

`misinfo_likelihood_agg` is engagement-weighted because a false claim nobody saw
is not the same problem as one that reached a hundred thousand people. The weight
is `1 + likes + shares`, so a zero-engagement post still contributes its own
weight rather than vanishing from the average.

`bot_like_ratio` divides by the authors the classifier *could* score, not by all
of them. News and GDELT "authors" are outlets, not accounts; Phase 2 skips them
with a reason code, and counting them as human would deflate the ratio on exactly
the narratives that news carries.

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

If the deepfake module never ran, `authenticity` is `null`, it appears in
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
by `score.fusion` at `fusion-v1.0.0`.

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

Note `authenticity = 0.0`, not `null`. That is a **measured** zero: this
narrative's media was checked and none of it was flagged, and its linked domains
scored low. A null would mean nothing was checked. The response distinguishes
them and so must any reader.

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
* It **inherits every upstream limitation**. The bot classifier's per-fold
  macro-F1 is 0.416 ± 0.187 against a pooled 0.705, so `coordination` is the
  weakest of the three components on an unfamiliar campaign. That caveat is
  surfaced on `/authors/{id}/score` rather than buried here.

The score is a triage aid. The drilldown is the product.
