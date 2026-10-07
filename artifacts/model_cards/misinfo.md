# Model card — misinformation-likelihood classifier

**Module:** `modeling/text/misinfo_clf.py` · **Version:** `v0.1.0`
**Output:** `record_scores.misinfo_prob` — a calibrated probability in [0, 1]

---

## Status

**A real run happened. Its weights are gone, and the artifacts here are still
the demo run.** All three statements are true at once, which is why this section
is longer than it should be.

**The benchmarks are on disk now** — `data/benchmarks/` holds liar,
fakenewsnet and coaid — so the blocker this card was originally written about is
resolved.

**A full-scale fine-tune was run on 2026-08-19.** `models/misinfo/v0.1.0/registry.json`
records it, and it is not a demo run (`is_demo: false`):

| metric | value | 95% CI |
|---|---|---|
| macro F1 | **0.8062** | [0.7956, 0.8161] |
| PR-AUC | 0.7967 | — |
| ROC-AUC | 0.9020 | — |
| Brier | 0.1131 | — |

Split: `grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)` · n_test 7085 · positive rate 0.2991

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| not-misinformation | 0.882 | 0.889 | 0.885 | 4966 |
| misinformation-like | 0.734 | 0.720 | 0.727 | 2119 |

**But the checkpoint's weights are no longer on disk.** `models/misinfo/v0.1.0/`
holds `config.json`, `tokenizer.json`, `calibrator.json`, `training.json` and
`registry.json` — and no `model.safetensors`. So:

- The table above **cannot be reproduced or re-derived** from anything in this
  repository. It is a recorded summary, not verifiable evidence.
- `nlp/availability.py` correctly reports `misinfo` as **degraded**, serving
  precomputed Parquet rather than scoring new text. (It used to report `ready`,
  because the weight-file check counted any `*.json`; that is fixed, and
  `tests/test_calibration_guards.py` pins it down.)
- **The full-scale TF-IDF comparison was never made**, so the headline finding
  below — that the fine-tune does not clear TF-IDF — still rests entirely on the
  55-row demo fixture and is not established either way at scale.

**Everything in the Metrics section below is the DEMO FIXTURE run** (n_test 55,
2026-08-13) and is *not a result*. The eval artifacts under
`artifacts/eval/misinfo/v0.1.0/` are that same demo run.

**To close this properly, retrain.** It is a pretrained-encoder fine-tune, not a
model built from scratch, and `notebooks/colab_finetune.ipynb` runs it on a
Colab T4 in well under an hour (~10 h on 8 GB Apple Silicon over MPS, measured
at ~2 pairs/sec). `configs/models.yaml` lists stronger bases than the
`roberta-base` that produced 0.806 — `microsoft/deberta-v3-base`, and
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` whose FEVER pretraining is fact
verification, the closest task-adjacent signal available for this label.

**The bar is unchanged:** select on validation, and if the transformer does not
clear full-scale TF-IDF + logistic regression, it should not ship — it costs
orders of magnitude more inference for no measured gain.

```bash
python -m modeling.cli train misinfo    # or notebooks/colab_finetune.ipynb
```

## Intended use

A component of a research scorecard for studying how narratives spread across
platforms. It scores how *similar a text is to claims that professional
fact-checkers have rated false*.

## Out-of-scope use

- **It is not a determination that a claim is false.** No output of this system
  is that, and the README says so as a project-level limitation.
- Not for moderation decisions, enforcement, or ranking anyone's account.
- Not for non-English text; those records are skipped, not scored.
- Not as a standalone signal — it is one of six fields in a scorecard.

---

## Training data and the label mapping

| dataset | what it is | access |
|---|---|---|
| LIAR | 12.8k PolitiFact statements by politicians, 6-way labels | open download |
| FakeNewsNet | PolitiFact + GossipCop article **titles** (not bodies) | crawler; titles only |
| CoAID | COVID-era health claims across 4 collection waves | git clone |

**The collapse is a modeling choice with consequences.** LIAR's six-way ordinal
scale becomes binary:

```
pants-fire, false, barely-true  ->  1 (misinformation-like)
half-true                       ->  DROPPED
mostly-true, true               ->  0
```

`half-true` is **dropped, not assigned**. Forcing it either way manufactures
label noise the metrics cannot see: a model that gets every half-true wrong
looks identical to one that gets them right, when they are split down the middle
by an arbitrary rule. Dropping costs roughly a sixth of LIAR's rows and buys a
target that means something.

The mapping lives in `configs/models.yaml`, with every key and value quoted —
YAML 1.1 parses a bare `false:` as a boolean, which silently stopped the mapping
matching LIAR's string labels and dropped every `false` row with no error
anywhere. The loader now also refuses to run if the map does not cover all six
labels.

**FakeNewsNet is titles only.** The repository ships ids and a crawler, not
article bodies, and the tweet half needs Twitter API keys this project does not
have. "Trained on FakeNewsNet" implies far more data than headlines; it is
headlines.

## Split strategy

Group-aware, through `modeling/datasets/splits.py`, which is the only splitter in
the codebase (enforced by a source scan in `tests/test_splits.py`).

| dataset | group key | why |
|---|---|---|
| LIAR | speaker | one politician's statements share phrasing, topic and fact-check history |
| FakeNewsNet | article id, plus outlet | one outlet's house style is memorizable |
| CoAID | normalized claim text | the same claim recurs across collection waves under different row ids |

Group ids are namespaced by dataset (`liar:…`, `coaid:…`) so two benchmarks
using small integer ids cannot collide into one pseudo-group.

Near-duplicates are collapsed **before** splitting. Two rows can carry different
claim ids and be the same sentence — a syndicated wire story republished under
two outlet ids is the canonical case — and that leaks even though the group keys
differ.

## Metrics

Reported with 95% bootstrap confidence intervals. **Accuracy is deliberately
absent**: on an imbalanced target it rewards predicting the majority class.
PR-AUC leads because it degrades exactly when the model starts crying wolf.

Current committed numbers, from `artifacts/eval/misinfo/v0.1.0/`:

| metric | value | 95% CI |
|---|---|---|
| macro F1 | 0.908 | [0.821, 0.978] |
| PR-AUC | 0.988 | [0.959, 1.000] |

Split: grouped by `group_id`, 243 groups, train/val/test = 224/31/55, seed
20260813. **Demo fixture — not a result.**

### Baselines

| baseline | macro F1 | delta | verdict |
|---|---|---|---|
| majority class | 0.368 [0.312, 0.415] | +0.540 | beats |
| TF-IDF + logreg | 0.927 [0.852, 0.982] | −0.019 | **not separable at this test size** |

**The fine-tune does not clear TF-IDF + logistic regression.** On 55 test rows
the intervals overlap heavily, so the honest reading is "no measurable
difference", not "slightly worse". This is reported here rather than tuned away.

On fixture data that outcome is unsurprising — the synthetic text is trivially
separable and a linear model saturates it. On real benchmarks the comparison has
to be re-run, and **if the transformer still fails to clear TF-IDF there, the
transformer should not ship**: it costs orders of magnitude more inference for no
measured gain.

## Calibration

Isotonic by configuration, with an automatic fallback to Platt scaling below 200
validation rows — isotonic fits a step function to noise on small validation
sets and emits confident 0.0/1.0 outputs. The fallback fired on the demo run (31
validation rows) and is recorded in the report.

Demo run: Platt on 31 rows, Brier 0.0756 → 0.0570 (improved). Reliability
diagram in `artifacts/eval/misinfo/v0.1.0/reliability.png`.

**A checkpoint without a `calibrator.json` is refused at load time.**
`misinfo_prob` is contractually a calibrated probability that Phase 4 multiplies
into a fused score; serving raw softmax under that column name would be a silent
lie.

## Known domain shift — read before quoting any number

**Three benchmarks, three different problems.** LIAR is politicians' statements
fact-checked by journalists. FakeNewsNet is news headlines. CoAID is COVID-era
health claims. Training on their union produces a model good at none of them
individually and whose errors are hard to attribute. The union is used because
each alone is too small, and per-benchmark test metrics are reported so the
mixture is visible rather than averaged away.

**The corpus is none of the above.** This project's records are Reddit comments,
Mastodon toots and GDELT article metadata — a different register, a different
length distribution, and a different relationship between text and claim.
**Expect a large drop.**

**Cross-domain, inside one benchmark.** FakeNewsNet's PolitiFact → GossipCop
holdout is run and reported. That number is more honest than in-domain F1
because house style is memorizable and the underlying task is not.

**The transfer gap is currently UNMEASURED.** Closing it needs a person:

```bash
python -m modeling.cli sample-for-labelling misinfo --n 100
# fill in the blank `label` column by hand, then:
python -m modeling.cli evaluate misinfo
```

That single table is worth more than any hyperparameter sweep, and until it
exists the honest statement is that production accuracy is unknown and expected
to be substantially below the benchmark numbers.

## Error analysis

`artifacts/error_analysis/misinfo.md`. The known failure modes, encoded as a
triage taxonomy: satire and parody read as sincere claims (the most common false
positive for this whole model family, because satire and disinformation share
surface form by design), sarcasm, posts quoting a false claim in order to debunk
it, text too short to carry a claim, and opinion with no checkable proposition.

The counts are automated; the analysis is not. The uncategorized examples are
where new categories come from.

## Fairness

Score distributions across language groups and source slices are reported in
notebook 05. English-only coverage means the fairness question is currently
about *register* rather than language: the model was trained on edited,
journalist-adjacent prose and is applied to conversational text.

## Deployment

| | |
|---|---|
| Inference | **precomputed offline into Parquet**, not on-demand |
| Device | CPU-viable at 256 tokens; a full-corpus pass is minutes, not seconds |
| Training | Colab T4; checkpoints saved every epoch |
| Backbone | `roberta-base`; `distilbert-base-uncased` fallback (and the `--demo` default) |
| Checkpoint | resolved by `modeling/registry.py`; never in git |

The demo run used `distilbert-base-uncased`, and the report records which
backbone produced its numbers.
