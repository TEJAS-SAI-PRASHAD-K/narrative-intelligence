# Model card — misinformation-likelihood classifier

**Module:** `modeling/text/misinfo_clf.py` · **Version:** `v0.2.0`
**Output:** `record_scores.misinfo_prob` — a calibrated probability in [0, 1]

---

## Status

**The transformer clears the TF-IDF baseline at the precision 0.85 operating point.**
This card reports the comparison run of 2026-11-09. The full report is
[docs/results/factcheck-comparison.md](../../docs/results/factcheck-comparison.md).

The checkpoint is a `roberta-base` fine-tune. Training saved the last epoch. The
checkpoint sha256 is `be9cc846c5d019881269bce7b1b007e5ffd937bc520a70a8e0b28dcaba5d70e6`.
The weights are on the GPU host only. No private Hugging Face copy exists yet, so a
new clone must train again.

The weights of the older `v0.1.0` run are lost. That run recorded macro F1 0.8062
[0.7956, 0.8161] on the same split. It is not reproducible, and `v0.2.0` replaces it.

Test split at the default threshold of 0.5:

| metric | value | 95% CI |
|---|---|---|
| macro F1 | **0.7967** | [0.7862, 0.8078] |
| ROC-AUC | 0.8978 | [0.8903, 0.9049] |
| PR-AUC | 0.7872 | [0.7701, 0.8026] |
| Brier | 0.1151 | not computed |

Split: `grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)` · n_test 7085 · positive rate 0.2991

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| not-misinformation | 0.868 | 0.900 | 0.884 | 4966 |
| misinformation-like | 0.743 | 0.680 | 0.710 | 2119 |

```bash
python scripts/run_comparison.py all    # needs a GPU, see docs/laya-finetune.md
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

The ship rule compares macro F1 against TF-IDF plus logistic regression on the shared
test split. Each model uses its own threshold, from its own validation rows, at
precision 0.85. The 95% paired-bootstrap interval of the difference must be above 0.

| model | threshold | validation recall | test macro F1 | difference to TF-IDF |
|---|---|---|---|---|
| `misinfo` v0.2.0 | 0.7389 | 0.482 | 0.753 [0.741, 0.764] | **+0.150 [0.135, 0.164]** |
| TF-IDF + logreg | 0.8573 | 0.244 | 0.603 [0.590, 0.616] | baseline |

**Decision: ship.** The lower limit is 0.1354.

Read the size of this difference with care. At the default threshold, TF-IDF has
macro F1 0.764 [0.753, 0.773], and the transformer has 0.797. That gap is 0.033. The
gap is larger at precision 0.85 because TF-IDF reaches that precision only at a
recall of 0.244. The operating point is the decision rule, and the default-threshold
gap is the conservative figure.

### Fact-check set and the unlabelled corpus

The set has 4281 known-false claims, and the frozen manifest is
`configs/factcheck_eval_manifest.csv`. This model scores the 100 English claims and
skips the 4181 other claims with `unsupported_language`. The skipped claims are 4180
Hindi claims and one claim in another language.

| measure, English titles | `misinfo` v0.2.0 | TF-IDF + logreg |
|---|---|---|
| recall on known-false claims | 0.520 [0.420, 0.610] | 0.030 [0.000, 0.070] |
| flag rate on 1350 corpus titles | 0.081 [0.067, 0.096] | 0.004 [0.001, 0.008] |
| lift | 6.382 [4.950, 8.297] | 6.750 [0.000, 31.556] |

A TF-IDF lift has no useful meaning here, because the model flags almost no corpus
title. The fact-check set holds only claims that publishers chose to debunk. Lift is
not accuracy on the corpus, and the corpus has no labels. These 1350 titles come from
one source type in a short collection window.

The operating threshold is a value of the comparison report. The scored tables hold
the calibrated probability, and `configs/models.yaml` does not store this threshold.

The files under `artifacts/eval/misinfo/v0.1.0/` are from the 55-row demo fixture.
They are not a result.

### Corpus sample labels

The 100-record English sample from `modeling.cli sample-for-labelling misinfo --n 100`
(2026-11-11) now has labels. An AI model wrote the labels, not a person. The sheet marks
every row `labeller=llm-unreviewed`. A person did not check the labels.

Every label is 0. Of the 100 rows, 20 are fact-check articles that debunk a claim
(DigitEye, Newschecker, Alt News, BOOM, The Quint). The other 80 are news reports,
features or opinion. No row pushes a false claim. The 95 percent Wilson interval for the
positive rate is [0.000, 0.037].

At its configured `decision_threshold` of 0.5, on the calibrated `record_scores`
values, `misinfo` v0.2.0 flags 42 of the 100 rows (Wilson [0.328, 0.518]). Against these
labels, every flag is a false positive. The model flags all 20 debunks (20 of 20,
[0.839, 1.000]) and 22 of the 80 other articles (0.275, [0.189, 0.381]). The model
reacts to articles about false claims, not only to articles that push one.

The fact-check comparison excludes fact-check publishers from its corpus. The live
scoring stage scores them. In the live corpus, fact-check articles therefore receive
high `misinfo` values, and fusion inherits them.

`modeling.cli evaluate misinfo` refuses transfer metrics on this sheet, because the
labelled rows hold one class. The transfer gap stays unmeasured. A useful next sample
needs more likely positives, for example social posts or posts that fact-checkers
debunked, and a person must label it. The label sheet and the corpus text are not in git.

## Calibration

Isotonic by configuration, with an automatic fallback to Platt scaling below 200
validation rows — isotonic fits a step function to noise on small validation
sets and emits confident 0.0/1.0 outputs. The fallback fired on the demo run (31
validation rows) and is recorded in the report.

Demo run: Platt on 31 rows, Brier 0.0756 → 0.0570 (improved). The real run has a
test Brier of 0.1151.

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

**The transfer gap is currently UNMEASURED.** A first sample is labelled, but the labels
come from an AI model and hold one class (see "Corpus sample labels"). Closing the gap
needs a person:

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
