# Model card — stance classifier

**Module:** `modeling/text/stance_clf.py` · **Version:** `v0.3.0`
**Output:** `record_scores.stance`, `record_scores.stance_conf`. The columns are
**null where no checkpoint is registered**.

---

## Result

The fine-tune ran on 2026-11-10 on a GPU host, with 3 epochs. The trainer restored
epoch 2, which has the lowest validation log loss (0.1376).

The test set is the FNC-1 competition test set, with 25,413 pairs. Its bodies do not
occur in training. Training and validation split by headline, and the trainer removed
156 training rows that share a headline with the test set.

| model | macro F1 | 95% CI |
|---|---|---|
| **fine-tune v0.3.0** | **0.7965** | [0.7862, 0.8066] |
| zero-shot NLI, same base model | 0.2496 | [0.2430, 0.2557] |
| TF-IDF + logreg | 0.2612 | [0.2537, 0.2688] |
| majority class | 0.2096 | [0.2088, 0.2105] |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| support | 0.739 | 0.736 | 0.737 | 1903 |
| deny | 0.649 | 0.541 | 0.590 | 697 |
| discuss | 0.861 | 0.871 | 0.866 | 4464 |
| unrelated | 0.991 | 0.995 | 0.993 | 18349 |

The fine-tune clears every baseline. The `deny` class is the weakest, and 46 percent
of `deny` pairs are missed. The FNC-1 competition score is not computed.

The benchmark is English news headlines and bodies. This result does not measure
stance on this corpus. The weights are on the GPU host only, so a new clone must
train again.

## Status: the training path and the checkpoint exist; the corpus has no stance labels

`stance` and `stance_conf` are written as **null** where no checkpoint is registered,
which the contract permits and Phase 4 renders as "not assessed". An untrained model
producing confident stance labels would be worse than an empty column.

**Scoring stage (v0.3.0).** With a registered checkpoint, `_stance_stage` scores
each record against the representative claim of its narrative and writes `stance`
and `stance_conf`. A record that is not scored gets a `stance:<reason>` entry in
`skip_reasons`: `empty_text`, `text_too_short`, `unsupported_language` (stance is
English only), `no_narrative`, `no_claim_text`, `claim_unsupported_language` and
`is_claim`. A checkpoint whose `id2label` does not name the four classes exactly
once is refused (`label_orientation_unresolved`) and the columns stay null. A real
checkpoint scored 83 of 636 news records on 2026-11-10: 60 `support`, 21 `discuss` and
2 `deny`. No labels exist for these rows, so the accuracy on this corpus is not known.

**What changed.** The trainer is no longer a stub. `modeling/training.py`
registers a real `stance` path — load → group-split → baselines → fine-tune →
report → error analysis → registry — and it has been executed end to end on the
committed fixtures: 50 train / 10 val / 24 test pairs, 1 epoch. Those numbers
are stamped `DEMO FIXTURE -- NOT A RESULT` and are plumbing evidence only. The
fixture checkpoint is deliberately **refused** by `nlp/availability.py`, which
reads `is_demo` out of `registry.json`, so it cannot be served by accident.

**It is a pretrained model fine-tuned, not a model built.** The base is
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, chosen over the previous
`roberta-base` for a specific reason: stance *is* premise/hypothesis entailment
wearing different labels — a body that supports a headline entails it, one that
denies it contradicts it — so an MNLI+FEVER checkpoint starts from a
representation of exactly the relation it has to learn. A bare encoder starts
from nothing.

### The baseline is the same checkpoint, run zero-shot

This is the number that matters, and it is free. `StanceClassifier.load_zero_shot`
reads the base model's own three-way NLI head as stance — entailment → support,
contradiction → deny, neutral → discuss — and scores the test set before any
fine-tuning happens. It answers the only question worth asking about this
module: *did fine-tuning buy anything over simply asking the NLI model?*

The head's logit positions are resolved from its `id2label`, never assumed. NLI
checkpoints genuinely differ in label order, and assuming a position would swap
`support` with `deny` on half the checkpoints on the Hub — silent, catastrophic,
and invisible in any aggregate metric. An unreadable head is refused rather than
guessed (`tests/test_label_orientation.py`).

**`unrelated` is the one class the zero-shot path cannot really express.** It is
a *retrieval* judgement — "this body is about something else" — not an
entailment one, and an off-topic body produces high neutral exactly like one
that discusses the claim without taking a side. So it is assigned by
thresholding the directional mass (`zero_shot.unrelated_below`). That is an
approximation with a knob in it, and it is precisely why this path is reported
as a baseline and never shipped as the model.

### `stance_conf` is NOT calibrated

`modeling/eval/calibrate.py` fits a binary isotonic or Platt curve; stance is
4-class. Rather than bolt a calibrator onto the wrong arity, `stance_conf` ships
as a raw softmax maximum, documented as uncalibrated. Nothing in
`configs/fusion.yaml` multiplies it — the fusion components are
narrative_severity, coordination and authenticity — so this costs nothing
downstream. Read it as a ranking signal, the way the README's score table says
to read `toxicity`, and never as a probability. Note the consequence: unlike the
misinfo loader, the stance loader does **not** refuse to load without a
calibrator, because there is nothing for it to refuse over.

### Running it

`notebooks/colab_finetune.ipynb`. Measured on 8 GB Apple Silicon over MPS,
DeBERTa-v3-base manages ~2 pairs/sec with gradient checkpointing — ~20 h for
FNC-1's 50k pairs across 3 epochs, versus well under an hour on a Colab T4. The
laptop config (batch 16, gradient checkpointing on) exists because without
checkpointing the run OOMs at 8 GB; the notebook raises both for the T4.

**What changed: the corpus is on disk and the label-coverage gap is closed.**
`data/benchmarks/stance/` holds **FNC-1** (Fake News Challenge), not SemEval-2016
as originally planned, and FNC-1 is the better corpus for this project:

| | SemEval-2016 Task 6 | **FNC-1** |
|---|---|---|
| classes | FAVOR / AGAINST / NONE | agree / disagree / discuss / **unrelated** |
| covers the contract? | **no** — cannot express `unrelated` | **yes**, one-to-one |
| size | ~4k pairs | **75,385 pairs / 2,587 bodies** |
| pairing | target phrase vs tweet | headline vs article body |

`modeling/datasets/fnc1.py` loads it and `modeling.training._train_stance`
prefers it over SemEval automatically.

**The imbalance is the headline caveat.** FNC-1 train is ~73% `unrelated`,
~18% `discuss`, ~7% `agree` and **~1.7% `disagree`**. A model that never
predicts `disagree` still scores ~98% accuracy, which is why accuracy is absent
from the report and the loss is class-weighted. `deny` is also the class the
product cares most about — a post pushing back on a circulating claim is the
pushback signal — so **read its per-class recall first**.

## What is already decided

These are written down so they are not re-litigated when someone picks this up.

**The claim comes from the narrative.** At inference time the claim is the
representative post of the record's narrative (Module A2), so stance is only
computable for records that belong to a cluster. Records in the noise bucket get
null for a second, independent reason.

**The group key is the headline (the claim).** FNC-1's ~50k train pairs carry only
~1.6k distinct headlines, each against many bodies. A body-grouped split put all
1466 validation headlines in train as well, measured on the pinned download, so
train and val are split by `claim_id`, a hash of the lowercased headline. The test
set is not resampled: it is FNC-1's own `competition_test`. It shares no bodies
with train, but it shares 10 headlines (156 train rows), and those rows are
purged from train with `domain_holdout(purge_col="claim_id")`. Measured split on
the pinned download: 43,909 train, 5,907 val, 25,413 test pairs.

**Known residual leak.** The headline/body graph is one connected component, so
headlines and bodies cannot both be kept disjoint between train and val. Headlines
are disjoint. Some article bodies occur on both sides, against different
headlines. Test is disjoint from train and val in both.

This path deliberately does **not** dedupe before splitting, unlike misinfo. In
FNC-1 one body legitimately appears against many different headlines;
near-duplicate collapse on `text` would delete most of the corpus.

**`unrelated` is unattested in SemEval.** Its NONE class conflates "mentions the
target without taking a side" (discuss) with "unrelated". The loader maps NONE →
discuss and documents it. **A model trained on SemEval alone can never predict
`unrelated`.**

That is a coverage gap in the label set, not a bug, and it must be stated
wherever stance output is used: **the absence of `unrelated` predictions is not
evidence that nothing is unrelated.** The alternative — splitting NONE across two
buckets by a heuristic — would invent labels the annotators never assigned.

## Intended use, once trained

Indicating whether a post supports, denies or merely discusses the claim its
narrative is built around. Useful for separating amplification from pushback
inside one narrative, which is otherwise invisible in a volume chart.

## Out-of-scope use

- Not a truth judgement. Stance is about the *post's relationship to a claim*,
  not the claim's accuracy.
- Not for non-English text.
- Not for records outside a narrative cluster: without a claim, there is nothing
  to take a stance toward.

## Known failure mode, in advance

Sarcasm inverts the intended stance while leaving the surface wording intact.
This is the dominant error class for every model in this family and it is
already in the error-analysis taxonomy (`sarcasm_or_irony`) ready for when there
are errors to analyse.
