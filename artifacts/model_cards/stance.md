# Model card — stance classifier

**Module:** `modeling/text/stance_clf.py` · **Version:** `v0.2.0`
**Output:** `record_scores.stance`, `record_scores.stance_conf` — **null until a
checkpoint is mounted**

---

## Status: the training path is complete and verified; the real fine-tune is GPU work

`stance` and `stance_conf` are still written as **null**, which the contract
permits and Phase 4 renders as "not assessed". An untrained model producing
confident stance labels would be worse than an empty column.

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

**The group key is the article body, never the pair.** FNC-1's ~50k train pairs
share only ~1.7k bodies, so a pair-level split puts the same body text on both
sides and every metric goes up. The test set is not resampled at all: it is
FNC-1's own `competition_test`, whose bodies are disjoint from train by
construction, reproduced with `domain_holdout(domain_col="official_split")`.
Train/val are then group-split by `body_id`.

Note that this path deliberately does **not** dedupe before splitting, unlike
misinfo. In FNC-1 one body legitimately appears against many different
headlines; near-duplicate collapse on `text` would delete most of the corpus.

(The same reasoning applied to SemEval's five-train/one-unseen-target structure,
which is why the group key was originally described as the claim/target. FNC-1
is what is on disk, and the body is its equivalent.)

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
