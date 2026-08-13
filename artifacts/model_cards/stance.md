# Model card — stance classifier

**Module:** `modeling/text/stance_clf.py` · **Version:** `v0.0.0-untrained`
**Output:** `record_scores.stance`, `record_scores.stance_conf` — **both null**

---

## Status: deliberately not trained

Stance was descoped behind the misinformation classifier, as the phase plan
allowed. `stance` and `stance_conf` are written as **null**, which the output
contract explicitly permits and which Phase 4 renders as "not assessed".

This is a legitimate outcome. An untrained model producing confident stance
labels would be worse than an empty column — it would populate a UI field with
noise that looks like signal. The complete training path ships in this module;
closing the gap needs the benchmark on disk and one command.

```bash
# after the SemEval-2016 Task 6 data arrives:
python -m modeling.cli train stance
```

## What is already decided

These are written down so they are not re-litigated when someone picks this up.

**The claim comes from the narrative.** At inference time the claim is the
representative post of the record's narrative (Module A2), so stance is only
computable for records that belong to a cluster. Records in the noise bucket get
null for a second, independent reason.

**The group key is the claim/target, never the post.** SemEval-2016 Task 6 has
five targets in training and a *sixth, unseen* target in test, and that structure
is the entire point of the benchmark: a model that memorizes "posts about Target
X are usually AGAINST" scores well within a target and collapses outside it.

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
