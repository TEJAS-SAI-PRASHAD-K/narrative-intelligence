# Model card — deepfake detector

**Module:** `modeling/media/deepfake_clf.py` · **Version:** `v0.0.0-untrained`
**Output:** `media_scores.deepfake_prob`, `.manipulation_type`, `.explanation`

---

## Status: not trained

FaceForensics++ requires a signed agreement and the fine-tune requires a GPU;
neither is available in this environment. What ships is the complete pipeline,
the split discipline, the aggregation policy and the honest null path.

The scoring stage runs and writes rows with `deepfake_prob = null` and an
explanation saying why. **`face_detected` and `frames_analyzed` are still
populated**, because "we sampled 16 frames and found no face" is information even
when no score follows.

## The two decisions that make any future number meaningful

### 1. Split by source video, never by frame

Frames from one video in both train and test is **the** mechanism behind
deepfake papers reporting 99% accuracy. Adjacent frames of one clip are
near-identical images; a model that memorizes one face scores perfectly on the
rest of that clip.

`modeling/datasets/faceforensics.py` groups on the **target identity** parsed
from FF++'s `<target>_<source>.mp4` naming, so a manipulated clip and the
original it was derived from land on the same side. Clips whose filename cannot
be parsed are dropped rather than grouped by guess. DFDC fakes with no named
`original` are dropped for the same reason — an untied fake is an unbounded leak
risk.

`tests/test_splits.py` asserts that a frame-level split raises `LeakageError`.

### 2. "No face found" and "real face" are different answers

No detection → `face_detected = false`, `deepfake_prob = null`. Never a low
score. A low score says "we looked and it seems real"; a null says "we could not
look". Conflating them produces a checker that quietly clears every clip it
failed to parse.

## Method

- **Backbone:** Xception (`xception41` via `timm`), ImageNet-pretrained, last
  block + head unfrozen. Never trained from scratch — that would memorize the
  training videos on a T4 budget.
- **Compression:** c23 by default, not raw. Raw is not what a video looks like
  after a platform's transcoder, and training on raw then deploying on
  re-encoded video is a domain shift the model loses to.
- **Frames:** 16, evenly spaced. Not the first 16 — a manipulation affecting only
  part of a clip is invisible to a head-only sample.
- **Face detection:** MTCNN preferred, OpenCV Haar cascade as fallback, explicit
  `none` mode for fixtures. The detector actually used is named in the
  explanation string, because a Haar-derived score deserves less confidence than
  an MTCNN one.
- **Crop margin:** 15%. Blending seams from a face swap sit at the *boundary* of
  the face region, so a tight crop cuts away the most discriminative pixels.
- **Aggregation:** mean of the **top-k** frame scores (k=5). Not the mean — a
  manipulation affecting a third of a clip is averaged into invisibility. Not the
  maximum — one bad crop becomes a confident accusation.

## `manipulation_type`

Emitted only when the training subset actually carried per-method labels. FF++
does (Deepfakes / Face2Face / FaceSwap / NeuralTextures, mapped to the contract's
faceswap / reenactment vocabulary); DFDC does not, so every DFDC-derived score
is `unknown`. **Inventing a method name is worse than admitting we do not know
one.**

## `explanation`

A plain-language string: how many crops across how many frames, which detector,
which frames scored highest, and whether the faces were small enough that the
score should be discounted. The deepfake checker is the most-demoed screen in the
product and a bare number reads as untrustworthy.

## Evaluation plan — what must be reported

1. **Per-manipulation-method metrics.** Aggregate F1 hides that a model is strong
   on FaceSwap and useless on NeuralTextures.
2. **Cross-method generalisation.** Train on three FF++ methods, test on the
   held-out fourth. `domain_holdout()` in the splitter does this.
3. **Cross-dataset generalisation.** FF++ → DFDC.
4. **Compressed / re-encoded inputs**, since that is the production condition.
5. **CPU latency**, measured. Xception over 16 frames is seconds per video —
   acceptable for on-demand upload in Phase 4, not for corpus-wide scoring, which
   is why corpus media scores are precomputed.

**Generalisation to unseen manipulation methods is the known weak point of this
entire model family. Those numbers will be worse than the in-method ones.
Reporting them honestly is a strength; a reviewer who does not see them should
assume the worst.**

## Out-of-scope use

- Not evidence that a video is authentic. A low score is weak evidence at best,
  and a null is no evidence at all.
- Not for images without a detectable face.
- Not for adversarially-crafted media: nothing here is robust to an attacker who
  knows the detector.
