# Model card — deepfake detector

**Module:** `modeling/media/deepfake_clf.py` · **Version:** `v0.2.0-pretrained`
**Output:** `media_scores.deepfake_prob`, `.manipulation_type`, `.explanation`

---

## Status: evaluated, with a pre-existing detector and no training

**Nothing was fine-tuned.** A pretrained detector —
`dima806/deepfake_vs_real_image_detection`, a ViT image classifier — is mounted
as-is and scored on DFDC. That is a deliberate substitution for the FF++
fine-tune this environment cannot run: FaceForensics++ is 17 GB behind a signed
agreement, and the DFDC copy on disk cannot support honest *training* (see
below). It can support honest *testing*, and a pretrained detector needs nothing
else.

### The measured result

| metric | value | 95% CI |
|---|---|---|
| macro F1 | **0.5640** | [0.4852, 0.6381] |
| ROC-AUC | **0.6652** | [0.5673, 0.7561] |
| PR-AUC | 0.8850 | [0.8259, 0.9321] |
| Brier | 0.2025 | — |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| authentic | 0.297 | 0.475 | 0.365 | 40 |
| manipulated | 0.835 | 0.702 | 0.763 | 151 |

Split: 191 held-out videos at video grain, positive rate
0.7906. It **beats** the majority-class baseline
(0.4415) by +0.1225.

**Read ROC-AUC, not PR-AUC, on this set.** At an 80% positive rate a random
ranker already scores PR-AUC ≈ 0.80, so the 0.885 above is close to
uninformative on its own. ROC-AUC 0.665 with a lower bound of
0.567 is what establishes that there is signal at all. This matters
concretely: a rejected candidate detector
(`prithivMLmods/Deep-Fake-Detector-v2-Model`) scored PR-AUC **0.825** — which
looks respectable — on a ROC-AUC of **0.512 [0.447, 0.584]**, i.e. no signal
whatsoever. It was discarded on that basis.

### `deepfake_prob` is UNCALIBRATED, and the threshold is 0.99

The Platt fit on the calibration half **inverted the ranking** — it mapped
higher raw scores to lower probabilities — and was rejected by the guard in
`modeling/eval/calibrate.py`. Brier had *improved* (0.182 → 0.151) while test
ROC-AUC went 0.665 → 0.335, which is exactly why a Brier check alone is not
sufficient and the guard tests ranking directly.

So this column is a **raw softmax**, not a probability. Two consequences:

1. **The operating point is 0.99**, not 0.5, measured on the held-apart half at
   a 0.85 precision target (precision 0.850, recall 0.773 there).
   `configs/fusion.yaml` has been set to match, and the value belongs to *this
   checkpoint* — re-derive it whenever the mounted detector changes.
2. **`deepfake_prob` must not be multiplied into anything** as if it were a
   probability. It is a ranking signal, to be read the way the README's score
   table says to read `toxicity`.

### How the threshold avoids being fitted on the test set

Nothing is trained, but a threshold and a calibration curve are both *fitted*,
and fitting them on the rows then reported is tuning on test however little else
is learned. So DFDC is split in half by `source_video`: one half fits the
calibrator and picks the operating point, the other half is what the table above
reports. Grouping is by video for the usual reason — frames of one clip on both
sides is *the* cause of implausible deepfake numbers.

### What this number is, and is not

It is **cross-dataset transfer**, which is the hardest regime in this field: a
detector trained by someone else, on someone else's data, applied to DFDC with
no adaptation. A modest number is the expected result. It is also a
**domain mismatch** — this detector targets AI-generated imagery while DFDC is
face swaps — so the FF++-trained alternative named in `configs/models.yaml`
should do better. It is not mounted by default because it ships `custom_code`,
meaning weights *and code* execute from the Hub at load time; that is a
supply-chain decision left to a human.

**An FF++ fine-tune remains the better model** and the plan below still stands.
This is a floor, established without the data.

**DFDC is on disk in a repackaged form**: 3,745 pre-extracted face crops over
381 source videos (305 fake, 76 real), as `fake/` and `real/` directories of
`<video_id>_<frame>.png`. The loader reads this layout, groups by `video_id`,
and marks every row `paired_source_known = False`.

**That layout is usable for testing and not for training.** The original release
ships a `metadata.json` whose `original` field names the real clip each fake was
derived from; this repackaging drops it, and the fake and real video ids do not
overlap, so there is no way to recover which actor a fake depicts. DFDC swaps
faces between actors recorded in the same sessions, so one person appears across
many clips — trained on this, a model could memorise a face and be scored on
that same face from the other side of the split.

Grouping by `video_id` does prevent the frame-level leakage that causes
implausible deepfake accuracy: ten crops of one clip cannot straddle a split.
That is sufficient for DFDC's actual role here — a cross-dataset generalisation
check against a model trained on FF++ — because a set used purely for testing
has no internal boundary to leak across.

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
