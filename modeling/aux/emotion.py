"""Emotion with ``j-hartmann/emotion-english-distilroberta-base``.

The product's scorecard wants seven buckets: fear, anger, disgust, joy,
surprise, sadness, neutral. This model emits exactly those seven (Ekman's six
plus neutral), so the mapping is one-to-one and **no bucket is synthesized**.

That is worth stating because the alternative was tempting. Several emotion
models ship five or six labels, and the obvious move is to split one label
across two buckets or route "anticipation" into "surprise". Every such move
invents a number the model never produced. The rule here: a bucket the model
does not cover is emitted as ``0.0`` and the gap is documented -- never
back-filled from a neighbouring label.

The output is the full probability distribution over all seven, not just the
argmax, because Phase 4 aggregates emotion per narrative and an argmax-only
signal loses the "fearful *and* angry" case that actually distinguishes
coordinated outrage from ordinary complaint.

Known limitation: this model was trained largely on English text from Twitter,
Reddit and dialogue corpora, and its "neutral" class absorbs anything it cannot
place. A high neutral score means "no strong signal", not "calm".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from modeling.aux.base import AuxScorer, label_scores, load_pipeline
from modeling.io import EMOTIONS

log = logging.getLogger(__name__)


class EmotionScorer(AuxScorer):
    module = "emotion"
    output_columns = ("emotion",)
    min_chars = 5

    def __init__(self, settings=None):
        super().__init__(settings)
        self._pipeline = None
        self._loaded = False
        self.label_map = {
            str(k).lower(): str(v).lower()
            for k, v in (self.config.get("label_map") or {}).items()
        }
        uncovered = set(EMOTIONS) - set(self.label_map.values())
        if uncovered:
            # Not an error: the contract's buckets are the product's, not the
            # model's. But it must be visible, because those buckets will be
            # flat zero for every row and a reader deserves to know why.
            log.warning(
                "emotion buckets %s are not produced by %s and will be 0.0 for every "
                "record; document this gap in the model card rather than mapping a "
                "neighbouring label into them.",
                sorted(uncovered),
                self.model_name,
            )
        self.uncovered = sorted(uncovered)

    def load(self) -> bool:
        if self._loaded:
            return self._pipeline is not None
        self._loaded = True
        self._pipeline = load_pipeline(self.model_name, "text-classification", top_k=None)
        return self._pipeline is not None

    def score_texts(self, texts: Sequence[str]) -> list[dict[str, float]]:
        raw = self._pipeline(list(texts), batch_size=self.batch_size, max_length=self.max_length)
        out: list[dict[str, float]] = []
        for prediction in raw:
            scores = label_scores(prediction)
            bucket = dict.fromkeys(EMOTIONS, 0.0)
            unmapped: list[str] = []
            for label, value in scores.items():
                target = self.label_map.get(label, label if label in bucket else None)
                if target is None:
                    unmapped.append(label)
                    continue
                # Accumulate rather than assign: if two model labels ever map to
                # one bucket, summing is the only defensible combination, and
                # overwriting would silently discard one of them.
                bucket[target] = round(bucket[target] + float(value), 6)
            if unmapped:
                log.debug("emotion labels with no bucket: %s", sorted(set(unmapped)))
            out.append(bucket)
        return out
