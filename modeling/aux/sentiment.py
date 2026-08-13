"""Sentiment with ``cardiffnlp/twitter-roberta-base-sentiment-latest``.

Chosen over the usual SST-2 models on domain grounds. SST-2 is movie reviews:
long, well-formed, written to be evaluative. This corpus is Reddit comments,
Mastodon toots and news headlines -- short, elliptical, full of @mentions, URLs
and irony. The Cardiff model is trained on ~124M tweets and fine-tuned on
TweetEval, which is the closest available domain, and it ships a genuine
three-way head (negative / neutral / positive) rather than a forced binary.

Two columns come out of one pass:

* ``sentiment`` -- the argmax label, for filtering and display.
* ``sentiment_score`` -- signed, in [-1, 1], computed as ``p(pos) - p(neg)``.

The signed score is deliberately *not* the argmax probability. A post at
p(pos)=0.45, p(neu)=0.10, p(neg)=0.45 is genuinely ambivalent and should read as
~0.0, not as "positive, confidence 0.45". Phase 4 aggregates this per author and
per narrative, and an aggregation over argmax confidences would be meaningless.

Sarcasm is the known failure mode and it is not fixable here: a sarcastic
"great, another study" reads as positive to every model in this family. It
appears in the error analysis rather than being papered over.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from modeling.aux.base import AuxScorer, label_scores, load_pipeline

log = logging.getLogger(__name__)

#: The model's label vocabulary. Older revisions emit LABEL_0/1/2 in the same
#: order, so both spellings are accepted.
POSITIVE = ("positive", "label_2")
NEUTRAL = ("neutral", "label_1")
NEGATIVE = ("negative", "label_0")


class SentimentScorer(AuxScorer):
    module = "sentiment"
    output_columns = ("sentiment", "sentiment_score")
    min_chars = 3

    def __init__(self, settings=None):
        super().__init__(settings)
        self._pipeline = None
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return self._pipeline is not None
        self._loaded = True
        # top_k=None returns every class, which the signed score needs.
        self._pipeline = load_pipeline(self.model_name, "text-classification", top_k=None)
        return self._pipeline is not None

    def score_texts(self, texts: Sequence[str]) -> list[dict[str, object]]:
        raw = self._pipeline(list(texts), batch_size=self.batch_size, max_length=self.max_length)
        out: list[dict[str, object]] = []
        for prediction in raw:
            scores = label_scores(prediction)
            positive = _first(scores, POSITIVE)
            neutral = _first(scores, NEUTRAL)
            negative = _first(scores, NEGATIVE)
            if positive is None or negative is None:  # pragma: no cover
                log.debug("unexpected sentiment labels %s", sorted(scores))
                out.append({"sentiment": None, "sentiment_score": None})
                continue
            probabilities = {
                "positive": positive,
                "neutral": neutral if neutral is not None else 0.0,
                "negative": negative,
            }
            label = max(probabilities, key=probabilities.__getitem__)
            out.append(
                {
                    "sentiment": label,
                    # Signed and symmetric: ambivalence reads as ~0, not as a
                    # confident label. See the module docstring.
                    "sentiment_score": round(positive - negative, 6),
                }
            )
        return out


def _first(scores: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in scores:
            return scores[name]
    return None
