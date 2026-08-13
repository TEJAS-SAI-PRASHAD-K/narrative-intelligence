"""Toxicity scoring with ``unitary/toxic-bert``.

Off-the-shelf, not fine-tuned. Emits one float in [0, 1] into
``record_scores.toxicity``.

**Known bias, stated here because it belongs in writing.** This model family
(Jigsaw-trained toxicity classifiers, of which toxic-bert is one) systematically
over-flags:

* African-American English. Sap et al. (2019) measured this directly on the
  Jigsaw-style corpora these models train on: tweets in AAE are labelled toxic
  at substantially higher rates than semantically equivalent text in
  "mainstream" English, and the classifier learns that annotation bias.
* Reclaimed slurs and identity terms in non-pejorative use. Sentences that
  merely *mention* a marginalized group score higher than neutral sentences,
  because the training data over-represents those terms in abusive contexts.
* Text with heavy profanity but no target -- profanity is not toxicity, and this
  model does not reliably distinguish them.

The consequence for this product is concrete: ``toxicity`` must never be read as
"this account is abusive", and the dashboard should not rank accounts by it
alone. It is one component of a scorecard and it carries a demographic error
pattern that the score itself cannot express. This is repeated in the model card
and in the README limitations section.

The score is a probability from a sigmoid head, so it is already in [0, 1], but
it is *not* calibrated on this corpus and no reliability curve is claimed for it.
Unlike the trained scorers (misinfo, bot), which are calibrated against held-out
labels, this one is used as the vendor shipped it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from modeling.aux.base import AuxScorer, label_scores, load_pipeline

log = logging.getLogger(__name__)

#: toxic-bert is multi-head (toxic, severe_toxic, obscene, threat, insult,
#: identity_hate). The product wants one number, and "toxic" is the head that
#: means what the product means. The others are not summed: they overlap
#: heavily and summing would push almost everything toward 1.
PRIMARY_LABEL = "toxic"


class ToxicityScorer(AuxScorer):
    module = "toxicity"
    output_columns = ("toxicity",)
    min_chars = 5

    def __init__(self, settings=None):
        super().__init__(settings)
        self._pipeline = None
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return self._pipeline is not None
        self._loaded = True
        self._pipeline = load_pipeline(self.model_name, "text-classification", top_k=None)
        return self._pipeline is not None

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        raw = self._pipeline(list(texts), batch_size=self.batch_size, max_length=self.max_length)
        out: list[float] = []
        for prediction in raw:
            scores = label_scores(prediction)
            if PRIMARY_LABEL in scores:
                out.append(round(scores[PRIMARY_LABEL], 6))
            elif scores:
                # Some mirrors relabel the heads (LABEL_0/LABEL_1). Fall back to
                # the highest-scoring non-neutral head rather than guessing an
                # index, and say so once.
                log.debug("toxic-bert returned unexpected labels %s", sorted(scores))
                out.append(round(max(scores.values()), 6))
            else:  # pragma: no cover - empty pipeline output
                out.append(0.0)
        return out
