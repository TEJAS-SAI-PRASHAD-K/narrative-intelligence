"""Which class means "bad"? Resolved from the checkpoint, never assumed.

Both resolvers covered here read a pretrained model's own ``id2label`` to find
out which logit position carries the class this project cares about. Getting it
backwards is the worst error either module could make -- a confident
"authentic" for every manipulated clip, or `support` reported where the text
says `deny` -- and no aggregate metric makes it obvious: an AUC near 0 reads as
a bad model rather than a flipped sign.

So the contract is: resolve it, or refuse. Never guess.
"""

from __future__ import annotations

import pytest

from modeling.media.deepfake_clf import DeepfakeScorer
from modeling.text.stance_clf import StanceClassifier


# --- deepfake: real vs manipulated -----------------------------------------
@pytest.mark.parametrize(
    "id2label,expected",
    [
        ({0: "Real", 1: "Fake"}, 1),
        ({0: "Fake", 1: "Real"}, 0),
        ({0: "Realism", 1: "Deepfake"}, 1),
        ({"0": "authentic", "1": "manipulated"}, 1),
        ({0: "REAL", 1: "FAKE"}, 1),
    ],
)
def test_fake_index_resolves_across_label_vocabularies(id2label, expected):
    """Checkpoints disagree on wording; all of these occur on the Hub."""
    assert DeepfakeScorer()._resolve_fake_index(id2label) == expected


def test_realism_is_not_matched_as_ai_generated():
    """"Realism" contains no real/fake ambiguity but does contain "ai"...

    ...via "Deepfake"'s sibling hint list. Real hints are matched first for
    exactly this reason, so "Realism" must land on the authentic side.
    """
    assert DeepfakeScorer()._resolve_fake_index({0: "Realism", 1: "Deepfake"}) == 1


@pytest.mark.parametrize(
    "id2label",
    [
        {},
        {0: "LABEL_0", 1: "LABEL_1"},            # the unhelpful HF default
        {0: "Real", 1: "Fake", 2: "Fake"},        # ambiguous: two fake classes
        {0: "cat", 1: "dog"},                     # not a forensics head at all
        {0: "Fake"},                              # one-sided
    ],
)
def test_an_unreadable_head_is_refused_rather_than_guessed(id2label):
    assert DeepfakeScorer()._resolve_fake_index(id2label) is None


# --- stance: the three-way NLI head ----------------------------------------
class _Config:
    """Minimal stand-in for a transformers config."""

    def __init__(self, id2label):
        self.id2label = id2label


def test_nli_head_is_resolved_by_name_not_by_position():
    """NLI checkpoints genuinely differ in label order.

    Assuming a position swaps `support` with `deny` on half the checkpoints on
    the Hub, which is silent and catastrophic.
    """
    classifier = StanceClassifier()

    standard = classifier._resolve_nli_head(
        _Config({0: "entailment", 1: "neutral", 2: "contradiction"})
    )
    assert standard == {"entailment": 0, "neutral": 1, "contradiction": 2}

    reversed_order = classifier._resolve_nli_head(
        _Config({0: "contradiction", 1: "neutral", 2: "entailment"})
    )
    assert reversed_order == {"entailment": 2, "neutral": 1, "contradiction": 0}


def test_nli_head_resolution_is_case_insensitive():
    resolved = StanceClassifier()._resolve_nli_head(
        _Config({0: "ENTAILMENT", 1: "Neutral", 2: "CONTRADICTION"})
    )
    assert resolved == {"entailment": 0, "neutral": 1, "contradiction": 2}


@pytest.mark.parametrize(
    "id2label",
    [
        {},
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        {0: "entailment", 1: "not_entailment"},   # two-way NLI: no neutral
        {0: "positive", 1: "negative"},            # a sentiment head
    ],
)
def test_a_non_nli_head_cannot_be_read_as_stance(id2label):
    assert StanceClassifier()._resolve_nli_head(_Config(id2label)) is None


def test_zero_shot_unrelated_comes_from_the_threshold_not_the_head():
    """The NLI head has three classes; the contract has four.

    `unrelated` is a retrieval judgement, not an entailment one, so it is
    assigned by thresholding the directional (entailment + contradiction) mass.
    This pins that behaviour down, because it is the documented approximation
    that keeps the zero-shot path a baseline rather than a model.
    """
    import numpy as np

    classifier = StanceClassifier()
    classifier._nli_index = {"entailment": 0, "neutral": 1, "contradiction": 2}
    cut = float(classifier.config.get("zero_shot", {}).get("unrelated_below", 0.40))

    # Mostly neutral, almost no directional mass -> unrelated.
    low_signal = np.array([[0.05, 0.90, 0.05]])
    assert classifier._zero_shot_predictions(low_signal)[0].label == "unrelated"

    # Strong entailment -> support, and never unrelated however the cut moves.
    assert classifier._zero_shot_predictions(np.array([[0.85, 0.10, 0.05]]))[0].label == "support"
    # Strong contradiction -> deny.
    assert classifier._zero_shot_predictions(np.array([[0.05, 0.10, 0.85]]))[0].label == "deny"
    # Neutral-dominant but with real directional mass -> discuss, not unrelated.
    assert cut < 0.5, "the fixture below assumes the configured cut is below 0.5"
    assert classifier._zero_shot_predictions(np.array([[0.25, 0.50, 0.25]]))[0].label == "discuss"


def test_untrained_stance_returns_reason_coded_nulls():
    """Batch scoring must complete with this module absent, never raise."""
    predictions = StanceClassifier().predict([("a claim", "a post"), ("b", "c")])
    assert [p.label for p in predictions] == [None, None]
    assert {p.reason for p in predictions} == {"model_untrained"}
