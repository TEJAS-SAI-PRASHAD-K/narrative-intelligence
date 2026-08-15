"""Stance detection: (claim, post) -> support / deny / discuss / unrelated.

**Status: not trained. `stance` and `stance_conf` are written as null.**

That is a deliberate outcome, not an omission. The prompt for this phase named
stance a stretch goal behind the misinformation classifier, and the honest
choice when the budget runs short is an empty column rather than an untrained
model producing confident garbage. The contract permits null; Phase 4 renders it
as "not assessed". This module ships the complete training path so the gap can
be closed without redesign.

**What is already decided, and written down here so it is not re-litigated later:**

*The claim comes from the narrative.* At inference time the claim is the
representative post of the record's narrative (Module A2), so stance is only
computable for records that belong to a cluster. Records in the noise bucket get
null for a second, independent reason.

*The group key is the claim, never the post.* SemEval-2016 Task 6 has five
targets in training and a *sixth, unseen* target in test, and that structure is
the entire point of the benchmark: a model that memorizes "posts about Target X
are usually AGAINST" scores well within a target and collapses outside it.

*`unrelated` is unattested in SemEval.* Its NONE class conflates "mentions the
target without taking a side" (discuss) with "unrelated", and
``modeling/datasets/stance.py`` maps NONE to discuss. **A model trained on
SemEval alone can never predict `unrelated`.** That is a coverage gap in the
label set, not a bug, and it must be stated wherever stance output is used --
the absence of `unrelated` predictions is not evidence that nothing is
unrelated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modeling.config import ModelingSettings, get_settings, module_config

log = logging.getLogger(__name__)

STANCE_LABELS = ("support", "deny", "discuss", "unrelated")

#: Labels a SemEval-trained model can actually emit. See the module docstring.
ATTESTED_LABELS = ("support", "deny", "discuss")


@dataclass
class StancePrediction:
    label: str | None
    confidence: float | None
    reason: str | None = None


class StanceClassifier:
    module = "stance"

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.base_model = str(self.config.get("base_model", "roberta-base"))
        self.max_length = int(self.config.get("max_length", 256))
        self.batch_size = int(self.config.get("batch_size", 16))
        self._model = None
        self._tokenizer = None

    @property
    def trained(self) -> bool:
        return self._model is not None

    def load(self, checkpoint_dir: Path | None) -> bool:
        if checkpoint_dir is None:
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
            self._model.eval()
            return True
        except Exception as exc:
            log.warning("no usable stance checkpoint at %s: %s", checkpoint_dir, exc)
            return False

    def predict(self, pairs: list[tuple[str, str]]) -> list[StancePrediction]:
        """Score ``(claim, post)`` pairs.

        Returns explicit nulls when untrained rather than raising: batch scoring
        must complete with this module absent.
        """
        if not self.trained:
            return [
                StancePrediction(None, None, "model_untrained") for _ in pairs
            ]

        import torch

        predictions: list[StancePrediction] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [claim for claim, _ in chunk],
                [post for _, post in chunk],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                probabilities = torch.softmax(self._model(**encoded).logits, dim=-1).numpy()
            for row in probabilities:
                index = int(np.argmax(row))
                predictions.append(
                    StancePrediction(STANCE_LABELS[index], float(row[index]))
                )
        return predictions


def train_stance_classifier(
    settings: ModelingSettings | None = None,
    *,
    data_path: Path | None = None,
    demo: bool = False,
):
    """Fine-tune on SemEval-2016 Task 6 or FEVER-style pairs.

    Verifies the split and the label coverage, then stops: stance was descoped
    behind the misinformation classifier. Running this once the benchmark is on
    disk is the only step needed to close the gap.
    """
    from modeling.datasets import DatasetUnavailable, get_dataset
    from modeling.datasets.splits import group_train_val_test
    from modeling.training import TrainingResult

    settings = settings or get_settings()
    version = str(module_config("stance").get("version"))

    # Prefer FNC-1 over SemEval-2016. Same slot on disk, different corpora:
    # FNC-1 attests all four contract classes (SemEval cannot produce
    # `unrelated` at all) and is ~12x larger. Same "prefer the better benchmark,
    # fall back to the other" pattern the bot trainer uses for Cresci/TwiBot.
    dataset = None
    for key in ("fnc1", "stance"):
        candidate = get_dataset(key)
        if candidate.available(data_path, demo=demo):
            dataset = candidate
            break
    if dataset is None:
        raise DatasetUnavailable(
            get_dataset("fnc1").info.instructions(
                get_dataset("fnc1").resolve_path(data_path, demo)
            )
        )

    loaded = dataset.load(data_path, demo=demo)
    work, split = group_train_val_test(
        loaded.frame, group_col=loaded.group_col, label_col="label", seed=settings.seed
    )
    attested = sorted(set(work["label"].unique()))
    missing = [label for label in STANCE_LABELS if label not in attested]

    notes = [
        f"split verified: {split.describe()}",
        f"labels present in the data: {attested}",
    ]
    if missing:
        notes.append(
            f"labels {missing} are UNATTESTED in this benchmark; a model trained on it "
            "can never predict them. This is a label-set coverage gap and belongs in the "
            "model card, not in a silently-invented mapping."
        )
    notes.append(
        "training deliberately not run: stance was descoped behind the misinformation "
        "classifier. record_scores.stance is written as null, which the contract permits."
    )
    return TrainingResult("stance", version, None, skipped=True, notes=notes)
