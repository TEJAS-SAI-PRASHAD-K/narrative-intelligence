"""Stance detection: (claim, post) -> support / deny / discuss / unrelated.

Trained by fine-tuning a pre-existing NLI checkpoint on FNC-1. Two decisions
carry most of the weight here, and both are about *starting from a model that
already knows something* rather than from a bare encoder:

**The base model is already an entailment model.**
``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` is tuned on MNLI, FEVER and
ANLI. Stance is premise/hypothesis entailment wearing different labels -- a body
that *supports* a headline entails it, one that *denies* it contradicts it --
so the fine-tune starts from a representation of exactly the relation it has to
learn. ``roberta-base``, the previous setting, starts from nothing.

**The same checkpoint is run zero-shot as a baseline.** That is the only way to
answer the question that matters: did fine-tuning buy anything over simply
asking the NLI model? It runs before the fine-tune, so the bar is fixed before
anyone has a stake in clearing it.

**What is already decided, and written down here so it is not re-litigated later:**

*The claim comes from the narrative.* At inference time the claim is the
representative post of the record's narrative (Module A2), so stance is only
computable for records that belong to a cluster. Records in the noise bucket get
null for a second, independent reason.

*The group key is the article body, never the pair.* FNC-1's 50k train pairs
share only ~1.7k bodies: a pair-level split puts the same body text on both
sides and every metric goes up. ``modeling/datasets/fnc1.py`` sets
``group_col = "body_id"`` and the test set is the organizers' own
``competition_test``, which shares zero bodies with train.

*FNC-1, not SemEval-2016.* SemEval's NONE class conflates "mentions the target
without taking a side" with "unrelated", so a SemEval-trained model can never
predict ``unrelated`` -- a permanent coverage gap in the label set. FNC-1's four
labels map one-to-one onto the contract. The reasoning is in
``modeling/datasets/fnc1.py``.

*``stance_conf`` is not calibrated.* ``modeling/eval/calibrate.py`` is binary
and stance is 4-class, and nothing in ``configs/fusion.yaml`` multiplies
``stance_conf``. It is a softmax maximum, documented as uncalibrated, and should
be read the way the README's score table says to read ``toxicity``. Unlike the
misinfo loader, this one therefore does *not* refuse to load without a
calibrator -- there is nothing for it to refuse over.

*The rare class is the one that matters.* FNC-1 is 73% ``unrelated`` and 1.7%
``deny``. A model that never predicts ``deny`` still scores 98% accuracy, so
accuracy is absent from the report and per-class recall on ``deny`` -- a post
pushing back on a circulating claim -- is the number to read first.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from modeling.config import ModelingSettings, get_settings, module_config

log = logging.getLogger(__name__)

STANCE_LABELS = ("support", "deny", "discuss", "unrelated")

#: Index <-> label, fixed so a checkpoint's head can never be read off by one.
LABEL_TO_INDEX = {label: index for index, label in enumerate(STANCE_LABELS)}


#: Wording that checkpoints use for each contract label. FNC-1's own names are
#: accepted because a head trained straight from its CSV labels carries them.
_LABEL_ALIASES = {
    "support": "support",
    "agree": "support",
    "deny": "deny",
    "disagree": "deny",
    "discuss": "discuss",
    "unrelated": "unrelated",
}


def resolve_stance_labels(id2label: Any) -> list[str] | None:
    """Return the contract label at each logit position, or ``None``.

    Read from the checkpoint's own ``id2label``. ``None`` means the head cannot
    be read with confidence: a missing map, the unhelpful ``LABEL_0`` default,
    an unknown name, a repeated class or a missing class. The caller must then
    refuse the checkpoint. A guessed order swaps ``support`` with ``deny``, and
    no aggregate metric shows it.
    """
    if not id2label:
        return None
    try:
        ordered = sorted((int(index), str(name)) for index, name in dict(id2label).items())
    except (TypeError, ValueError):
        return None
    if [index for index, _ in ordered] != list(range(len(STANCE_LABELS))):
        return None
    labels = [_LABEL_ALIASES.get(name.strip().lower()) for _, name in ordered]
    if None in labels or set(labels) != set(STANCE_LABELS):
        return None
    return [str(label) for label in labels]


@dataclass
class StancePrediction:
    label: str | None
    confidence: float | None
    reason: str | None = None


@dataclass
class TrainedStanceModel:
    """A fine-tuned classifier plus what the model card has to cite."""

    model_dir: Path
    base_model: str
    label_names: list[str] = field(default_factory=lambda: list(STANCE_LABELS))
    metadata: dict[str, Any] = field(default_factory=dict)


class StanceClassifier:
    module = "stance"

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.base_model = str(
            self.config.get("base_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        )
        self.fallback_model = str(self.config.get("fallback_model", "microsoft/deberta-v3-base"))
        self.max_length = int(self.config.get("max_length", 256))
        self.batch_size = int(self.config.get("batch_size", 16))
        self.learning_rate = float(self.config.get("learning_rate", 2e-5))
        self.epochs = int(self.config.get("epochs", 3))
        self.warmup_ratio = float(self.config.get("warmup_ratio", 0.1))
        self.weight_decay = float(self.config.get("weight_decay", 0.01))
        # Trades ~30% compute for a large memory saving by recomputing
        # activations in the backward pass. Measured on an 8 GiB machine:
        # DeBERTa-v3-base at batch 8 / seq 256 peaks at 5.0 GiB with this on
        # and OOMs without it. Off by default only if a box has room to spare.
        self.gradient_checkpointing = bool(self.config.get("gradient_checkpointing", True))
        self._model = None
        self._tokenizer = None
        #: Set when the loaded weights are the raw NLI head rather than a
        #: stance fine-tune. Changes how logits are read, and is reported.
        self._zero_shot = False
        self._nli_index: dict[str, int] | None = None
        #: Contract label at each logit position of a fine-tuned head.
        self._index_to_label: list[str] = list(STANCE_LABELS)
        #: Reason code for the last refused ``load``, or ``None``.
        self.load_error: str | None = None

    @property
    def trained(self) -> bool:
        return self._model is not None

    @property
    def mode(self) -> str:
        """``"fine-tuned"``, ``"zero-shot"`` or ``"untrained"``. Goes in the report."""
        if not self.trained:
            return "untrained"
        return "zero-shot" if self._zero_shot else "fine-tuned"

    # --- base model loading ----------------------------------------------
    def _load_base(self, model_name: str, *, num_labels: int | None = None):
        """Load the configured base model, falling back to the plain encoder.

        Mirrors ``MisinfoClassifier._load_base``. ``num_labels=None`` keeps the
        checkpoint's own head, which is what the zero-shot path needs; passing
        a value replaces it, which is what fine-tuning needs.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        for candidate in (model_name, self.fallback_model):
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                if num_labels is None:
                    model = AutoModelForSequenceClassification.from_pretrained(candidate)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        candidate,
                        num_labels=num_labels,
                        # The NLI head is 3-way and the stance head is 4-way, so
                        # the classifier weights cannot be reused. Say so rather
                        # than letting transformers raise a size mismatch.
                        ignore_mismatched_sizes=True,
                    )
                # Force fp32. Several of these checkpoints declare float16 in
                # their config, and a half-precision encoder trained without a
                # grad scaler produces either a dtype error or silent NaNs.
                # Inference is CPU here anyway, where fp16 buys nothing.
                model = model.float()
                if candidate != model_name:
                    log.warning(
                        "could not load %s; fell back to %s. The model card must record "
                        "which one produced the reported metrics.",
                        model_name,
                        candidate,
                    )
                return tokenizer, model
            except Exception as exc:
                log.warning("loading %s failed: %s", candidate, exc)
        raise RuntimeError(
            f"neither {model_name} nor {self.fallback_model} could be loaded. "
            "Check the network, or pre-download with `modeling warm-cache`."
        )

    # --- training ---------------------------------------------------------
    def fine_tune(
        self,
        train_pairs: list[tuple[str, str]],
        train_labels: np.ndarray,
        val_pairs: list[tuple[str, str]],
        val_labels: np.ndarray,
        *,
        output_dir: Path,
        base_model: str | None = None,
        epochs: int | None = None,
    ) -> TrainedStanceModel:
        """Fine-tune and keep the epoch with the lowest validation loss.

        A plain PyTorch loop for the same reason the misinfo trainer is one: it
        keeps visible that the validation split selects the checkpoint, and that
        the imbalance is handled by class weighting rather than resampling.
        Resampling would duplicate rows, and duplicated rows straddle the group
        boundary the splitter just enforced.

        The checkpoint on disk is rewritten whenever validation loss improves,
        so the saved weights are the best epoch, not the last one, and those
        weights are loaded back into memory before the run returns. The first
        epoch always improves on infinity, so a checkpoint exists after one
        epoch: 50k pairs is hours of work and an interrupted run that saves
        only at the end saves nothing.
        """
        import torch
        from torch.utils.data import DataLoader

        if len(val_pairs) == 0:
            raise ValueError(
                "fine_tune needs a non-empty validation set: the validation loss selects "
                "the checkpoint, and with none no epoch can be chosen or saved."
            )

        model_name = base_model or self.base_model
        device = self.settings.resolve_device()
        n_epochs = epochs if epochs is not None else self.epochs
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer, model = self._load_base(model_name, num_labels=len(STANCE_LABELS))
        # Pin the head's label order into the checkpoint so a reload cannot
        # silently permute the classes.
        model.config.id2label = {i: label for i, label in enumerate(STANCE_LABELS)}
        model.config.label2id = dict(LABEL_TO_INDEX)
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            # use_cache and checkpointing are mutually exclusive; transformers
            # warns and silently disables one of them otherwise.
            model.config.use_cache = False
        model.to(device)

        train_loader = DataLoader(
            self._encode(tokenizer, train_pairs, train_labels),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.settings.seed),
        )

        counts = np.bincount(train_labels.astype(int), minlength=len(STANCE_LABELS)).astype(float)
        weights = torch.tensor(
            counts.sum() / np.clip(counts * len(STANCE_LABELS), 1, None),
            dtype=torch.float32,
            device=device,
        )
        log.info(
            "class counts %s (%s) -> loss weights %s",
            counts.tolist(),
            list(STANCE_LABELS),
            weights.tolist(),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total_steps = max(1, len(train_loader) * n_epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=self.warmup_ratio,
            anneal_strategy="linear",
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        best_val = float("inf")
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        history: list[dict[str, float]] = []
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for input_ids, attention_mask, labels in train_loader:
                optimizer.zero_grad()
                logits = model(
                    input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
                ).logits
                loss = loss_fn(logits, labels.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())

            val_probabilities = self._raw_probabilities(model, tokenizer, val_pairs, device)
            val_loss = _cross_entropy(val_labels, val_probabilities)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(epoch_loss / max(1, len(train_loader)), 4),
                    "val_log_loss": round(val_loss, 4),
                }
            )
            log.info("epoch %d/%d: %s", epoch + 1, n_epochs, history[-1])

            # Save only on improvement. The first finite epoch always improves,
            # so a checkpoint exists before anything can go wrong. A NaN or
            # infinite loss is never an improvement and never the best epoch.
            if not math.isfinite(val_loss):
                log.warning(
                    "epoch %d: validation loss is %s; this epoch cannot be the best",
                    epoch + 1,
                    val_loss,
                )
            elif val_loss < best_val:
                best_val, best_epoch = val_loss, epoch + 1
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                (output_dir / "best_epoch.json").write_text(
                    json.dumps({"epoch": best_epoch, "val_log_loss": val_loss}), encoding="utf-8"
                )
                # Held on the CPU so a large model does not need twice its
                # weights in accelerator memory.
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError(
                f"no epoch produced a finite validation loss in {n_epochs} epochs, so no "
                f"checkpoint was written to {output_dir}. Check the learning rate and the "
                "validation labels; do not register this directory."
            )
        # Later scoring in this run must use the weights that were saved,
        # not whatever the last epoch left behind.
        model.load_state_dict(best_state)
        log.info("restored the best epoch (%d, val_log_loss %.4f)", best_epoch, best_val)

        self._model, self._tokenizer = model, tokenizer
        self._zero_shot = False
        self._index_to_label = list(STANCE_LABELS)

        metadata = {
            "base_model": model_name,
            "epochs": n_epochs,
            "best_epoch": best_epoch,
            "history": history,
            "calibration": "none (4-class; see the module docstring)",
            "n_train": len(train_pairs),
            "n_val": len(val_pairs),
            "label_names": list(STANCE_LABELS),
        }
        (output_dir / "training.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return TrainedStanceModel(
            model_dir=output_dir, base_model=model_name, metadata=metadata
        )

    # --- encoding / raw scoring -------------------------------------------
    def _encode(self, tokenizer, pairs: list[tuple[str, str]], labels: np.ndarray | None = None):
        """Encode (claim, post) as a sentence pair.

        Claim first: it is the premise, the post is the hypothesis, and that is
        the order the NLI pretraining used. Reversing it throws away most of
        what the base checkpoint knows.
        """
        import torch
        from torch.utils.data import TensorDataset

        # `padding=True` pads to the longest sequence in the batch rather than
        # always to max_length. On FNC-1's article bodies that changes little,
        # but at scoring time the corpus is short social posts and padding them
        # all to 256 wastes most of the compute.
        encoded = tokenizer(
            [claim for claim, _ in pairs],
            [post for _, post in pairs],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if labels is None:
            return TensorDataset(encoded["input_ids"], encoded["attention_mask"])
        return TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(np.asarray(labels, dtype=np.int64)),
        )

    def _raw_probabilities(
        self, model, tokenizer, pairs: list[tuple[str, str]], device: str
    ) -> np.ndarray:
        """``(n, n_classes)`` softmax over whatever head the model carries."""
        import torch
        from torch.utils.data import DataLoader

        if not pairs:
            return np.zeros((0, len(STANCE_LABELS)))
        model.eval()
        loader = DataLoader(self._encode(tokenizer, pairs), batch_size=self.batch_size)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                logits = model(
                    input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
                ).logits
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out)

    # --- inference ---------------------------------------------------------
    def load(self, checkpoint_dir: Path | None) -> bool:
        """Load a fine-tuned checkpoint. Returns False when there is none.

        The label order is read from the checkpoint's ``id2label``. A head that
        does not name exactly the four stance classes is refused, with
        ``load_error`` set to ``label_orientation_unresolved``.
        """
        self.load_error = None
        if checkpoint_dir is None:
            self.load_error = "no_checkpoint"
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        except Exception as exc:
            log.warning("no usable stance checkpoint at %s: %s", checkpoint_dir, exc)
            self.load_error = "checkpoint_unreadable"
            return False

        order = resolve_stance_labels(getattr(model.config, "id2label", None))
        if order is None:
            log.error(
                "refusing the stance checkpoint at %s: id2label=%s does not name support, "
                "deny, discuss and unrelated exactly once each, so the meaning of each "
                "logit cannot be resolved",
                checkpoint_dir,
                getattr(model.config, "id2label", None),
            )
            self.load_error = "label_orientation_unresolved"
            return False

        self._tokenizer, self._model = tokenizer, model
        self._index_to_label = order
        self._model.to(self.settings.resolve_device())
        self._model.eval()
        self._zero_shot = False
        return True

    def load_zero_shot(self, model_name: str | None = None) -> bool:
        """Load the NLI base model and read its head as stance.

        This is the **baseline**, not the model. It exists so the fine-tune has
        something to clear, and so `modeling evaluate stance` can answer "was
        the fine-tune worth it" with a number rather than an opinion. The NLI
        head has no ``unrelated`` class; see ``_zero_shot_predictions``.
        """
        name = model_name or self.base_model
        try:
            self._tokenizer, self._model = self._load_base(name, num_labels=None)
        except Exception as exc:
            log.warning("could not load the zero-shot NLI base %s: %s", name, exc)
            return False
        self._model.to(self.settings.resolve_device())
        self._model.eval()
        self._zero_shot = True
        self._nli_index = self._resolve_nli_head(self._model.config)
        if self._nli_index is None:
            log.warning(
                "%s does not expose an entailment/neutral/contradiction head "
                "(id2label=%s); it cannot be read as stance zero-shot",
                name,
                getattr(self._model.config, "id2label", None),
            )
            self._model = None
            return False
        return True

    def _resolve_nli_head(self, config) -> dict[str, int] | None:
        """Map entailment/neutral/contradiction to their logit positions.

        Read off ``id2label`` rather than assumed: the ordering genuinely
        differs between NLI checkpoints, and guessing it silently swaps
        ``support`` with ``deny`` -- the single worst error this module could
        make, and one no aggregate metric would make obvious.
        """
        id2label = getattr(config, "id2label", None) or {}
        wanted = {
            "entailment": str(self.config.get("zero_shot", {}).get(
                "entailment_label", "entailment")).lower(),
            "neutral": str(self.config.get("zero_shot", {}).get(
                "neutral_label", "neutral")).lower(),
            "contradiction": str(self.config.get("zero_shot", {}).get(
                "contradiction_label", "contradiction")).lower(),
        }
        found: dict[str, int] = {}
        for index, label in id2label.items():
            text = str(label).strip().lower()
            for key, target in wanted.items():
                if text == target or text.startswith(target[:4]):
                    found[key] = int(index)
        return found if len(found) == 3 else None

    def predict(self, pairs: list[tuple[str, str]]) -> list[StancePrediction]:
        """Score ``(claim, post)`` pairs.

        Returns explicit nulls when untrained rather than raising: batch scoring
        must complete with this module absent.
        """
        if not self.trained:
            return [StancePrediction(None, None, "model_untrained") for _ in pairs]
        if not pairs:
            return []

        probabilities = self._raw_probabilities(
            self._model, self._tokenizer, pairs, self.settings.resolve_device()
        )
        if self._zero_shot:
            return self._zero_shot_predictions(probabilities)
        return [
            StancePrediction(self._index_to_label[int(np.argmax(row))], float(np.max(row)))
            for row in probabilities
        ]

    def _zero_shot_predictions(self, probabilities: np.ndarray) -> list[StancePrediction]:
        """Read a 3-way NLI head as 4-way stance.

        entailment -> support, contradiction -> deny, neutral -> discuss. The
        fourth class is the problem: ``unrelated`` is a *retrieval* judgement
        ("this body is about something else"), not an entailment one, and the
        NLI head has no way to express it. A body that is off-topic produces
        high neutral, exactly like one that discusses the claim without taking a
        side.

        So ``unrelated`` is assigned by thresholding: when neither entailment
        nor contradiction carries meaningful mass, the pair is called unrelated
        rather than discuss. That is an approximation with a knob in it
        (``zero_shot.unrelated_below``), which is precisely why this path is
        reported as a baseline and never shipped as the model.
        """
        assert self._nli_index is not None
        entail = self._nli_index["entailment"]
        neutral = self._nli_index["neutral"]
        contra = self._nli_index["contradiction"]
        cut = float(self.config.get("zero_shot", {}).get("unrelated_below", 0.40))

        predictions: list[StancePrediction] = []
        for row in probabilities:
            directional = float(row[entail] + row[contra])
            if directional < cut and row[neutral] >= row[entail] and row[neutral] >= row[contra]:
                predictions.append(
                    StancePrediction("unrelated", float(row[neutral]), "nli_neutral_below_cut")
                )
                continue
            choice = max(
                (("support", entail), ("deny", contra), ("discuss", neutral)),
                key=lambda item: row[item[1]],
            )
            predictions.append(StancePrediction(choice[0], float(row[choice[1]])))
        return predictions


def _cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Multiclass log loss. Local for the same reason misinfo's is local:
    importing sklearn.metrics here would pull it into the inference path."""
    if len(labels) == 0:
        return float("inf")
    indices = np.asarray(labels, dtype=int)
    picked = probabilities[np.arange(len(indices)), indices]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))
