"""Misinformation-likelihood classifier: fine-tuned transformer + calibration.

``roberta-base`` by default, ``distilbert-base-uncased`` when the Colab budget is
tight. Trained on the union of LIAR, FakeNewsNet and CoAID, mapped to a binary
target.

**Read this before quoting any number this model produces.**

*The label collapse is a modeling choice with consequences.* LIAR's six-way
ordinal scale becomes binary; ``half-true`` is dropped rather than assigned. See
``modeling/datasets/liar.py`` for why.

*The three benchmarks are three different problems.* LIAR is politicians'
statements fact-checked by journalists. FakeNewsNet is news headlines. CoAID is
COVID-era health claims. Training on their union produces a model that is good at
none of them individually and whose errors are hard to attribute. The union is
used anyway -- each alone is too small -- and per-dataset test metrics are
reported so the mixture is visible rather than averaged away.

*The corpus is none of the above.* This project's records are Reddit comments,
Mastodon toots and GDELT article metadata. **Expect a large drop.** Benchmark F1
is not production accuracy and must never be presented as such. The transfer gap
is measured directly: 100 real corpus records, hand-labelled, scored by this
model, reported as its own table. That table is worth more than any
hyperparameter sweep.

*Grouping.* By claim/statement id and by speaker within LIAR, by article id and
outlet within FakeNewsNet, by claim text within CoAID. The PolitiFact ->
GossipCop domain holdout is run at least once and reported: that number is more
honest than in-domain F1, and reviewers respect it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config
from modeling.eval.calibrate import Calibrator

log = logging.getLogger(__name__)

LABEL_NAMES = ["not-misinformation", "misinformation-like"]


@dataclass
class TrainedMisinfoModel:
    """A fine-tuned classifier plus everything needed to use it honestly."""

    model_dir: Path
    base_model: str
    calibrator: Calibrator | None
    threshold: float
    label_names: list[str] = field(default_factory=lambda: list(LABEL_NAMES))
    metadata: dict[str, Any] = field(default_factory=dict)


class MisinfoClassifier:
    """Fine-tune and predict. Written as a plain PyTorch loop on purpose.

    ``transformers.Trainer`` would be shorter and would hide the two things most
    worth seeing here: that the validation split drives both early stopping and
    calibration, and that class weighting rather than resampling handles the
    imbalance (resampling would duplicate rows across the group boundary the
    splitter just enforced).
    """

    module = "misinfo"

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.base_model = str(self.config.get("base_model", "roberta-base"))
        self.fallback_model = str(self.config.get("fallback_model", "distilbert-base-uncased"))
        self.max_length = int(self.config.get("max_length", 256))
        self.batch_size = int(self.config.get("batch_size", 16))
        self.learning_rate = float(self.config.get("learning_rate", 2e-5))
        self.epochs = int(self.config.get("epochs", 3))
        self.warmup_ratio = float(self.config.get("warmup_ratio", 0.1))
        self.weight_decay = float(self.config.get("weight_decay", 0.01))
        self.threshold = float(self.config.get("decision_threshold", 0.5))
        self._model = None
        self._tokenizer = None
        self._calibrator: Calibrator | None = None

    # --- training --------------------------------------------------------
    def fine_tune(
        self,
        train_texts: list[str],
        train_labels: np.ndarray,
        val_texts: list[str],
        val_labels: np.ndarray,
        *,
        output_dir: Path,
        base_model: str | None = None,
        epochs: int | None = None,
    ) -> TrainedMisinfoModel:
        """Fine-tune and checkpoint every epoch.

        Checkpointing every epoch is not optional discipline -- a Colab runtime
        dies without warning, and a three-epoch run that saves at the end saves
        nothing.
        """
        import torch
        from torch.utils.data import DataLoader

        model_name = base_model or self.base_model
        device = self.settings.resolve_device()
        n_epochs = epochs if epochs is not None else self.epochs
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer, model = self._load_base(model_name)
        model.to(device)

        train_loader = DataLoader(
            self._encode(tokenizer, train_texts, train_labels),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.settings.seed),
        )

        # Class weighting rather than resampling. Resampling would duplicate
        # rows, and duplicated rows straddle the group boundary the splitter
        # just spent effort enforcing.
        counts = np.bincount(train_labels.astype(int), minlength=2).astype(float)
        weights = torch.tensor(
            (counts.sum() / np.clip(counts * 2, 1, None)), dtype=torch.float32, device=device
        )
        log.info("class counts %s -> loss weights %s", counts.tolist(), weights.tolist())

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

            val_scores = self._raw_scores(model, tokenizer, val_texts, device)
            val_loss = _log_loss(val_labels, val_scores)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(epoch_loss / max(1, len(train_loader)), 4),
                    "val_log_loss": round(val_loss, 4),
                }
            )
            log.info("epoch %d/%d: %s", epoch + 1, n_epochs, history[-1])

            # Save every epoch, before anything can go wrong.
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            if val_loss < best_val:
                best_val = val_loss
                (output_dir / "best_epoch.json").write_text(
                    json.dumps({"epoch": epoch + 1, "val_log_loss": val_loss}), encoding="utf-8"
                )

        self._model, self._tokenizer = model, tokenizer

        # Calibrate on the validation split -- never on training scores, which
        # are overconfident by construction.
        val_scores = self._raw_scores(model, tokenizer, val_texts, device)
        calibrator = Calibrator(str(self.config.get("calibration", "isotonic")))
        calibration = calibrator.fit(val_scores, val_labels)
        log.info(calibration.summary())
        self._calibrator = calibrator
        (output_dir / "calibrator.json").write_text(
            json.dumps(calibrator.state()), encoding="utf-8"
        )

        metadata = {
            "base_model": model_name,
            "epochs": n_epochs,
            "history": history,
            "calibration": calibration.as_dict(),
            "n_train": len(train_texts),
            "n_val": len(val_texts),
        }
        (output_dir / "training.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return TrainedMisinfoModel(
            model_dir=output_dir,
            base_model=model_name,
            calibrator=calibrator,
            threshold=self.threshold,
            metadata=metadata,
        )

    def _load_base(self, model_name: str):
        """Load the configured base model, falling back to the smaller one."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        for candidate in (model_name, self.fallback_model):
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model = AutoModelForSequenceClassification.from_pretrained(
                    candidate, num_labels=2
                )
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

    def _encode(self, tokenizer, texts: list[str], labels: np.ndarray | None = None):
        import torch
        from torch.utils.data import TensorDataset

        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
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

    def _raw_scores(self, model, tokenizer, texts: list[str], device: str) -> np.ndarray:
        """Uncalibrated positive-class probabilities."""
        import torch
        from torch.utils.data import DataLoader

        model.eval()
        loader = DataLoader(self._encode(tokenizer, texts), batch_size=self.batch_size)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                logits = model(
                    input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
                ).logits
                out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    # --- inference -------------------------------------------------------
    def load(self, model_dir: Path) -> bool:
        """Load a trained checkpoint for CPU inference."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self._model.to(self.settings.resolve_device())
        except Exception as exc:
            log.warning("could not load misinfo checkpoint at %s: %s", model_dir, exc)
            return False

        calibrator_path = Path(model_dir) / "calibrator.json"
        if calibrator_path.exists():
            self._calibrator = Calibrator.from_state(
                json.loads(calibrator_path.read_text(encoding="utf-8"))
            )
        else:
            # No calibrator means the raw softmax would be written into a column
            # Phase 4 multiplies. Refuse rather than ship an uncalibrated score
            # under a calibrated column's name.
            log.error(
                "checkpoint at %s has no calibrator.json. misinfo_prob is contractually a "
                "calibrated probability; refusing to serve raw softmax under that name.",
                model_dir,
            )
            return False
        return True

    def predict(self, texts: list[str]) -> np.ndarray:
        """Calibrated probabilities in [0, 1]."""
        if self._model is None:
            raise RuntimeError("no model loaded")
        raw = self._raw_scores(
            self._model, self._tokenizer, texts, self.settings.resolve_device()
        )
        if self._calibrator is None:
            raise RuntimeError("no calibrator; refusing to emit an uncalibrated misinfo_prob")
        return self._calibrator.transform(raw)


def build_training_frame(
    datasets: dict[str, pd.DataFrame], *, drop_conflicts: bool = True
) -> pd.DataFrame:
    """Union the text benchmarks into one frame with a shared group key.

    The group key is namespaced by source dataset (``liar:speaker-x`` vs
    ``coaid:claim-y``). Without the namespace, two datasets that both use small
    integer ids would collide and the splitter would treat unrelated rows as one
    group -- which does not leak, but silently shrinks the effective group count
    and makes the split coarser than it looks.
    """
    frames = []
    for name, frame in datasets.items():
        if not len(frame):
            continue
        work = pd.DataFrame(
            {
                "text": frame["text"].astype(str),
                "label": frame["label"].astype(int),
                "source_dataset": name,
            }
        )
        group_col = {
            "liar": "speaker",
            "fakenewsnet": "claim_id",
            "coaid": "claim_id",
        }.get(name, "claim_id")
        work["group_id"] = name + ":" + frame[group_col].astype(str).to_numpy()
        work["domain"] = (
            frame["domain"].astype(str).to_numpy()
            if "domain" in frame.columns
            else name
        )
        frames.append(work)

    if not frames:
        return pd.DataFrame(columns=["text", "label", "group_id", "domain", "source_dataset"])

    merged = pd.concat(frames, ignore_index=True)
    if drop_conflicts:
        # The same text labelled both ways across datasets is an annotation
        # disagreement between benchmarks, not signal. Drop it rather than let
        # a coin flip decide which benchmark wins.
        normalized = merged["text"].str.strip().str.lower()
        conflicting = normalized.groupby(normalized).transform("size").gt(1) & normalized.map(
            merged.groupby(normalized)["label"].nunique()
        ).gt(1)
        if conflicting.any():
            log.warning(
                "dropping %d rows whose text appears with both labels across benchmarks",
                int(conflicting.sum()),
            )
            merged = merged.loc[~conflicting].reset_index(drop=True)
    return merged


def _log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    scores = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    labels = np.asarray(labels, dtype=float)
    return float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)))
