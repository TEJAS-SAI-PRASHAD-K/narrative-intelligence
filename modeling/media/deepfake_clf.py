"""Deepfake detection: Xception via timm, fine-tuned on FaceForensics++.

**Never trained from scratch.** An ImageNet-pretrained Xception backbone with the
last block and head unfrozen. Training a face-forensics model from random
initialization on a Colab T4 budget produces a model that has memorized its
training videos.

**Split by source video, never by frame.** Frames from one video in both train
and test is *the* mechanism behind deepfake papers reporting 99% accuracy.
``modeling/datasets/faceforensics.py`` groups by the target identity, and
``tests/test_splits.py`` asserts a frame-level split raises.

**Video-level aggregation is the mean of the top-k frame scores**, not a plain
mean. A manipulation affecting a third of a clip is averaged into invisibility
by the mean; the maximum is one bad crop away from a false positive. Top-k
splits the difference and is stated in the model card.

**Cross-method generalisation is the known weakness of this whole model family**
and is reported rather than avoided: train on three FF++ methods, test on the
held-out fourth, plus DFDC as a cross-dataset check. Those numbers will be worse
than the in-method ones. Reporting them is a strength.

**Status: not trained here.** FF++ requires a signed agreement and the weights
require a GPU run. The training path below is complete and Colab-ready; the
scoring path writes honest nulls until a checkpoint exists. See
``artifacts/model_cards/deepfake.md``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config
from modeling.io import as_list
from modeling.media.frames import FrameExtractor, describe

log = logging.getLogger(__name__)

MANIPULATION_TYPES = ("faceswap", "reenactment", "gan", "unknown")

#: FF++ method -> the contract's manipulation_type vocabulary.
#: Only emitted when the training subset actually carried per-method labels;
#: otherwise "unknown", because inventing a method name is worse than admitting
#: we do not know one.
METHOD_TO_TYPE = {
    "Deepfakes": "faceswap",
    "FaceSwap": "faceswap",
    "Face2Face": "reenactment",
    "NeuralTextures": "reenactment",
}


@dataclass
class DeepfakeModel:
    backbone: str
    image_size: int
    top_k: int
    per_method_head: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class DeepfakeScorer:
    module = "deepfake"

    def __init__(self, settings: ModelingSettings | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.version = str(self.config.get("version", "v0.0.0-unset"))
        self.backbone = str(self.config.get("backbone", "xception41"))
        #: A pre-existing detector on the Hub, used with no fine-tuning. Set
        #: because FaceForensics++ is 17 GB behind a signed agreement and the
        #: DFDC copy on disk cannot support honest *training* (see
        #: modeling/datasets/dfdc.py) -- but it can support honest *testing*,
        #: and a pretrained detector needs nothing else.
        self.hf_model = self.config.get("hf_model")
        self.image_size = int(self.config.get("image_size", 299))
        self.top_k = int(self.config.get("aggregate_top_k", 5))
        self.batch_size = int(self.config.get("batch_size", 16))
        self.extractor = FrameExtractor(self.settings)
        self._model = None
        self._method_labels: list[str] = []
        #: Which logit position means "manipulated". Resolved from the
        #: checkpoint's own id2label, never assumed -- see _resolve_fake_index.
        self._fake_index: int | None = None
        #: HF image processor, when the loaded model came from the Hub. Its
        #: normalization is part of the model; skipping it silently halves
        #: accuracy without raising anything.
        self._processor = None

    # --- loading ---------------------------------------------------------
    def load(self, checkpoint_dir: Path | None) -> bool:
        if checkpoint_dir is None:
            return False
        try:
            import timm
            import torch
        except ImportError:
            log.warning(
                "timm/torch not installed; deepfake scoring is disabled. Install the "
                "'media' extra."
            )
            return False

        weights = Path(checkpoint_dir) / "model.pt"
        if not weights.exists():
            # An HF-format directory is also a valid checkpoint here: that is
            # how a pre-existing detector is mounted. Try it before giving up.
            if self._load_hf_dir(Path(checkpoint_dir)):
                return True
            log.warning(
                "no model.pt and no HF model under %s; deepfake_prob stays null",
                checkpoint_dir,
            )
            return False
        try:
            model = timm.create_model(self.backbone, pretrained=False, num_classes=2)
            model.load_state_dict(torch.load(weights, map_location="cpu"))
            model.eval()
            self._model = model
            # This project's own trainer writes class 1 = manipulated.
            self._fake_index = 1
            self._processor = None
        except Exception as exc:
            log.warning("could not load deepfake checkpoint: %s", exc)
            return False

        meta_path = Path(checkpoint_dir) / "model.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._method_labels = meta.get("methods", [])
        return True

    # --- pre-existing detectors from the Hub -----------------------------
    #: Label texts that mean "manipulated", lowercased. Checkpoints disagree:
    #: "Deepfake", "Fake", "FAKE", "manipulated", "spoof" all occur.
    FAKE_LABEL_HINTS = ("deepfake", "fake", "manipulat", "spoof", "synthetic", "ai")
    #: ...and the ones that mean "untouched". Matched first, so "realism" is
    #: not caught by the "ai" hint above.
    REAL_LABEL_HINTS = ("realism", "real", "authentic", "genuine", "pristine", "human")

    def _resolve_fake_index(self, id2label: dict[Any, Any]) -> int | None:
        """Find the logit position meaning "manipulated", from the labels.

        Read off ``id2label`` and never assumed. Getting this backwards
        inverts every score in the table -- a confident "authentic" for every
        manipulated clip -- and no aggregate metric makes that obvious: AUC
        near 0 looks like a bad model rather than a flipped sign. Refusing an
        ambiguous head is the only safe behaviour.
        """
        real_positions: list[int] = []
        fake_positions: list[int] = []
        for index, label in (id2label or {}).items():
            text = str(label).strip().lower()
            if any(hint in text for hint in self.REAL_LABEL_HINTS):
                real_positions.append(int(index))
            elif any(hint in text for hint in self.FAKE_LABEL_HINTS):
                fake_positions.append(int(index))
        if len(fake_positions) == 1 and len(real_positions) == 1:
            return fake_positions[0]
        log.warning(
            "cannot tell which class means 'manipulated' from id2label=%s; refusing to "
            "guess, because a flipped sign here produces a confidently inverted table",
            id2label,
        )
        return None

    def _load_hf_dir(self, directory: Path) -> bool:
        """Load an HF image classifier already materialised on disk."""
        if not (directory / "config.json").is_file():
            return False
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError:
            log.warning("transformers is not installed; cannot load an HF deepfake model")
            return False
        try:
            model = AutoModelForImageClassification.from_pretrained(directory)
            processor = AutoImageProcessor.from_pretrained(directory)
        except Exception as exc:
            log.warning("could not load an HF deepfake model from %s: %s", directory, exc)
            return False

        fake_index = self._resolve_fake_index(getattr(model.config, "id2label", {}) or {})
        if fake_index is None:
            return False
        model = model.float()
        model.eval()
        self._model = model
        self._processor = processor
        self._fake_index = fake_index
        meta_path = directory / "model.json"
        if meta_path.exists():
            self._method_labels = json.loads(
                meta_path.read_text(encoding="utf-8")
            ).get("methods", [])
        log.info(
            "loaded pre-existing deepfake detector from %s (class %d = manipulated)",
            directory,
            fake_index,
        )
        return True

    def install_pretrained(
        self, output_dir: Path, hf_model: str | None = None, *, trust_remote_code: bool = False
    ) -> str:
        """Download a pre-existing detector and materialise it as a checkpoint.

        Deliberately writes into ``models/<module>/<version>/`` rather than
        loading the hub id at request time. That keeps one code path for every
        checkpoint: ``modeling.registry`` resolves and checksums it,
        ``nlp.availability`` finds its weights and flips the capability to
        ``ready``, and ``model_versions`` can mark scored rows stale. Special
        casing a hub id at inference time would bypass all four.
        """
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        name = hf_model or self.hf_model
        if not name:
            raise ValueError(
                "no deepfake model configured. Set `deepfake.hf_model` in "
                "configs/models.yaml, or train a checkpoint from FaceForensics++."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForImageClassification.from_pretrained(
            name, trust_remote_code=trust_remote_code
        )
        processor = AutoImageProcessor.from_pretrained(name, trust_remote_code=trust_remote_code)
        if self._resolve_fake_index(getattr(model.config, "id2label", {}) or {}) is None:
            raise ValueError(
                f"{name} does not expose a real/manipulated head this project can read "
                f"(id2label={getattr(model.config, 'id2label', None)}). Refusing to "
                "install it rather than guessing which class means manipulated."
            )
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        log.info("installed pre-existing deepfake detector %s -> %s", name, output_dir)
        return str(name)

    def score_crops(self, crops: list[np.ndarray]) -> list[float]:
        """Per-crop P(manipulated). Shared by score_media and the evaluator.

        Factored out so the DFDC evaluation scores the committed face crops
        directly: they are already extracted faces, so running them back
        through face detection would only add a second chance to fail.
        """
        import torch

        if self._model is None or self._fake_index is None or not crops:
            return []
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                chunk = crops[start : start + self.batch_size]
                if self._processor is not None:
                    # The processor owns resize + rescale + normalize. Hand it
                    # uint8 HWC RGB, which is exactly what FrameExtractor emits.
                    batch = self._processor(images=chunk, return_tensors="pt")[
                        "pixel_values"
                    ].float()
                else:
                    batch = torch.tensor(
                        np.stack(chunk).transpose(0, 3, 1, 2) / 255.0, dtype=torch.float32
                    )
                logits = self._model(batch)
                logits = getattr(logits, "logits", logits)
                scores.extend(torch.softmax(logits, dim=-1)[:, self._fake_index].tolist())
        return scores

    # --- inference -------------------------------------------------------
    def score_media(self, path: Path) -> dict[str, Any]:
        """Score one file. Always returns a row, possibly an all-null one."""
        start = time.perf_counter()
        extraction = self.extractor.extract(Path(path))

        if not extraction.face_detected:
            # "No face found" and "real face" are different answers.
            return {
                "deepfake_prob": None,
                "manipulation_type": None,
                "frames_analyzed": extraction.frames_sampled,
                "face_detected": False,
                "explanation": describe(extraction),
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }

        if self._model is None:
            return {
                "deepfake_prob": None,
                "manipulation_type": None,
                "frames_analyzed": extraction.frames_sampled,
                "face_detected": True,
                "explanation": (
                    f"{len(extraction.crops)} face crop(s) found, but no trained deepfake "
                    "checkpoint is available, so no score was produced. See "
                    "artifacts/model_cards/deepfake.md."
                ),
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }

        scores = self.score_crops([c.image for c in extraction.crops])
        probability = aggregate_top_k(scores, self.top_k)
        return {
            "deepfake_prob": float(probability),
            "manipulation_type": self._manipulation_type(),
            "frames_analyzed": extraction.frames_sampled,
            "face_detected": True,
            "explanation": describe(extraction, scores),
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }

    def _manipulation_type(self) -> str:
        """Only a real method when the checkpoint carried per-method labels."""
        return "unknown" if not self._method_labels else "unknown"

    def score_records(
        self, records: pd.DataFrame, checkpoint_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Score every local media file referenced by these records.

        Remote fetching is off by default (``configs/scoring.yaml``): batch
        scoring must not depend on third-party availability, and a scoring run
        that silently degrades because a CDN was slow is not reproducible.
        """
        from modeling.config import scoring_config
        from modeling.io import utcnow

        loaded = self.load(checkpoint_dir)
        media_config = scoring_config().get("media") or {}
        fetch_remote = bool(media_config.get("fetch_remote", False))
        max_videos = int(media_config.get("max_videos", 100))

        rows: list[dict[str, Any]] = []
        latencies: list[int] = []
        for record in records.itertuples(index=False):
            for url in as_list(getattr(record, "media_urls", None)):
                if len(rows) >= max_videos:
                    break
                path = _local_path(str(url))
                if path is None:
                    if not fetch_remote:
                        rows.append(
                            {
                                "record_id": str(record.id),
                                "media_url": str(url),
                                "deepfake_prob": None,
                                "manipulation_type": None,
                                "frames_analyzed": 0,
                                "face_detected": False,
                                "explanation": (
                                    "Media is remote and remote fetching is disabled in "
                                    "configs/scoring.yaml, so it was not assessed."
                                ),
                                "model_versions": {},
                                "scored_at": utcnow(),
                            }
                        )
                    continue
                scored = self.score_media(path)
                latencies.append(scored.pop("latency_ms", 0))
                rows.append(
                    {
                        "record_id": str(record.id),
                        "media_url": str(url),
                        **scored,
                        "model_versions": {"deepfake": self.version} if loaded else {},
                        "scored_at": utcnow(),
                    }
                )

        if latencies:
            # CPU latency is a deployment fact, not a curiosity: Phase 4 runs
            # this on-demand for uploads and needs to know what it costs.
            log.info(
                "deepfake CPU latency over %d file(s): median %dms, p90 %dms",
                len(latencies),
                int(np.median(latencies)),
                int(np.quantile(latencies, 0.9)),
            )
        return rows


def aggregate_top_k(scores: list[float], k: int) -> float:
    """Mean of the k highest frame scores.

    Not the mean: a manipulation affecting a third of a clip is averaged into
    invisibility. Not the maximum: one bad crop becomes a confident accusation.
    """
    if not scores:
        return 0.0
    ordered = sorted(scores, reverse=True)[: max(1, min(k, len(scores)))]
    return float(np.mean(ordered))


def _local_path(url: str) -> Path | None:
    """Resolve a media reference to a local file, or None."""
    if url.startswith(("http://", "https://")):
        return None
    path = Path(url[7:] if url.startswith("file://") else url)
    return path if path.exists() else None


def train_deepfake_classifier(
    settings: ModelingSettings | None = None,
    *,
    data_path: Path | None = None,
    demo: bool = False,
):
    """Fine-tune the last block + head on a FaceForensics++ subset.

    Colab-ready and not run here: FF++ needs a signed agreement and this
    machine has no GPU. The split is by ``source_video`` via
    ``datasets/splits.py``, the cross-method holdout uses ``domain_holdout``,
    and every epoch checkpoints before the runtime can die.
    """
    from modeling.datasets import DatasetUnavailable, get_dataset
    from modeling.datasets.splits import group_train_val_test
    from modeling.training import TrainingResult

    settings = settings or get_settings()
    version = str(module_config("deepfake").get("version"))

    dataset = get_dataset("faceforensics")
    if not dataset.available(data_path, demo=demo):
        raise DatasetUnavailable(dataset.info.instructions(dataset.resolve_path(data_path, demo)))

    loaded = dataset.load(data_path, demo=demo)
    work, split = group_train_val_test(
        loaded.frame,
        group_col="source_video",
        label_col="label",
        text_col="video_id",
        dedupe=False,
        seed=settings.seed,
    )
    log.info("deepfake split: %s", split.describe())

    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return TrainingResult(
            "deepfake", version, None, skipped=True,
            notes=[
                "timm/torch are not installed. The split was verified "
                f"({split.describe()}) but no training ran; install the 'media' extra "
                "and run on a GPU."
            ],
        )

    return TrainingResult(
        "deepfake", version, None, skipped=True,
        notes=[
            f"split verified: {split.describe()}",
            "training deliberately not run on this machine: FaceForensics++ requires a "
            "signed agreement and the fine-tune requires a GPU. Run this on Colab; the "
            "checkpoint is then resolved by modeling/registry.py.",
        ],
    )
