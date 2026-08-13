"""Video/image -> sampled frames -> face crops.

The pipeline is: sample N evenly spaced frames, detect faces, crop with margin,
resize to the backbone's input size.

**"No face found" and "real face" are different answers.** When no face is
detected, the contract gets ``face_detected = false`` and ``deepfake_prob =
null`` -- never a low score. A low score says "we looked and it seems real"; a
null says "we could not look". Conflating them is the difference between a
deepfake checker people can trust and one that quietly clears every clip it
failed to parse.

**Detector choice.** MTCNN (via ``facenet-pytorch``) when available, OpenCV's
Haar cascade as a fallback, and an explicit ``none`` mode that treats the whole
frame as the crop. The ``none`` mode exists for the demo fixtures, whose
synthetic faces no real detector recognises -- it is selected explicitly, never
silently, so a demo run cannot be mistaken for a detection run.

Everything here imports lazily. ``modeling/media/`` is an optional extra, and
the rest of Phase 2 must import and test without OpenCV installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from modeling.config import ModelingSettings, get_settings, module_config

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class FaceCrop:
    """One detected face, with the provenance needed for the explanation string."""

    image: np.ndarray  # HxWx3 uint8 RGB
    frame_index: int
    size: int  # the larger side of the detected box, in source pixels
    confidence: float | None = None


@dataclass
class ExtractionResult:
    crops: list[FaceCrop] = field(default_factory=list)
    frames_sampled: int = 0
    detector: str = "none"
    note: str = ""

    @property
    def face_detected(self) -> bool:
        return bool(self.crops)

    @property
    def min_face_size(self) -> int | None:
        return min((c.size for c in self.crops), default=None)


class FrameExtractor:
    module = "deepfake"

    def __init__(self, settings: ModelingSettings | None = None, detector: str | None = None):
        self.settings = settings or get_settings()
        self.config = module_config(self.module)
        self.n_frames = int(self.config.get("frames_per_video", 16))
        self.image_size = int(self.config.get("image_size", 299))
        self.min_face_size = int(self.config.get("face_min_size", 64))
        self.detector_name = detector or str(self.config.get("detector", "auto"))
        self._detector = None
        self._resolved = None

    # --- detector resolution ---------------------------------------------
    def resolve_detector(self) -> str:
        """Pick a detector once, and say which one was picked.

        Reported into ``media_scores.explanation``: a score produced with the
        Haar fallback deserves less confidence than one from MTCNN, and the
        reader should be able to tell which they are looking at.
        """
        if self._resolved is not None:
            return self._resolved
        if self.detector_name != "auto":
            self._resolved = self.detector_name
            return self._resolved

        try:
            from facenet_pytorch import MTCNN  # noqa: F401

            self._resolved = "mtcnn"
        except ImportError:
            try:
                import cv2  # noqa: F401

                self._resolved = "haar"
                log.info(
                    "facenet-pytorch not installed; falling back to the OpenCV Haar cascade. "
                    "Haar misses profile and partially-occluded faces that MTCNN finds, so "
                    "expect more face_detected=false rows."
                )
            except ImportError:
                self._resolved = "none"
                log.warning(
                    "neither facenet-pytorch nor opencv is installed; face detection is "
                    "disabled and the whole frame will be used as the crop. Install the "
                    "'media' extra for real detection."
                )
        return self._resolved

    # --- frame sampling ---------------------------------------------------
    def sample_frames(self, path: Path) -> tuple[list[np.ndarray], str]:
        """N evenly spaced frames from a video, or one frame from an image."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in IMAGE_SUFFIXES:
            image = _read_image(path)
            return ([image], "image") if image is not None else ([], "unreadable_image")

        if suffix not in VIDEO_SUFFIXES:
            return [], f"unsupported_suffix:{suffix or 'none'}"

        try:
            import cv2
        except ImportError:
            return [], "opencv_not_installed"

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return [], "unreadable_video"
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            capture.release()
            return [], "empty_video"

        # Evenly spaced rather than the first N: a manipulation that only
        # affects part of a clip is invisible to a head-only sample.
        indices = np.unique(np.linspace(0, total - 1, min(self.n_frames, total)).astype(int))
        frames: list[np.ndarray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        return frames, "video"

    # --- face detection ---------------------------------------------------
    def extract(self, path: Path) -> ExtractionResult:
        """Frames -> face crops, resized to the backbone's input."""
        frames, kind = self.sample_frames(path)
        detector = self.resolve_detector()
        if not frames:
            return ExtractionResult([], 0, detector, note=kind)

        crops: list[FaceCrop] = []
        for index, frame in enumerate(frames):
            for box, confidence in self._detect(frame, detector):
                crop = _crop_with_margin(frame, box, self.image_size)
                if crop is None:
                    continue
                size = int(max(box[2] - box[0], box[3] - box[1]))
                if size < self.min_face_size and detector != "none":
                    # Below the backbone's effective resolution, compression
                    # artefacts and manipulation artefacts are the same thing.
                    continue
                crops.append(FaceCrop(crop, index, size, confidence))

        return ExtractionResult(
            crops=crops,
            frames_sampled=len(frames),
            detector=detector,
            note=kind if crops else f"{kind}:no_face_detected",
        )

    def _detect(
        self, frame: np.ndarray, detector: str
    ) -> list[tuple[tuple[int, int, int, int], float | None]]:
        height, width = frame.shape[:2]
        if detector == "none":
            return [((0, 0, width, height), None)]

        if detector == "mtcnn":
            if self._detector is None:
                from facenet_pytorch import MTCNN

                self._detector = MTCNN(
                    keep_all=True, device=self.settings.resolve_device(), post_process=False
                )
            boxes, probabilities = self._detector.detect(frame)
            if boxes is None:
                return []
            return [
                ((int(b[0]), int(b[1]), int(b[2]), int(b[3])), float(p) if p is not None else None)
                for b, p in zip(boxes, probabilities, strict=True)
            ]

        # Haar cascade
        import cv2

        if self._detector is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(str(cascade_path))
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        found = self._detector.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)
        return [((int(x), int(y), int(x + w), int(y + h)), None) for x, y, w, h in found]


def _read_image(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.array(image.convert("RGB"))
    except Exception as exc:  # pragma: no cover - unreadable file
        log.debug("could not read image %s: %s", path, exc)
        return None


def _crop_with_margin(
    frame: np.ndarray, box: tuple[int, int, int, int], size: int, margin: float = 0.15
) -> np.ndarray | None:
    """Crop a face box with margin and resize.

    The margin matters: blending seams from a face swap sit at the *boundary* of
    the face region, so a tight crop cuts away the most discriminative pixels.
    """
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    pad_x = int((x1 - x0) * margin)
    pad_y = int((y1 - y0) * margin)
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(width, x1 + pad_x), min(height, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame[y0:y1, x0:x1]
    try:
        from PIL import Image

        return np.array(Image.fromarray(crop).resize((size, size), Image.BILINEAR))
    except ImportError:  # pragma: no cover - pillow is a media-extra dep
        return None


def describe(result: ExtractionResult, frame_scores: list[float] | None = None) -> str:
    """The plain-language ``media_scores.explanation`` string.

    A bare number on the most-demoed screen in the product reads as
    untrustworthy. This says what was looked at and how confident that makes the
    answer.
    """
    if not result.face_detected:
        return (
            f"No face detected in {result.frames_sampled} sampled frame(s) "
            f"(detector: {result.detector}; {result.note}). No deepfake score was produced — "
            "this is 'could not assess', not 'appears authentic'."
        )

    parts = [
        f"Analysed {len(result.crops)} face crop(s) across {result.frames_sampled} sampled "
        f"frame(s) using {result.detector}."
    ]
    smallest = result.min_face_size
    if smallest is not None and smallest < 100:
        parts.append(
            f"The smallest face was {smallest}px, which is near the limit of what this "
            "model resolves — treat the score as low confidence."
        )
    if frame_scores:
        ordered = sorted(range(len(frame_scores)), key=lambda i: -frame_scores[i])[:3]
        top = ", ".join(
            f"frame {result.crops[i].frame_index} ({frame_scores[i]:.2f})" for i in ordered
        )
        parts.append(f"Highest-scoring crops: {top}.")
        spread = max(frame_scores) - min(frame_scores)
        if spread > 0.4:
            parts.append(
                "Scores vary widely across frames, which usually means only part of the "
                "clip is manipulated — or that the detector locked onto different faces."
            )
    if result.detector == "haar":
        parts.append(
            "Detection used the OpenCV Haar cascade, which misses profile and occluded "
            "faces; some frames may have been skipped."
        )
    return " ".join(parts)
