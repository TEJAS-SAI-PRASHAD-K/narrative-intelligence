"""FaceForensics++: real videos and four named manipulation methods.

Behind a signed form. Layout, once downloaded with the authors' script:

    original_sequences/youtube/<compression>/videos/*.mp4
    manipulated_sequences/Deepfakes/<compression>/videos/*.mp4
    manipulated_sequences/Face2Face/<compression>/videos/*.mp4
    manipulated_sequences/FaceSwap/<compression>/videos/*.mp4
    manipulated_sequences/NeuralTextures/<compression>/videos/*.mp4

Two decisions are baked in here.

**c23, not raw.** The raw compression level is not what a video looks like after
it has been through a social platform's transcoder. Training on raw and
deploying on re-encoded video is a domain shift the model will lose to, so c23
is the default and c40 is available for a robustness check.

**The group key is the source video.** Manipulated filenames encode their
source pair (``123_456.mp4`` = target 123, source 456), so a manipulated clip and
the original it was made from must land on the same side of the split. This
loader emits ``video_id`` for the clip and ``source_video`` for the identity
that must not straddle -- and ``source_video`` is the group column. Splitting by
frame, or even by clip, is the mechanism behind implausible 99% accuracies;
``tests/test_splits.py`` asserts against it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from modeling.config import module_config
from modeling.datasets.base import (
    BenchmarkDataset,
    DatasetInfo,
    DatasetUnavailable,
    register_dataset,
)

log = logging.getLogger(__name__)

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
COMPRESSIONS = ("c23", "c40", "raw")

#: Manipulated clips are named "<target>_<source>.mp4".
PAIR_RE = re.compile(r"^(\d+)_(\d+)$")

#: Video in the real dataset; stills are accepted too, because the committed
#: demo fixture is PNG. A valid .mp4 needs a codec dependency the test suite
#: must not require, and video bytes do not belong in git. The fixture
#: reproduces the directory layout and the <target>_<source> naming exactly, so
#: the pairing and grouping logic here is genuinely exercised -- only frame
#: extraction is stubbed, and that is covered against real video when the
#: benchmark is present.
MEDIA_GLOBS = ("*.mp4", "*.avi", "*.png", "*.jpg")


def _media_files(root, pattern_dir: str):
    hits = []
    for extension in MEDIA_GLOBS:
        hits.extend(root.rglob(f"{pattern_dir}/{extension}"))
    return sorted(hits)


@register_dataset
class FaceForensics(BenchmarkDataset):
    info = DatasetInfo(
        key="faceforensics",
        label="FaceForensics++ (real + 4 manipulation methods)",
        access="signed_agreement",
        url="https://github.com/ondyari/FaceForensics",
        citation=(
            "Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & "
            "Niessner, M. (2019). FaceForensics++: Learning to Detect Manipulated "
            "Facial Images. ICCV 2019."
        ),
        expected_layout=[
            "original_sequences/youtube/c23/videos/*.mp4",
            "manipulated_sequences/Deepfakes/c23/videos/*.mp4",
            "manipulated_sequences/Face2Face/c23/videos/*.mp4",
            "manipulated_sequences/FaceSwap/c23/videos/*.mp4",
            "manipulated_sequences/NeuralTextures/c23/videos/*.mp4",
        ],
        manual_steps=[
            "sign the form at https://github.com/ondyari/FaceForensics#access",
            "run their download script with -c c23 -d all",
            "point it at data/benchmarks/faceforensics/",
            "a subset (e.g. -n 200) is enough for a Colab T4 budget",
        ],
        notes=(
            "Per-method labels exist, so manipulation_type is real rather than 'unknown'. "
            "Cross-method generalisation is the known weak point and is reported."
        ),
    )
    #: The identity that must not straddle the split. See the module docstring.
    group_col = "source_video"
    label_col = "label"
    #: The manipulation method, for the cross-method holdout table.
    domain_col = "method"

    def __init__(self, compression: str | None = None):
        self.compression = compression or str(module_config("deepfake").get("compression", "c23"))
        if self.compression not in COMPRESSIONS:
            raise ValueError(f"compression must be one of {COMPRESSIONS}, got {self.compression!r}")

    def validate(self, path: Path) -> None:
        if not path.exists():
            raise self.unavailable(path)
        originals = _media_files(path / "original_sequences", f"{self.compression}/videos")
        manipulated = _media_files(path / "manipulated_sequences", f"{self.compression}/videos")
        if not originals or not manipulated:
            raise DatasetUnavailable(
                self.info.instructions(path)
                + f"\n  found {len(originals)} original and {len(manipulated)} manipulated "
                f"{self.compression} clips; both are required"
            )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        dropped: dict[str, int] = {}
        rows = []

        for video in _media_files(path / "original_sequences", f"{self.compression}/videos"):
            rows.append(
                {
                    "video_id": video.stem,
                    # An original is its own source.
                    "source_video": video.stem,
                    "path": str(video),
                    "label": 0,
                    "method": "original",
                    "compression": self.compression,
                }
            )

        for method in METHODS:
            method_dir = path / "manipulated_sequences" / method
            if not method_dir.exists():
                dropped[f"missing_method_{method}"] = 1
                log.warning("FaceForensics++: method %s absent; cross-method table will be partial",
                            method)
                continue
            for video in _media_files(method_dir, f"{self.compression}/videos"):
                match = PAIR_RE.match(video.stem)
                if match is None:
                    dropped["unparseable_pair_name"] = dropped.get("unparseable_pair_name", 0) + 1
                    # Without the pair we cannot tie the fake to its original,
                    # and an untied fake can leak. Skip it rather than guess.
                    continue
                target, source = match.groups()
                rows.append(
                    {
                        "video_id": video.stem,
                        # Group on the *target* identity: the manipulated clip
                        # shares the target's face, background and lighting with
                        # original_sequences/<target>.mp4.
                        "source_video": target,
                        "paired_source": source,
                        "path": str(video),
                        "label": 1,
                        "method": method,
                        "compression": self.compression,
                    }
                )

        if not rows:
            raise DatasetUnavailable(
                f"FaceForensics++ at {path} yielded no {self.compression} clips"
            )
        frame = pd.DataFrame(rows)
        frame["source_dataset"] = "faceforensics"
        return frame, dropped
