"""DFDC: the Deepfake Detection Challenge dataset, behind an agreement.

Used here as a **held-out generalisation set, not a training set**. FF++ is the
training corpus (it carries per-method labels, which DFDC does not); DFDC's job
is to answer "does this transfer to video the model has never seen the
production pipeline of", which is the question that actually matters for a
deepfake checker people upload arbitrary clips to.

Layout, per downloaded part:

    dfdc_train_part_NN/metadata.json
    dfdc_train_part_NN/*.mp4

``metadata.json`` maps filename -> {label: REAL|FAKE, split, original}. The
``original`` field is what makes an honest split possible: every FAKE names the
REAL clip it was derived from, and the two must never straddle the boundary.
Files whose ``original`` is missing are dropped rather than treated as their own
group -- an untied fake is an unbounded leak risk.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from modeling.datasets.base import (
    BenchmarkDataset,
    DatasetInfo,
    DatasetUnavailable,
    register_dataset,
)

log = logging.getLogger(__name__)

#: The real dataset is .mp4; the committed demo fixture is .png for the reasons
#: given in modeling/datasets/faceforensics.py. metadata.json drives the parse
#: either way, so the pairing logic is unaffected.

#: Pre-extracted face crops, named "<video_id>_<frame_index>.png" under fake/
#: and real/. This is how the widely-mirrored Kaggle repackagings ship, and it
#: carries no metadata.json -- see _read_crops for what that costs.
CROP_NAME = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)$")
CROP_DIRS = {"fake": 1, "real": 0}


@register_dataset
class DFDC(BenchmarkDataset):
    info = DatasetInfo(
        key="dfdc",
        label="DFDC (Deepfake Detection Challenge)",
        access="signed_agreement",
        url="https://ai.meta.com/datasets/dfdc/",
        citation=(
            "Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, M., & "
            "Canton Ferrer, C. (2020). The DeepFake Detection Challenge (DFDC) Dataset. "
            "arXiv:2006.07397."
        ),
        expected_layout=[
            "dfdc_train_part_00/metadata.json",
            "dfdc_train_part_00/*.mp4",
            "(any number of part directories)",
        ],
        manual_steps=[
            "accept the dataset agreement at https://ai.meta.com/datasets/dfdc/",
            "download one or more parts into data/benchmarks/dfdc/",
            "one part (~10GB) is enough: this is a generalisation check, not a training set",
        ],
        notes=(
            "No per-method labels, so manipulation_type is 'unknown' for every DFDC clip. "
            "Used as a cross-dataset test set only."
        ),
    )
    group_col = "source_video"
    label_col = "label"
    domain_col = "part"

    def validate(self, path: Path) -> None:
        if not path.exists():
            raise self.unavailable(path)
        if sorted(path.rglob("metadata.json")):
            return
        if _crop_dirs_present(path):
            return
        raise DatasetUnavailable(
            self.info.instructions(path)
            + f"\n  no metadata.json and no fake/ + real/ crop directories under {path}"
        )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        if not sorted(path.rglob("metadata.json")) and _crop_dirs_present(path):
            return _read_crops(path)
        dropped: dict[str, int] = {}
        rows = []
        for metadata_file in sorted(path.rglob("metadata.json")):
            part = metadata_file.parent.name
            try:
                payload = json.loads(metadata_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.warning("unparseable %s: %s", metadata_file, exc)
                dropped["unparseable_metadata"] = dropped.get("unparseable_metadata", 0) + 1
                continue

            for filename, entry in payload.items():
                if not isinstance(entry, dict):
                    continue
                video = metadata_file.parent / filename
                if not video.exists():
                    dropped["metadata_without_video"] = (
                        dropped.get("metadata_without_video", 0) + 1
                    )
                    continue
                label_raw = str(entry.get("label", "")).strip().upper()
                if label_raw not in {"REAL", "FAKE"}:
                    dropped["unknown_label"] = dropped.get("unknown_label", 0) + 1
                    continue
                label = 1 if label_raw == "FAKE" else 0

                if label == 0:
                    source_video = Path(filename).stem
                else:
                    original = entry.get("original")
                    if not original:
                        # An untied fake cannot be grouped with its source, so
                        # including it risks putting the same face on both sides.
                        dropped["fake_without_original"] = (
                            dropped.get("fake_without_original", 0) + 1
                        )
                        continue
                    source_video = Path(str(original)).stem

                rows.append(
                    {
                        "video_id": Path(filename).stem,
                        "source_video": source_video,
                        "path": str(video),
                        "label": label,
                        # DFDC does not name the generation method.
                        "method": "unknown",
                        "part": part,
                    }
                )

        if not rows:
            raise DatasetUnavailable(
                f"DFDC at {path}: metadata.json found but no usable video rows. "
                "Check the .mp4 files were downloaded alongside the metadata."
            )
        frame = pd.DataFrame(rows)
        frame["source_dataset"] = "dfdc"
        return frame, dropped


def _crop_dirs_present(path: Path) -> bool:
    return all(
        (path / name).is_dir() and any((path / name).glob("*.png")) for name in CROP_DIRS
    )


def _read_crops(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse the pre-extracted-crops repackaging: fake/ and real/ PNGs.

    **What this layout can and cannot support.**

    Filenames encode ``<video_id>_<frame_index>``, so grouping by ``video_id``
    prevents the frame-level leakage that is the usual cause of implausible
    deepfake accuracy -- ten crops of one clip cannot straddle a split.

    What it cannot support is *identity* pairing. The original release ships a
    ``metadata.json`` whose ``original`` field names the real clip each fake was
    derived from; this repackaging drops it, and the fake and real video ids do
    not overlap, so there is no way to recover which actor a given fake depicts.
    DFDC swaps faces between actors recorded in the same sessions, so one
    person appears across many clips: trained on this, a model could memorise a
    face and be scored on that same face from the other side of the split.

    **Hence: test-only.** DFDC's role in this project is a cross-dataset
    generalisation check against a model trained on FaceForensics++, and a set
    used purely for testing has no internal boundary to leak across. The frame
    ``paired_source_known`` is False on every row so a future trainer can refuse
    it, and the warning below fires on every load.
    """
    dropped: dict[str, int] = {}
    rows = []
    for directory, label in CROP_DIRS.items():
        for image in sorted((path / directory).glob("*.png")):
            match = CROP_NAME.match(image.stem)
            if match is None:
                dropped["unparseable_crop_name"] = dropped.get("unparseable_crop_name", 0) + 1
                continue
            rows.append(
                {
                    "video_id": match.group("video"),
                    # The clip is its own group. Not the actor -- that is the
                    # part this layout cannot tell us.
                    "source_video": match.group("video"),
                    "frame_index": int(match.group("frame")),
                    "path": str(image),
                    "label": label,
                    "method": "unknown",
                    "part": f"crops_{directory}",
                }
            )

    if not rows:
        raise DatasetUnavailable(
            f"DFDC crops at {path}: fake/ and real/ exist but no parseable "
            "<video_id>_<frame>.png files were found inside them."
        )

    frame = pd.DataFrame(rows)
    frame["source_dataset"] = "dfdc"
    frame["layout"] = "crops"
    # Read by the deepfake trainer, which must refuse a set it cannot group by
    # identity. Never silently True.
    frame["paired_source_known"] = False

    log.warning(
        "DFDC loaded from the pre-extracted-crops layout: %d frames over %d videos "
        "(%d fake, %d real). This packaging has no metadata.json, so a fake cannot be "
        "tied to the real clip it came from and the set is USABLE FOR TESTING ONLY. "
        "Training on it risks identity leakage; see modeling/datasets/dfdc.py.",
        len(frame),
        frame["video_id"].nunique(),
        int((frame["label"] == 1).sum()),
        int((frame["label"] == 0).sum()),
    )
    return frame, dropped
