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
        metadata_files = sorted(path.rglob("metadata.json"))
        if not metadata_files:
            raise DatasetUnavailable(
                self.info.instructions(path) + f"\n  no metadata.json found under {path}"
            )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
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
