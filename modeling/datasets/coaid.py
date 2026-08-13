"""CoAID: COVID-19 healthcare misinformation.

A git repository of CSVs split across per-wave directories (``05-01-2020``,
``07-01-2020``, ``09-01-2020``, ``11-01-2020``). Each wave contains news and
claim files in fake/real pairs.

Two properties worth knowing before reading any CoAID number:

* **The waves overlap.** The same claim reappears across collection dates. Left
  alone, a random split puts wave 1's copy in train and wave 3's in test. The
  loader therefore groups by normalized claim text, not by row id.
* **It is heavily imbalanced toward real.** Report PR-AUC, not ROC-AUC, and
  never accuracy.

CoAID is also the closest of the three text benchmarks to the project's own
corpus in register -- it contains social-media-adjacent claims rather than
politicians' statements -- which makes it the most informative of the three for
transfer, and the smallest.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from modeling.datasets.base import (
    BenchmarkDataset,
    DatasetInfo,
    DatasetUnavailable,
    drop_empty_text,
    normalize_text,
    register_dataset,
)

#: filename stem pattern -> label. CoAID's naming is consistent across waves.
FILE_PATTERNS = [
    (re.compile(r"^News(Fake)COVID-19.*\.csv$", re.I), 1, "news"),
    (re.compile(r"^News(Real)COVID-19.*\.csv$", re.I), 0, "news"),
    (re.compile(r"^Claim(Fake)COVID-19.*\.csv$", re.I), 1, "claim"),
    (re.compile(r"^Claim(Real)COVID-19.*\.csv$", re.I), 0, "claim"),
]

#: Columns that may carry the claim text, in preference order. CoAID is not
#: uniform across waves: some files have `title` + `content`, some only `title`.
TEXT_COLUMNS = ("title", "content", "newstitle", "claim")


@register_dataset
class CoAID(BenchmarkDataset):
    info = DatasetInfo(
        key="coaid",
        label="CoAID (COVID-19 healthcare misinformation)",
        access="open",
        url="https://github.com/cuilimeng/CoAID",
        citation=(
            "Cui, L., & Lee, D. (2020). CoAID: COVID-19 Healthcare Misinformation "
            "Dataset. arXiv:2006.00885."
        ),
        expected_layout=[
            "05-01-2020/NewsFakeCOVID-19.csv",
            "05-01-2020/NewsRealCOVID-19.csv",
            "07-01-2020/ ...",
            "(any wave directory containing News*/Claim* CSVs)",
        ],
        manual_steps=[
            "git clone https://github.com/cuilimeng/CoAID data/benchmarks/coaid",
        ],
        notes="Heavily imbalanced toward real. Waves overlap; group by claim text.",
    )
    group_col = "claim_id"
    label_col = "label"
    domain_col = "record_kind"  # news vs claim

    def validate(self, path: Path) -> None:
        if not path.exists():
            raise self.unavailable(path)
        hits = [
            file
            for file in path.rglob("*.csv")
            if any(pattern.match(file.name) for pattern, _, _ in FILE_PATTERNS)
        ]
        if not hits:
            raise DatasetUnavailable(
                self.info.instructions(path)
                + f"\n  no News*/Claim*COVID-19 CSVs found under {path}"
            )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        dropped: dict[str, int] = {}
        frames = []
        for file in sorted(path.rglob("*.csv")):
            match = next(
                (
                    (label, kind)
                    for pattern, label, kind in FILE_PATTERNS
                    if pattern.match(file.name)
                ),
                None,
            )
            if match is None:
                continue
            label, kind = match
            try:
                frame = pd.read_csv(file, dtype=str, on_bad_lines="warn")
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                dropped["unparseable_file"] = dropped.get("unparseable_file", 0) + 1
                # A wave we cannot parse is a logged skip, not a crash: CoAID's
                # later waves have inconsistent quoting.
                _log_skip(file, exc)
                continue
            columns = {c.lower(): c for c in frame.columns}
            text_col = next((columns[c] for c in TEXT_COLUMNS if c in columns), None)
            if text_col is None:
                dropped["no_text_column"] = dropped.get("no_text_column", 0) + 1
                continue
            frame["text"] = normalize_text(frame[text_col].fillna(""))
            frame["label"] = label
            frame["record_kind"] = kind
            frame["wave"] = file.parent.name
            frames.append(frame[["text", "label", "record_kind", "wave"]])

        if not frames:
            raise DatasetUnavailable(
                f"CoAID at {path} parsed to zero usable rows. Check the clone is complete."
            )
        raw = pd.concat(frames, ignore_index=True)
        raw = drop_empty_text(raw, "text", dropped, min_chars=10)

        # Group by claim *content*, not by row: the same claim recurs across
        # waves under different row ids, and grouping by id would leak it.
        raw["claim_id"] = raw["text"].str.lower().map(
            lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
        )
        before = len(raw)
        raw = raw.drop_duplicates(subset=["claim_id", "label"])
        dropped["cross_wave_duplicate"] = before - len(raw)

        # A claim that appears with both labels across waves is a genuine
        # annotation conflict. Drop it rather than letting a coin flip decide.
        conflicted = raw.groupby("claim_id")["label"].nunique()
        bad = set(conflicted[conflicted > 1].index)
        if bad:
            raw = raw.loc[~raw["claim_id"].isin(bad)]
            dropped["conflicting_labels"] = len(bad)

        raw["source_dataset"] = "coaid"
        return raw[["claim_id", "text", "label", "record_kind", "wave", "source_dataset"]], dropped


def _log_skip(file: Path, exc: Exception) -> None:
    import logging

    logging.getLogger(__name__).warning("skipping unparseable CoAID file %s: %s", file, exc)
