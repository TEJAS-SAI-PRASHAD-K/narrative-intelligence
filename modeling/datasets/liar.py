"""LIAR: 12.8k PolitiFact short statements with 6-way truthfulness labels.

The only benchmark in this project that downloads unattended
(``scripts/download_benchmarks.py --only liar``), and the one whose *label
mapping* deserves the most scrutiny.

**The collapse is a modeling choice with consequences.** LIAR's six labels are
an ordinal scale a human editor assigned; the product needs a probability. The
mapping used here is:

    pants-fire, false, barely-true  -> false (positive class: "misinformation-like")
    half-true                       -> DROPPED
    mostly-true, true               -> true

``half-true`` is dropped rather than assigned. Forcing it either way
manufactures label noise the metrics cannot see: a model that gets every
half-true wrong looks identical to one that gets them right when they are
split down the middle by an arbitrary rule. Dropping costs ~21% of the rows and
buys a target that means something. This is stated again in the model card.

**What LIAR is not.** These are politicians' statements, fact-checked by
journalists, in a register nothing like a Reddit comment. Grouping is by
speaker *and* by statement, and the honest reading of any in-domain F1 here is
"how well the model does on PolitiFact", not "how well it detects
misinformation".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from modeling.config import module_config
from modeling.datasets.base import (
    BenchmarkDataset,
    DatasetInfo,
    drop_empty_text,
    normalize_text,
    register_dataset,
    require_files,
)

#: LIAR ships headerless TSVs. These are the columns, in order, from the paper.
LIAR_COLUMNS = [
    "statement_id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

SIX_WAY = {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}


@register_dataset
class Liar(BenchmarkDataset):
    info = DatasetInfo(
        key="liar",
        label="LIAR (12.8k PolitiFact statements, 6-way labels)",
        access="open",
        url="https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        citation=(
            "Wang, W. Y. (2017). 'Liar, Liar Pants on Fire': A New Benchmark Dataset "
            "for Fake News Detection. ACL 2017."
        ),
        expected_layout=["train.tsv", "valid.tsv", "test.tsv"],
        manual_steps=[
            "python scripts/download_benchmarks.py --only liar",
            "or unzip liar_dataset.zip into data/benchmarks/liar/",
        ],
        notes=(
            "Political statements, not social media text. Expect a large drop when "
            "transferred to a Reddit/Mastodon corpus."
        ),
    )
    #: Group by speaker: the same politician's statements share phrasing, topic
    #: and fact-check history, and splitting inside a speaker leaks all three.
    group_col = "speaker"
    label_col = "label"
    domain_col = "party"

    def validate(self, path: Path) -> None:
        require_files(self, path, "train.tsv", "valid.tsv", "test.tsv")

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        dropped: dict[str, int] = {}
        frames = []
        for split_name in ("train", "valid", "test"):
            frame = pd.read_csv(
                path / f"{split_name}.tsv",
                sep="\t",
                header=None,
                names=LIAR_COLUMNS,
                dtype=str,
                quoting=3,  # LIAR statements contain unbalanced quotes
                on_bad_lines="warn",
            )
            # The official split is discarded on purpose: it is not
            # speaker-grouped, so reusing it would leak speakers across the
            # boundary. We re-split with datasets/splits.py.
            frame["original_split"] = split_name
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True)

        before = len(raw)
        raw = raw.loc[raw["label"].isin(SIX_WAY)]
        if before - len(raw):
            dropped["unknown_label"] = before - len(raw)

        raw["statement"] = normalize_text(raw["statement"])
        raw = drop_empty_text(raw, "statement", dropped, min_chars=10)

        # A missing speaker is not a group of its own; it is an unknown. Give it
        # a stable synthetic id per statement so it cannot silently merge every
        # anonymous statement into one giant pseudo-speaker.
        speaker = raw["speaker"].fillna("").astype(str).str.strip().str.lower()
        synthetic = raw["statement"].map(
            lambda s: "unknown-" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        )
        raw["speaker"] = speaker.where(speaker != "", synthetic)

        mapping = _normalized_map(module_config("misinfo").get("liar_label_map", {}))
        unmapped = SIX_WAY - set(mapping)
        if unmapped:
            raise ValueError(
                f"liar_label_map in configs/models.yaml does not cover {sorted(unmapped)}. "
                "An unmapped label is silently dropped, which loses a slice of the corpus "
                "without any sign that anything went wrong."
            )
        binary = raw["label"].map(lambda v: _map_label(v, mapping))
        dropped["half_true_dropped"] = int((binary == "drop").sum())
        keep = binary != "drop"
        out = raw.loc[keep].copy()
        out["label_6way"] = out["label"]
        # 1 = misinformation-like (the minority-ish positive class), 0 = not.
        out["label"] = (binary.loc[keep] == "false").astype(int)
        out["text"] = out["statement"]
        out["claim_id"] = out["statement_id"].fillna("").astype(str)
        out["party"] = out["party"].fillna("unknown").astype(str)
        out["source_dataset"] = "liar"
        return (
            out[
                [
                    "claim_id",
                    "text",
                    "label",
                    "label_6way",
                    "speaker",
                    "party",
                    "subject",
                    "context",
                    "original_split",
                    "source_dataset",
                ]
            ],
            dropped,
        )


def _yaml_scalar(value: object) -> str:
    """Normalize a YAML scalar to a string.

    YAML 1.1 parses bare ``true`` / ``false`` as booleans on *both* sides of a
    mapping. Left alone, ``false: false`` becomes ``{False: False}`` and stops
    matching LIAR's string labels -- dropping a sixth of the corpus with no
    error anywhere. The config now quotes everything; this is the belt to that
    braces, because the failure is invisible in the metrics.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _normalized_map(mapping: dict) -> dict[str, str]:
    return {_yaml_scalar(k): _yaml_scalar(v) for k, v in mapping.items()}


def _map_label(value: str, mapping: dict[str, str]) -> str:
    """Apply the configured collapse. Unknown labels are dropped, not guessed."""
    return mapping.get(value, "drop")
