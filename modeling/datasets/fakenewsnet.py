"""FakeNewsNet: PolitiFact + GossipCop article labels.

FakeNewsNet ships **ids and a crawler**, not content. The repository's CSVs
carry ``id, news_url, title, tweet_ids``; the article bodies and the tweet half
are collected by the authors' own tooling under their terms, and the tweet half
needs Twitter API keys this project does not have.

**We use the article/label half only** -- specifically the ``title`` column,
which the CSVs do contain. That is stated here, in the model card, and in the
README, because "trained on FakeNewsNet" implies far more data than headlines.

The reason this dataset earns its place despite that limitation is
``domain_col``: PolitiFact (political fact-checks) and GossipCop (celebrity
gossip) are a genuine domain shift inside one benchmark. Training on one and
testing on the other produces a number that means something, and it is a number
reviewers respect precisely because it is lower than the in-domain one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modeling.datasets.base import (
    BenchmarkDataset,
    DatasetInfo,
    DatasetUnavailable,
    drop_empty_text,
    find_dir_containing,
    normalize_text,
    register_dataset,
)

#: filename -> (domain, label). The four CSVs at the root of the repo's dataset/.
FILES = {
    "politifact_fake.csv": ("politifact", 1),
    "politifact_real.csv": ("politifact", 0),
    "gossipcop_fake.csv": ("gossipcop", 1),
    "gossipcop_real.csv": ("gossipcop", 0),
}


@register_dataset
class FakeNewsNet(BenchmarkDataset):
    info = DatasetInfo(
        key="fakenewsnet",
        label="FakeNewsNet (PolitiFact + GossipCop article labels)",
        access="crawler",
        url="https://github.com/KaiDMML/FakeNewsNet",
        citation=(
            "Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2020). FakeNewsNet: "
            "A Data Repository with News Content, Social Context and Spatiotemporal "
            "Information for Studying Fake News on Social Media. Big Data 8(3)."
        ),
        expected_layout=[
            "dataset/politifact_fake.csv",
            "dataset/politifact_real.csv",
            "dataset/gossipcop_fake.csv",
            "dataset/gossipcop_real.csv",
        ],
        manual_steps=[
            "git clone https://github.com/KaiDMML/FakeNewsNet data/benchmarks/fakenewsnet",
            "the four dataset/*.csv files are enough for this project",
            "do NOT run their crawler: the tweet half needs Twitter API keys we do not have, "
            "and this project uses the article/label half only",
        ],
        notes="Titles only. Article bodies require the authors' crawler.",
    )
    #: Group by the article's own id: one story can appear under several urls,
    #: and the id is the closest thing to a story key the CSVs carry.
    group_col = "claim_id"
    label_col = "label"
    #: politifact vs gossipcop -- the built-in domain shift.
    domain_col = "domain"

    def _dataset_dir(self, path: Path) -> Path:
        """Find the directory holding the four CSVs, however it got there.

        The repo nests them under ``dataset/``; ``git clone`` into the benchmark
        folder adds another level (``fakenewsnet/FakeNewsNet/dataset/``); a user
        who copied only the CSVs has them at the root. All three are the same
        dataset and all three should load.
        """
        # Enough to identify the directory: both halves must be present anyway,
        # and validate() reports properly if one is missing.
        found = find_dir_containing(path, "politifact_fake.csv", "gossipcop_fake.csv")
        if found != path:
            return found
        if (path / "dataset").is_dir():
            return path / "dataset"
        return path

    def validate(self, path: Path) -> None:
        if not path.exists():
            raise self.unavailable(path)
        base = self._dataset_dir(path)
        present = [name for name in FILES if (base / name).exists()]
        if not present:
            raise DatasetUnavailable(
                self.info.instructions(path)
                + f"\n  none of {sorted(FILES)} found under {base}"
            )
        # One domain is workable; both is what makes the cross-domain table
        # possible. Warn loudly rather than failing on a partial copy.
        domains = {FILES[name][0] for name in present}
        if len(domains) < 2:
            raise DatasetUnavailable(
                self.info.instructions(path)
                + f"\n  only the {sorted(domains)[0]} half is present. Both halves are "
                "required: the PolitiFact -> GossipCop transfer number is the point of "
                "using this dataset."
            )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        dropped: dict[str, int] = {}
        base = self._dataset_dir(path)
        frames = []
        for name, (domain, label) in FILES.items():
            file = base / name
            if not file.exists():
                dropped[f"missing_{name}"] = 0
                continue
            frame = pd.read_csv(file, dtype=str, on_bad_lines="warn")
            if "title" not in frame.columns:
                raise DatasetUnavailable(
                    f"{file} has no 'title' column (found {list(frame.columns)}). "
                    "This is not the FakeNewsNet dataset CSV."
                )
            frame["domain"] = domain
            frame["label"] = label
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True)

        raw["text"] = normalize_text(raw["title"].fillna(""))
        raw = drop_empty_text(raw, "text", dropped, min_chars=10)

        raw["claim_id"] = raw["id"].fillna("").astype(str)
        missing_id = raw["claim_id"] == ""
        if missing_id.any():
            # Fall back to the url, then to the title, so the group key is never
            # empty -- an empty group key would merge unrelated rows.
            raw.loc[missing_id, "claim_id"] = (
                raw.loc[missing_id, "news_url"].fillna("").astype(str)
            )
            still_missing = raw["claim_id"] == ""
            raw.loc[still_missing, "claim_id"] = raw.loc[still_missing, "text"]

        # The publishing outlet, extracted from the url. Used as a second group
        # key: a single outlet's house style is memorizable.
        raw["outlet"] = (
            raw.get("news_url", pd.Series([""] * len(raw)))
            .fillna("")
            .astype(str)
            .str.replace(r"^https?://", "", regex=True)
            .str.split("/")
            .str[0]
            .str.replace(r"^www\.", "", regex=True)
            .replace("", "unknown")
        )
        raw["source_dataset"] = "fakenewsnet"
        return raw[["claim_id", "text", "label", "domain", "outlet", "source_dataset"]], dropped
