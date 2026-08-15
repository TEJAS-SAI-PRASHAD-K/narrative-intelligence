"""FNC-1: the Fake News Challenge stance corpus.

**Why this and not SemEval-2016 Task 6.** The product contract asks for four
stance classes -- support / deny / discuss / unrelated. SemEval has three
(FAVOR / AGAINST / NONE), and its NONE conflates "mentions the target without
taking a side" with "unrelated", so a SemEval-trained model *can never predict
`unrelated`*. That was a permanent coverage gap in the stance model card.

FNC-1's label set maps one-to-one onto the contract:

===============  ==========  ============================================
FNC-1            contract    what it means
===============  ==========  ============================================
``agree``        support     the body supports the headline's claim
``disagree``     deny        the body refutes it
``discuss``      discuss     the body discusses it without taking a side
``unrelated``    unrelated   the body is about something else
===============  ==========  ============================================

No bucket is synthesized and no label is dropped. It is also ~12x larger than
SemEval (50k train pairs vs ~4k).

**The shape is different, and that matters.** SemEval pairs a *target phrase*
with a *tweet*. FNC-1 pairs a *headline* with a *full article body*. At
inference time this project pairs a narrative's representative claim with a
corpus record, which sits between the two: claim-like on one side, short social
text on the other. FNC-1 is the closer match on label set and the worse match on
text length, and that tradeoff is stated in the model card rather than hidden.

**The group key is the article body.** Verified against the organizers' own
split: their competition test set shares *zero* bodies with train and only 155
of 25,413 headlines (0.6%). The headline/body bipartite graph is a single
connected component, so component-based grouping would put everything in one
group and is useless here -- body-level grouping is both the strongest available
key and the published protocol.

**Severe imbalance, and the rare class is the one that matters.** In train:
unrelated 73%, discuss 18%, agree 7%, **disagree 1.7%**. A model that never
predicts `disagree` still scores 98% accuracy. Accuracy is banned from the
report for exactly this reason; per-class F1 and PR-AUC lead. `disagree` is also
the class the product cares most about -- a post *denying* a circulating claim
is the pushback signal -- so its per-class recall is the number to read first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from modeling.config import get_settings
from modeling.datasets.base import (
    BenchmarkDataset,
    DatasetInfo,
    DatasetUnavailable,
    drop_empty_text,
    normalize_text,
    register_dataset,
)

log = logging.getLogger(__name__)

#: FNC-1 label -> the contract's vocabulary. One-to-one; nothing invented.
LABEL_MAP = {
    "agree": "support",
    "disagree": "deny",
    "discuss": "discuss",
    "unrelated": "unrelated",
}

#: (stances, bodies) pairs, in the order the organizers ship them. The
#: `_unlabeled` variants are the blind competition inputs and are ignored: a
#: file with no Stance column cannot train or evaluate anything.
FILE_PAIRS = (
    ("train_stances.csv", "train_bodies.csv", "train"),
    ("competition_test_stances.csv", "competition_test_bodies.csv", "competition_test"),
)


@register_dataset
class FNC1(BenchmarkDataset):
    info = DatasetInfo(
        key="fnc1",
        label="FNC-1 (Fake News Challenge stance: headline vs article body)",
        access="open",
        url="https://github.com/FakeNewsChallenge/fnc-1",
        citation=(
            "Pomerleau, D., & Rao, D. (2017). Fake News Challenge Stage 1 (FNC-I): "
            "Stance Detection. http://www.fakenewschallenge.org/"
        ),
        expected_layout=[
            "train_stances.csv          (Headline, Body ID, Stance)",
            "train_bodies.csv           (Body ID, articleBody)",
            "competition_test_stances.csv   (optional, adds ~25k pairs)",
            "competition_test_bodies.csv    (optional)",
        ],
        manual_steps=[
            "git clone https://github.com/FakeNewsChallenge/fnc-1 data/benchmarks/stance",
            "the *_unlabeled.csv files are the blind competition inputs and are not used",
        ],
        notes=(
            "Covers all four contract stance classes, unlike SemEval-2016. Heavily "
            "imbalanced: disagree is 1.7% of train, and it is the class the product "
            "cares most about."
        ),
    )
    #: The article body. See the module docstring: this is the organizers' own
    #: split key, and the bipartite graph gives no finer honest grouping.
    group_col = "body_id"
    label_col = "label"
    #: Which of the organizers' files a row came from, so their published
    #: protocol can be reproduced without re-deriving it.
    domain_col = "official_split"

    def default_path(self) -> Path:
        """FNC-1 lives under ``benchmarks/stance/``, not ``benchmarks/fnc1/``.

        Both corpora answer the same question, so they share the slot: whichever
        one a user obtained ends up in the same directory, and
        ``train_stance_classifier`` picks whichever actually parses. Keeping two
        near-identical directory names would just invite putting the files in
        the wrong one.
        """
        return get_settings().benchmarks_dir / "stance"

    def fixture_path(self) -> Path:
        """Same shared slot on the fixture side, for the same reason."""
        from modeling.datasets.base import FIXTURE_ROOT

        return FIXTURE_ROOT / "stance"

    def _resolve_dir(self, path: Path) -> Path:
        """Accept the directory itself or a nested clone inside it."""
        if (path / "train_stances.csv").exists():
            return path
        for candidate in sorted(p for p in path.glob("*") if p.is_dir()):
            if (candidate / "train_stances.csv").exists():
                return candidate
        return path

    def validate(self, path: Path) -> None:
        if not path.exists():
            raise self.unavailable(path)
        base = self._resolve_dir(path)
        missing = [
            name
            for name in ("train_stances.csv", "train_bodies.csv")
            if not (base / name).exists()
        ]
        if missing:
            raise DatasetUnavailable(
                self.info.instructions(path) + f"\n  missing: {', '.join(missing)}"
            )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        dropped: dict[str, int] = {}
        base = self._resolve_dir(path)
        frames: list[pd.DataFrame] = []

        for stance_file, body_file, split_name in FILE_PAIRS:
            stance_path, body_path = base / stance_file, base / body_file
            if not (stance_path.exists() and body_path.exists()):
                if split_name != "train":
                    log.info(
                        "FNC-1: %s not present; training on the %d train pairs only",
                        split_name,
                        sum(len(f) for f in frames),
                    )
                continue

            stances = pd.read_csv(stance_path)
            bodies = pd.read_csv(body_path)
            for frame, name, required in (
                (stances, stance_file, {"Headline", "Body ID", "Stance"}),
                (bodies, body_file, {"Body ID", "articleBody"}),
            ):
                if not required <= set(frame.columns):
                    raise DatasetUnavailable(
                        f"{base / name} has columns {list(frame.columns)}; expected "
                        f"{sorted(required)}. This is not the FNC-1 CSV."
                    )

            before = len(stances)
            merged = stances.merge(bodies, on="Body ID", how="inner")
            orphans = before - len(merged)
            if orphans:
                # A stance row whose body is absent has no document to take a
                # stance on. Dropping is the only honest option.
                dropped["stance_without_body"] = dropped.get("stance_without_body", 0) + orphans

            # Body ids restart from 0 in each file, so namespace them by split
            # or the two files' bodies silently merge into one pseudo-group.
            merged["body_id"] = split_name + ":" + merged["Body ID"].astype(str)
            merged["official_split"] = split_name
            frames.append(merged)

        if not frames:
            raise DatasetUnavailable(f"FNC-1 at {base} parsed to zero rows")

        raw = pd.concat(frames, ignore_index=True)

        raw["label"] = raw["Stance"].astype(str).str.strip().str.lower().map(LABEL_MAP)
        unmapped = int(raw["label"].isna().sum())
        if unmapped:
            dropped["unmapped_label"] = unmapped
            raw = raw.loc[raw["label"].notna()]

        # The headline is the claim; the body is the text taking a stance on it.
        raw["target"] = normalize_text(raw["Headline"].fillna(""))
        raw["text"] = normalize_text(raw["articleBody"].fillna(""))
        raw = drop_empty_text(raw, "text", dropped, min_chars=20)
        raw = drop_empty_text(raw, "target", dropped, min_chars=5)

        raw["source_dataset"] = "fnc1"
        out = raw[
            ["body_id", "target", "text", "label", "official_split", "source_dataset"]
        ].reset_index(drop=True)

        rarest = out["label"].value_counts().idxmin()
        share = out["label"].value_counts(normalize=True).min()
        log.info(
            "FNC-1: %d pairs over %d bodies; rarest class %r is %.1f%% of rows -- report "
            "per-class F1 and PR-AUC, never accuracy",
            len(out),
            out["body_id"].nunique(),
            rarest,
            100 * share,
        )
        return out, dropped
