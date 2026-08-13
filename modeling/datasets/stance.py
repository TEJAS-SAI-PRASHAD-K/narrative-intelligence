"""Stance benchmarks: SemEval-2016 Task 6 and FEVER-style claim/evidence pairs.

Stance is conditioned on a *pair*: (claim, post) -> support / deny / discuss /
unrelated. At inference time the claim comes from a narrative's representative
post (Module A2); at training time it comes from the benchmark's target column.

The group key is the **claim/target**, never the post. SemEval-2016 has five
targets in its training set and one unseen target in the test set, and that
structure is the whole point of the benchmark: a model that memorizes "posts
about Target X are usually AGAINST" scores well within a target and collapses
outside it.

Label mapping. SemEval uses FAVOR / AGAINST / NONE against a target; the product
wants support / deny / discuss / unrelated. FAVOR -> support and AGAINST -> deny
are clean. NONE is *not* clean: it conflates "mentions the target without taking
a side" (discuss) with "unrelated". This loader maps NONE -> discuss and
documents that ``unrelated`` is therefore **unattested in SemEval training
data** -- a model trained on SemEval alone will never predict it. FEVER's
NOT ENOUGH INFO maps to discuss for the same reason.
"""

from __future__ import annotations

import hashlib
import json
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

STANCE_LABELS = ("support", "deny", "discuss", "unrelated")

SEMEVAL_MAP = {
    "favor": "support",
    "against": "deny",
    # See the module docstring: NONE is a conflation, resolved to discuss.
    "none": "discuss",
}

FEVER_MAP = {
    "supports": "support",
    "refutes": "deny",
    "notenoughinfo": "discuss",
    "not enough info": "discuss",
}


@register_dataset
class SemEvalStance(BenchmarkDataset):
    info = DatasetInfo(
        key="stance",
        label="SemEval-2016 Task 6 (stance toward a target) / FEVER-style pairs",
        access="request_form",
        url="https://alt.qcri.org/semeval2016/task6/",
        citation=(
            "Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., & Cherry, C. (2016). "
            "SemEval-2016 Task 6: Detecting Stance in Tweets. SemEval-2016."
        ),
        expected_layout=[
            "trainingdata-all-annotations.txt   (tab-separated: ID, Target, Tweet, Stance)",
            "testdata-taskA-all-annotations.txt",
            "or fever/train.jsonl  (claim, label, evidence)",
        ],
        manual_steps=[
            "request the SemEval-2016 Task 6 data from the task organisers",
            "place the annotation .txt files in data/benchmarks/stance/",
            "FEVER alternative: download train.jsonl into data/benchmarks/stance/fever/",
        ],
        notes=(
            "'unrelated' is unattested in SemEval; a SemEval-only model cannot predict it. "
            "This is a coverage gap, not a bug -- see the model card."
        ),
    )
    #: The target/claim. Splitting inside a target leaks the target's prior.
    group_col = "claim_id"
    label_col = "label"
    domain_col = "target"

    def validate(self, path: Path) -> None:
        if not path.exists():
            raise self.unavailable(path)
        has_semeval = any(path.glob("*annotations*.txt")) or any(path.glob("*.txt"))
        has_fever = any(path.rglob("*.jsonl"))
        if not (has_semeval or has_fever):
            raise DatasetUnavailable(
                self.info.instructions(path)
                + f"\n  no annotation .txt or FEVER .jsonl files under {path}"
            )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        dropped: dict[str, int] = {}
        frames = []
        for file in sorted(path.glob("*.txt")):
            frames.append(self._read_semeval(file, dropped))
        for file in sorted(path.rglob("*.jsonl")):
            frames.append(self._read_fever(file, dropped))
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            raise DatasetUnavailable(f"stance data at {path} parsed to zero rows")

        raw = pd.concat(frames, ignore_index=True)
        raw = drop_empty_text(raw, "text", dropped, min_chars=5)
        before = len(raw)
        raw = raw.loc[raw["label"].isin(STANCE_LABELS)]
        dropped["unmapped_label"] = before - len(raw)
        raw["source_dataset"] = "stance"
        return raw[["claim_id", "target", "text", "label", "source_dataset"]], dropped

    def _read_semeval(self, file: Path, dropped: dict[str, int]) -> pd.DataFrame | None:
        try:
            frame = pd.read_csv(file, sep="\t", dtype=str, encoding="latin-1", on_bad_lines="warn")
        except (pd.errors.ParserError, UnicodeDecodeError):
            dropped["unparseable_file"] = dropped.get("unparseable_file", 0) + 1
            return None
        columns = {c.strip().lower(): c for c in frame.columns}
        if not {"target", "tweet", "stance"} <= set(columns):
            dropped["not_semeval_format"] = dropped.get("not_semeval_format", 0) + 1
            return None
        out = pd.DataFrame(
            {
                "target": normalize_text(frame[columns["target"]].fillna("")),
                "text": normalize_text(frame[columns["tweet"]].fillna("")),
                "label": frame[columns["stance"]]
                .fillna("")
                .str.strip()
                .str.lower()
                .map(SEMEVAL_MAP)
                .fillna("drop"),
            }
        )
        out["claim_id"] = out["target"].str.lower().map(_hash16)
        return out

    def _read_fever(self, file: Path, dropped: dict[str, int]) -> pd.DataFrame | None:
        rows = []
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    dropped["bad_jsonl_line"] = dropped.get("bad_jsonl_line", 0) + 1
                    continue
                claim = str(payload.get("claim", "")).strip()
                label = str(payload.get("label", "")).strip().lower().replace("_", "")
                if not claim:
                    continue
                # FEVER's evidence is sentence ids into Wikipedia, not text.
                # Without a Wikipedia dump the (claim, evidence) pair cannot be
                # reconstructed, so the claim doubles as the post. That is a
                # weaker training signal and is recorded as such.
                rows.append(
                    {
                        "target": claim,
                        "text": claim,
                        "label": FEVER_MAP.get(label, "drop"),
                        "claim_id": _hash16(claim.lower()),
                    }
                )
        if not rows:
            return None
        dropped["fever_claim_as_post"] = dropped.get("fever_claim_as_post", 0) + len(rows)
        return pd.DataFrame(rows)


def _hash16(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
