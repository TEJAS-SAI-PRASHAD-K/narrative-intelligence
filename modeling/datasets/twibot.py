"""TwiBot-22: a large Twitter bot-detection benchmark, behind a request form.

TwiBot-22 is the biggest labeled bot dataset available, and the most
misleading one to train on naively. Two traps:

**Trap 1: features you cannot compute.** The full release carries ~40 features
per account, many of them Twitter-specific (verified status, profile banner,
follow-graph neighbourhoods). Our corpus is Reddit/Mastodon/YouTube. A model
trained on 40 features when we can compute 12 is not a model, it is a lookup
table for a platform we do not have. The loader therefore emits a *raw* account
frame and leaves the intersection to ``modeling/accounts/features.py``, which
computes the same features on both sides from the same code.

**Trap 2: graph-structured groups.** Accounts in TwiBot-22 come from crawled
follow neighbourhoods. Two accounts one hop apart share almost all their
network features. Grouping by account id alone is not enough where a community
id is available, so this loader exposes both.

Nothing here downloads. TwiBot-22 requires an accepted request form.
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

#: The account metadata TwiBot-22 shares with Mastodon, and therefore the only
#: part of it this project can use. Anything Twitter-only is deliberately not
#: read: a feature we cannot compute on our own corpus is a feature that will
#: not exist at inference time.
PORTABLE_FIELDS = {
    "id": "native_id",
    "username": "handle",
    "created_at": "created_at",
    "followers_count": "followers",
    "following_count": "following",
    "tweet_count": "post_count",
    "description": "description",
}

#: TwiBot-22's public_metrics nests the counts one level down.
NESTED_METRICS = {
    "followers_count": "followers",
    "following_count": "following",
    "tweet_count": "post_count",
    "listed_count": "listed",
}


@register_dataset
class TwiBot22(BenchmarkDataset):
    info = DatasetInfo(
        key="twibot",
        label="TwiBot-22 (Twitter bot detection benchmark)",
        access="request_form",
        url="https://twibot22.github.io/",
        citation=(
            "Feng, S., Tan, Z., Wan, H., et al. (2022). TwiBot-22: Towards Graph-Based "
            "Twitter Bot Detection Benchmark. NeurIPS 2022 Datasets and Benchmarks."
        ),
        expected_layout=[
            "label.csv        (id, label)  where label is 'bot' or 'human'",
            "user.json        (list of user objects with public_metrics)",
            "split.csv        (optional; ignored -- we re-split grouped)",
        ],
        manual_steps=[
            "submit the request form at https://twibot22.github.io/",
            "place label.csv and user.json in data/benchmarks/twibot/",
            "the edge and tweet files are not used by this project",
        ],
        notes=(
            "Twitter-only. Cross-platform transfer to Mastodon/Reddit is untested and "
            "should be assumed degraded until measured."
        ),
    )
    group_col = "account_id"
    label_col = "label"
    domain_col = "platform"

    def validate(self, path: Path) -> None:
        if not path.exists():
            raise self.unavailable(path)
        label_file = _first_existing(path, "label.csv", "labels.csv")
        user_file = _first_existing(path, "user.json", "users.json", "node.json")
        missing = []
        if label_file is None:
            missing.append("label.csv")
        if user_file is None:
            missing.append("user.json")
        if missing:
            raise DatasetUnavailable(
                self.info.instructions(path) + f"\n  missing: {', '.join(missing)}"
            )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        dropped: dict[str, int] = {}
        label_file = _first_existing(path, "label.csv", "labels.csv")
        user_file = _first_existing(path, "user.json", "users.json", "node.json")

        labels = pd.read_csv(label_file, dtype=str)
        columns = {c.strip().lower(): c for c in labels.columns}
        if "id" not in columns or "label" not in columns:
            raise DatasetUnavailable(
                f"{label_file} must have 'id' and 'label' columns; found {list(labels.columns)}"
            )
        labels = labels.rename(columns={columns["id"]: "native_id", columns["label"]: "label_raw"})
        labels["label"] = (
            labels["label_raw"].astype(str).str.strip().str.lower().map({"bot": 1, "human": 0})
        )
        before = len(labels)
        labels = labels.loc[labels["label"].notna()]
        dropped["unknown_label"] = before - len(labels)
        labels["label"] = labels["label"].astype(int)

        users = _read_users(user_file, dropped)
        merged = users.merge(labels[["native_id", "label"]], on="native_id", how="inner")
        dropped["unlabeled_user"] = len(users) - len(merged)
        if not len(merged):
            raise DatasetUnavailable(
                f"TwiBot-22 at {path}: no user ids matched label.csv. Check both files are "
                "from the same release."
            )

        merged["account_id"] = "twitter:" + merged["native_id"].astype(str)
        merged["platform"] = "twitter"
        # TwiBot-22 has no campaign column; the account is the finest honest
        # group key available here. Cresci-2017 does have one, and uses it.
        merged["campaign"] = merged["account_id"]
        merged["source_dataset"] = "twibot"
        return merged, dropped


def _read_users(path: Path, dropped: dict[str, int]) -> pd.DataFrame:
    """Parse TwiBot-22's user file, keeping only the portable fields.

    The file is a JSON array in the public release and JSON-lines in some
    mirrors; both are accepted, because failing on the mirror would send a user
    on a pointless hunt.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    payload: list[dict]
    if text.startswith("["):
        payload = json.loads(text)
    else:
        payload = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                payload.append(json.loads(line))
            except json.JSONDecodeError:
                dropped["bad_json_line"] = dropped.get("bad_json_line", 0) + 1

    rows = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        row: dict[str, object] = {}
        for source_key, dest in PORTABLE_FIELDS.items():
            if source_key in entry:
                row[dest] = entry[source_key]
        metrics = entry.get("public_metrics") or {}
        if isinstance(metrics, dict):
            for source_key, dest in NESTED_METRICS.items():
                if source_key in metrics:
                    row[dest] = metrics[source_key]
        if "native_id" not in row:
            dropped["user_without_id"] = dropped.get("user_without_id", 0) + 1
            continue
        # Keep the id verbatim. TwiBot-22 prefixes user ids with "u" in both
        # user.json and label.csv, so normalizing one side breaks the join --
        # and it fails as "zero labelled users", which reads like a download
        # problem rather than a parsing one.
        row["native_id"] = str(row["native_id"])
        rows.append(row)

    frame = pd.DataFrame(rows)
    for column in ("followers", "following", "post_count"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = pd.NA
    if "created_at" in frame.columns:
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    else:
        frame["created_at"] = pd.NaT
    if "description" not in frame.columns:
        frame["description"] = ""
    if "handle" not in frame.columns:
        frame["handle"] = ""
    return frame


def _first_existing(path: Path, *names: str) -> Path | None:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None
