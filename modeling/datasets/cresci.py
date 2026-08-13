"""Cresci-2017: genuine accounts vs. several distinct bot campaigns.

Requested through the Bot Repository. Ships as one directory per account class,
each holding ``users.csv`` and ``tweets.csv``:

    genuine_accounts.csv/         (human)
    social_spambots_1.csv/        (retweeters of an Italian political candidate)
    social_spambots_2.csv/        (spammers of a mobile app)
    social_spambots_3.csv/        (spammers of amazon.com products)
    traditional_spambots_1..4.csv/
    fake_followers.csv/

**The campaign structure is the trap.** Each ``social_spambots_N`` directory is
one botnet running one template. Accounts inside it are near-identical by
construction. Group by account id and 5-fold CV will place siblings from the
same botnet in train and test, and the model will report near-perfect F1 for
having memorized one campaign's signature.

So the group key here is the **campaign directory**, not the account. That is
strictly stronger, it is what ``group_col`` returns, and it is the difference
between an F1 that means "detects bots" and one that means "recognizes these
seven botnets".
"""

from __future__ import annotations

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

#: directory stem -> (label, campaign family). 1 = bot.
CAMPAIGNS = {
    "genuine_accounts": (0, "genuine"),
    "social_spambots_1": (1, "social_spambots"),
    "social_spambots_2": (1, "social_spambots"),
    "social_spambots_3": (1, "social_spambots"),
    "traditional_spambots_1": (1, "traditional_spambots"),
    "traditional_spambots_2": (1, "traditional_spambots"),
    "traditional_spambots_3": (1, "traditional_spambots"),
    "traditional_spambots_4": (1, "traditional_spambots"),
    "fake_followers": (1, "fake_followers"),
}

#: Cresci's users.csv columns that survive the cross-platform intersection.
PORTABLE_COLUMNS = {
    "id": "native_id",
    "screen_name": "handle",
    "created_at": "created_at",
    "followers_count": "followers",
    "friends_count": "following",
    "statuses_count": "post_count",
    "description": "description",
}


@register_dataset
class Cresci2017(BenchmarkDataset):
    info = DatasetInfo(
        key="cresci",
        label="Cresci-2017 (genuine accounts + labelled bot campaigns)",
        access="request_form",
        url="https://botometer.osome.iu.edu/bot-repository/datasets.html",
        citation=(
            "Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017). "
            "The Paradigm-Shift of Social Spambots. WWW '17 Companion."
        ),
        expected_layout=[
            "genuine_accounts.csv/users.csv",
            "social_spambots_1.csv/users.csv",
            "... one directory per account class, each containing users.csv",
        ],
        manual_steps=[
            "request access at the Bot Repository (link above)",
            "unpack the archive into data/benchmarks/cresci/",
            "only users.csv is read; tweets.csv is optional and used for content features",
        ],
        notes=(
            "Group by campaign, not account: each spambot directory is one botnet running "
            "one template, and account-level grouping leaks it."
        ),
    )
    #: The campaign directory. See the module docstring for why not account_id.
    group_col = "campaign"
    label_col = "label"
    domain_col = "campaign_family"

    def validate(self, path: Path) -> None:
        if not path.exists():
            raise self.unavailable(path)
        found = _campaign_dirs(path)
        if not found:
            raise DatasetUnavailable(
                self.info.instructions(path)
                + f"\n  no <class>.csv/users.csv directories found under {path}"
            )
        labels = {CAMPAIGNS[name][0] for name, _ in found}
        if len(labels) < 2:
            raise DatasetUnavailable(
                self.info.instructions(path)
                + "\n  only one class present; both genuine and bot directories are required"
            )

    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        dropped: dict[str, int] = {}
        frames = []
        for campaign, users_file in _campaign_dirs(path):
            label, family = CAMPAIGNS[campaign]
            try:
                frame = pd.read_csv(users_file, dtype=str, on_bad_lines="warn", low_memory=False)
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                log.warning("skipping unparseable %s: %s", users_file, exc)
                dropped["unparseable_file"] = dropped.get("unparseable_file", 0) + 1
                continue
            columns = {c.strip().lower(): c for c in frame.columns}
            if "id" not in columns:
                dropped["no_id_column"] = dropped.get("no_id_column", 0) + 1
                continue
            out = pd.DataFrame(index=frame.index)
            for source_key, dest in PORTABLE_COLUMNS.items():
                out[dest] = frame[columns[source_key]] if source_key in columns else pd.NA
            out["label"] = label
            out["campaign"] = campaign
            out["campaign_family"] = family
            frames.append(out)

        if not frames:
            raise DatasetUnavailable(f"Cresci-2017 at {path} parsed to zero usable rows")
        merged = pd.concat(frames, ignore_index=True)

        before = len(merged)
        merged = merged.loc[merged["native_id"].notna()]
        dropped["user_without_id"] = before - len(merged)

        for column in ("followers", "following", "post_count"):
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        # Cresci's created_at is Twitter's legacy format ("Mon Jan 02 15:04:05
        # +0000 2006"); pandas parses it, but not without being told to be
        # lenient about the mixed formats across the archives.
        merged["created_at"] = pd.to_datetime(
            merged["created_at"], errors="coerce", utc=True, format="mixed"
        )
        unparsed = int(merged["created_at"].isna().sum())
        if unparsed:
            dropped["unparseable_created_at"] = unparsed

        merged["account_id"] = "twitter:" + merged["native_id"].astype(str)
        merged["platform"] = "twitter"
        merged["source_dataset"] = "cresci"
        merged["description"] = merged["description"].fillna("")
        merged["handle"] = merged["handle"].fillna("")
        return merged, dropped


def _campaign_dirs(path: Path) -> list[tuple[str, Path]]:
    """Locate ``<class>.csv/users.csv`` directories, tolerating flat layouts."""
    found: list[tuple[str, Path]] = []
    for campaign in CAMPAIGNS:
        for candidate in (
            path / f"{campaign}.csv" / "users.csv",
            path / campaign / "users.csv",
            path / f"{campaign}_users.csv",
        ):
            if candidate.exists():
                found.append((campaign, candidate))
                break
    return found
