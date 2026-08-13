"""Benchmark loaders: local path in, validated DataFrame out.

Every benchmark this project uses is access-gated or manual-download. LIAR is a
direct download; FakeNewsNet ships ids and a crawler; CoAID is a git clone;
TwiBot-22, Cresci-2017, FaceForensics++, DFDC and Celeb-DF are all behind a
request form or a signed agreement.

The consequence is a hard rule: **no loader may download anything.** A loader
takes a path, validates what it finds, and either returns data or raises
:class:`DatasetUnavailable` carrying the exact steps a human must take. A loader
that silently returned an empty frame would produce a model trained on nothing
and a metrics file full of numbers.

Every loader also ships a ``demo()`` path backed by a tiny committed fixture, so
``modeling.cli ... --demo`` runs end to end on a clean clone with no network, no
credentials and no downloaded corpora.

The fixtures are *shape-faithful and value-meaningless*. They reproduce the real
column names, separators, label vocabularies and directory layouts so the
parsing code is genuinely exercised; they contain invented rows, so nothing
learned from them is a result. Anything trained on ``--demo`` data is plumbing
evidence, never a metric.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ingest.config import REPO_ROOT
from modeling.config import get_settings

log = logging.getLogger(__name__)

FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "benchmarks"


class DatasetUnavailable(FileNotFoundError):
    """The dataset is not on disk and cannot be fetched automatically.

    The message is the deliverable: it must tell a human exactly what to do.
    """


@dataclass(frozen=True)
class DatasetInfo:
    """Everything a model card needs to cite a training set."""

    key: str
    label: str
    #: "open" | "request_form" | "signed_agreement" | "crawler"
    access: str
    url: str
    citation: str
    #: What the loader expects to find under the given path.
    expected_layout: list[str] = field(default_factory=list)
    manual_steps: list[str] = field(default_factory=list)
    notes: str = ""

    def instructions(self, path: Path) -> str:
        lines = [
            f"{self.label} is not available at {path}.",
            f"  access: {self.access}   source: {self.url}",
            "  expected layout:",
            *[f"    {item}" for item in self.expected_layout],
        ]
        if self.manual_steps:
            lines.append("  steps:")
            lines.extend(f"    {i}. {step}" for i, step in enumerate(self.manual_steps, 1))
        lines.append(
            "  or run with --demo to exercise the pipeline on the committed fixture "
            "(shape-faithful, value-meaningless: not a result)."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class LoadedDataset:
    """A validated benchmark, plus the provenance that follows it into reports."""

    info: DatasetInfo
    frame: pd.DataFrame
    #: Column to group by when splitting. Never None -- a benchmark without a
    #: usable group key is a benchmark we cannot evaluate honestly.
    group_col: str
    label_col: str
    #: Column identifying a domain for cross-domain holdout, when one exists.
    domain_col: str | None = None
    is_demo: bool = False
    #: Rows the loader refused, by reason code. Same discipline as Phase 1.
    dropped: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset": self.info.key,
            "rows": len(self.frame),
            "groups": int(self.frame[self.group_col].nunique()),
            "group_col": self.group_col,
            "is_demo": self.is_demo,
            "dropped": dict(self.dropped),
        }
        if self.label_col in self.frame.columns:
            out["label_balance"] = (
                self.frame[self.label_col].value_counts(normalize=True).round(3).to_dict()
            )
        if self.domain_col and self.domain_col in self.frame.columns:
            out["domains"] = sorted(self.frame[self.domain_col].astype(str).unique().tolist())
        return out

    def describe(self) -> str:
        parts = [f"{self.info.key}: {len(self.frame)} rows"]
        if self.is_demo:
            parts.append("DEMO FIXTURE -- not a result")
        parts.append(f"grouped by {self.group_col} ({self.frame[self.group_col].nunique()} groups)")
        return ", ".join(parts)


class BenchmarkDataset(ABC):
    """One benchmark. Subclasses implement ``_read`` and declare their info."""

    #: Set by subclasses.
    info: DatasetInfo
    group_col: str = "group_id"
    label_col: str = "label"
    domain_col: str | None = None

    # --- paths -----------------------------------------------------------
    def default_path(self) -> Path:
        return get_settings().benchmarks_dir / self.info.key

    def fixture_path(self) -> Path:
        return FIXTURE_ROOT / self.info.key

    def resolve_path(self, path: Path | str | None, demo: bool) -> Path:
        if demo:
            return self.fixture_path()
        return Path(path) if path else self.default_path()

    # --- availability ----------------------------------------------------
    def available(self, path: Path | str | None = None, *, demo: bool = False) -> bool:
        """Whether ``load`` would succeed. Never raises, never downloads."""
        try:
            self.validate(self.resolve_path(path, demo))
            return True
        except DatasetUnavailable:
            return False

    @abstractmethod
    def validate(self, path: Path) -> None:
        """Raise :class:`DatasetUnavailable` unless ``path`` holds this dataset.

        Validate *structure*, not just existence. "The directory is there" is
        not the same as "the files this parser needs are there", and the
        difference shows up as a confusing pandas traceback three steps later.
        """

    @abstractmethod
    def _read(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        """Parse the dataset into a frame plus a drop-reason tally."""

    # --- loading ---------------------------------------------------------
    def load(self, path: Path | str | None = None, *, demo: bool = False) -> LoadedDataset:
        resolved = self.resolve_path(path, demo)
        self.validate(resolved)
        frame, dropped = self._read(resolved)
        frame = frame.reset_index(drop=True)

        for col in (self.group_col, self.label_col):
            if col not in frame.columns:
                raise ValueError(
                    f"{self.info.key} loader produced no {col!r} column. "
                    "A benchmark without a group key cannot be split honestly."
                )
        if frame[self.group_col].isna().any():
            n = int(frame[self.group_col].isna().sum())
            raise ValueError(
                f"{self.info.key}: {n} rows have a null {self.group_col}. Refusing: "
                "null group keys collapse into one pseudo-group and leak."
            )

        loaded = LoadedDataset(
            info=self.info,
            frame=frame,
            group_col=self.group_col,
            label_col=self.label_col,
            domain_col=self.domain_col,
            is_demo=demo,
            dropped=dropped,
        )
        if demo:
            log.warning(
                "%s loaded from the DEMO FIXTURE (%d rows). Any metric computed on this is "
                "plumbing evidence, not a result.",
                self.info.key,
                len(frame),
            )
        else:
            log.info("%s: %s", self.info.key, loaded.describe())
        if dropped:
            log.info(
                "%s dropped %s",
                self.info.key,
                ", ".join(f"{k}={v}" for k, v in dropped.items() if v),
            )
        return loaded

    def groups(self, frame: pd.DataFrame) -> pd.Series:
        """The split key. Exposed so callers never invent their own."""
        return frame[self.group_col].astype(str)

    def unavailable(self, path: Path) -> DatasetUnavailable:
        return DatasetUnavailable(self.info.instructions(path))


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def require_files(dataset: BenchmarkDataset, path: Path, *names: str) -> None:
    """Raise unless every named file exists under ``path``."""
    if not path.exists():
        raise dataset.unavailable(path)
    missing = [name for name in names if not (path / name).exists()]
    if missing:
        raise DatasetUnavailable(
            dataset.info.instructions(path) + f"\n  missing: {', '.join(missing)}"
        )


def require_any(dataset: BenchmarkDataset, path: Path, pattern: str) -> list[Path]:
    """Raise unless at least one file matches ``pattern`` under ``path``."""
    if not path.exists():
        raise dataset.unavailable(path)
    hits = sorted(path.rglob(pattern))
    if not hits:
        raise DatasetUnavailable(
            dataset.info.instructions(path) + f"\n  no files matching {pattern!r} under {path}"
        )
    return hits


def normalize_text(series: pd.Series) -> pd.Series:
    """Collapse whitespace and strip. Never lowercases: casing is signal
    (ALL-CAPS headlines) and the tokenizers handle it."""
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def drop_empty_text(
    frame: pd.DataFrame, column: str, dropped: dict[str, int], *, min_chars: int = 3
) -> pd.DataFrame:
    before = len(frame)
    out = frame.loc[frame[column].astype(str).str.len() >= min_chars]
    removed = before - len(out)
    if removed:
        dropped["empty_text"] = dropped.get("empty_text", 0) + removed
    return out


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type[BenchmarkDataset]] = {}


def register_dataset(cls: type[BenchmarkDataset]) -> type[BenchmarkDataset]:
    _REGISTRY[cls.info.key] = cls
    return cls


def get_dataset(key: str) -> BenchmarkDataset:
    if key not in _REGISTRY:
        raise KeyError(f"unknown benchmark {key!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def all_datasets() -> list[BenchmarkDataset]:
    return [cls() for cls in _REGISTRY.values()]


def availability_table(demo: bool = False) -> list[dict[str, Any]]:
    """What is on disk right now. Backs ``modeling.cli datasets``."""
    rows = []
    for dataset in all_datasets():
        path = dataset.resolve_path(None, demo)
        rows.append(
            {
                "key": dataset.info.key,
                "access": dataset.info.access,
                "path": str(path),
                "available": dataset.available(demo=demo),
                "fixture": dataset.available(demo=True),
            }
        )
    return rows
