"""Phase 2 settings, seeds and on-disk layout.

One place for every knob that changes a number in a report. If a seed, a device
choice or a library version is not visible here, a rerun that disagrees with the
committed metrics cannot be diagnosed.

Phase 1's ``ingest.config.Settings`` is read but never modified: this module
composes on top of it and adds the modeling-only paths. ``ingest/`` is a
read-only contract to Phase 2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ingest.config import REPO_ROOT
from ingest.config import get_settings as get_ingest_settings

log = logging.getLogger(__name__)

CONFIG_DIR = REPO_ROOT / "configs"

#: The single seed. Everything stochastic in Phase 2 derives from it.
#: Changing it is a deliberate act that invalidates every committed metric.
DEFAULT_SEED = 20260813


class ModelingSettings(BaseSettings):
    """Modeling-side configuration.

    Note the asymmetry that drives most of these defaults: training happens on a
    Colab T4, inference happens on a CPU worker with no GPU. Anything that is
    only viable on the GPU must either be small enough for CPU inference or have
    its scores precomputed into Parquet here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- determinism -----------------------------------------------------
    seed: int = DEFAULT_SEED

    # --- device ----------------------------------------------------------
    #: "auto" resolves to cuda -> mps -> cpu. Set explicitly to "cpu" to
    #: reproduce the hosted-inference path on a laptop with a GPU.
    device: str = "auto"

    # --- language policy -------------------------------------------------
    #: v1 is English-only. Non-English text is *skipped with a reason code*,
    #: never scored by an English model and reported as if it were valid.
    #: See modeling/aux/*.py and the README limitations section.
    languages: tuple[str, ...] = ("en",)
    #: When ``lang`` is None (common for short text), do we score it? Phase 1
    #: leaves lang None rather than guessing; we treat None as "assume English"
    #: only if this is True, and we record the assumption per row.
    score_unknown_language: bool = True

    # --- LLM (bounded use only; see modeling/text/summarize.py) ----------
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    #: Hard ceiling so a runaway loop cannot bill an unbounded amount.
    llm_max_calls_per_run: int = 500

    # --- checkpoint hosting ----------------------------------------------
    #: Where registry.py looks for remote checkpoints. Never a git remote.
    hf_repo: str | None = None
    hf_token: str | None = None
    gdrive_dir: Path | None = None

    # --- paths -----------------------------------------------------------
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")

    log_level: str = "INFO"

    @field_validator("anthropic_api_key", "hf_repo", "hf_token", mode="before")
    @classmethod
    def _blank_is_absent(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("data_dir", "artifacts_dir", "models_dir", "gdrive_dir", mode="after")
    @classmethod
    def _absolute(cls, v: Path | None) -> Path | None:
        if v is None:
            return None
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    @field_validator("languages", mode="before")
    @classmethod
    def _split_languages(cls, v: Any) -> Any:
        if isinstance(v, str):
            return tuple(part.strip().lower() for part in v.split(",") if part.strip())
        return v

    # --- derived layout --------------------------------------------------
    @property
    def normalized_dir(self) -> Path:
        """Phase 1's corpus. Read-only to Phase 2."""
        return self.data_dir / "normalized"

    @property
    def authors_dir(self) -> Path:
        return self.data_dir / "authors"

    @property
    def input_manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def benchmarks_dir(self) -> Path:
        return self.data_dir / "benchmarks"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def scored_dir(self) -> Path:
        return self.data_dir / "scored"

    @property
    def scored_manifest_path(self) -> Path:
        return self.scored_dir / "manifest.json"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def eval_dir(self) -> Path:
        return self.artifacts_dir / "eval"

    @property
    def model_card_dir(self) -> Path:
        return self.artifacts_dir / "model_cards"

    @property
    def error_analysis_dir(self) -> Path:
        return self.artifacts_dir / "error_analysis"

    def ensure_dirs(self) -> None:
        for path in (
            self.embeddings_dir,
            self.scored_dir,
            self.cache_dir,
            self.models_dir,
            self.eval_dir,
            self.model_card_dir,
            self.error_analysis_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # --- helpers ---------------------------------------------------------
    def resolve_device(self) -> str:
        """Resolve ``device="auto"``. Import torch lazily: the aux-free paths
        (splits, io, features) must import without torch installed."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def language_allowed(self, lang: str | None) -> bool:
        """The one place the English-only policy is decided."""
        if lang is None:
            return self.score_unknown_language
        return lang.lower() in self.languages


@lru_cache(maxsize=1)
def get_settings() -> ModelingSettings:
    return ModelingSettings()


def set_all_seeds(seed: int | None = None) -> int:
    """Seed every RNG this project touches. Call once, at entry points.

    Returns the seed so callers can log it into an artifact without re-reading
    settings (and so the seed that was *actually used* is what gets reported).
    """
    resolved = seed if seed is not None else get_settings().seed
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    try:
        import numpy as np

        np.random.seed(resolved)
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass
    try:
        import torch

        torch.manual_seed(resolved)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved)
        # Deterministic cuDNN costs throughput; on a T4 fine-tune that cost is
        # worth paying, because a non-reproducible F1 is not a result.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return resolved


def library_versions() -> dict[str, str]:
    """Versions of everything that can move a metric. Stamped into every eval
    artifact, because "it reproduced on my machine" is not reproducibility."""
    import importlib.metadata as md
    import platform

    out: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "xgboost",
        "shap",
        "torch",
        "transformers",
        "sentence-transformers",
        "timm",
        "networkx",
        "anthropic",
    ):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "absent"
    return out


def manifest_hash(path: Path | None = None) -> str:
    """Stable hash of Phase 1's manifest: identifies the input corpus version.

    Written into every eval artifact so a metric can be tied to the exact corpus
    it was computed on. Absent manifest -> a sentinel, not a crash: the demo
    path runs on fixtures with no manifest at all.
    """
    resolved = path or get_settings().input_manifest_path
    if not resolved.exists():
        return "no-manifest"
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unreadable-manifest"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def run_fingerprint(seed: int | None = None) -> dict[str, Any]:
    """The provenance block embedded in every ``metrics.json``."""
    settings = get_settings()
    return {
        "seed": seed if seed is not None else settings.seed,
        "device": settings.resolve_device(),
        "input_manifest_hash": manifest_hash(),
        "languages": list(settings.languages),
        "library_versions": library_versions(),
    }


def load_yaml(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``; ``{}`` when absent."""
    import yaml

    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        log.warning("config %s not found at %s; using defaults", name, path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def models_config() -> dict[str, Any]:
    """Per-module hyperparameters, checkpoint names + versions, thresholds."""
    return load_yaml("models")


def scoring_config() -> dict[str, Any]:
    """Batch scoring config: sources, date ranges, batch sizes."""
    return load_yaml("scoring")


def module_config(module: str) -> dict[str, Any]:
    """Hyperparameters for one module, with the shared defaults merged in."""
    cfg = models_config()
    merged: dict[str, Any] = dict(cfg.get("defaults") or {})
    merged.update(cfg.get(module) or {})
    return merged


def model_version(module: str) -> str:
    """The version string that lands in every scored row's ``model_versions``.

    Sourced from ``configs/models.yaml`` so bumping a model is a config change
    and a stale score is detectable by comparison, not by memory.
    """
    return str(module_config(module).get("version", "v0.0.0-unset"))


def ingest_settings():
    """Phase 1 settings, for reading the corpus. Never mutate the result."""
    return get_ingest_settings()


def setup_logging(level: str | None = None) -> None:
    """Same discipline as Phase 1: every skip and every null has a logged reason."""
    resolved = (level or os.environ.get("LOG_LEVEL") or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in (
        "urllib3",
        "requests",
        "filelock",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "matplotlib",
        "numba",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
