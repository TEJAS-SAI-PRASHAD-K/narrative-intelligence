"""Settings and on-disk layout. Secrets come from ``.env``, never from code."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


class Settings(BaseSettings):
    """Everything tunable, in one place.

    Every credential is optional on purpose: the pipeline must run to completion
    on a clean clone with an empty ``.env``, skipping the sources it cannot
    authenticate and saying so loudly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Mastodon ---
    mastodon_api_base_url: str = "https://mastodon.social"
    mastodon_access_token: str | None = None

    # --- YouTube ---
    youtube_api_key: str | None = None
    youtube_max_searches_per_day: int = 20
    youtube_daily_quota_units: int = 10_000

    # --- News ---
    newsapi_key: str | None = None

    # --- GDELT (no credentials; these just bound the pull) ---
    gdelt_max_records: int = 250
    gdelt_lookback_days: int = 7

    # --- Kaggle ---
    kaggle_username: str | None = None
    kaggle_key: str | None = None

    # --- paths / behaviour ---
    data_dir: Path = Path("data")
    user_agent: str = "narrative-intelligence-research/0.1 (academic capstone)"
    log_level: str = "INFO"
    http_rate_limit_per_sec: float = 2.0

    @field_validator(
        "mastodon_access_token",
        "youtube_api_key",
        "newsapi_key",
        "kaggle_username",
        "kaggle_key",
        mode="before",
    )
    @classmethod
    def _blank_is_absent(cls, v: Any) -> Any:
        """``KEY=`` in a .env file means "not set", not "the empty string"."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("data_dir", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    # --- derived layout --------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def checkpoint_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def benchmarks_dir(self) -> Path:
        return self.data_dir / "benchmarks"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    def raw_dir_for(self, source: str) -> Path:
        path = self.raw_dir / source
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.normalized_dir, self.checkpoint_dir):
            path.mkdir(parents=True, exist_ok=True)

    # --- credential gates ------------------------------------------------
    def has_credentials(self, source: str) -> bool:
        """Whether a source can run at all. Drives graceful degradation."""
        return {
            "reddit_convokit": True,  # no credentials needed
            "reddit_kaggle": bool(self.kaggle_key) or Path.home().joinpath(".kaggle/kaggle.json").exists(),
            "mastodon": bool(self.mastodon_access_token),
            "gdelt": True,  # open data
            "news_rss": True,  # RSS needs nothing; NewsAPI is an optional bonus
            "youtube": bool(self.youtube_api_key),
        }.get(source, True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_yaml(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``. Returns ``{}`` if absent."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def sources_config() -> dict[str, Any]:
    """Which subreddits / instances / feeds / dataset slugs to pull."""
    return load_yaml("sources")


def topics_config() -> dict[str, Any]:
    """Seed keywords and boolean queries defining the case under study."""
    return load_yaml("topics")


def setup_logging(level: str | None = None) -> None:
    """Consistent, greppable logs. Every drop and every skip goes through here."""
    resolved = (level or os.environ.get("LOG_LEVEL") or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries are chatty at DEBUG and drown out our own lines.
    for noisy in ("urllib3", "requests", "charset_normalizer", "filelock", "convokit"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
