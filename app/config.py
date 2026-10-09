"""Phase 4 settings. Twelve-factor, so Phase 6 can move this to Render/Neon by
changing environment variables and nothing else.

This composes on top of Phase 1's ``ingest.config.Settings`` the same way
Phase 2's ``modeling.config`` does: Phase 1 owns the corpus paths and the source
credentials, and this module adds the serving-side knobs. Phase 1 is a read-only
contract here.

Every default that names a host is deliberately *not* a host: ``DATABASE_URL``
and ``REDIS_URL`` have no default at all, because a default of
``localhost:5432`` is the single most common way a container ends up silently
talking to the wrong database. The one place a hostname is allowed to appear is
``.env.example``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ingest.config import REPO_ROOT
from ingest.config import get_settings as get_ingest_settings

log = logging.getLogger(__name__)

CONFIG_DIR = REPO_ROOT / "configs"

#: Bumped whenever a route's *shape* changes. The frontend pins this.
API_VERSION = "v1"


class ApiSettings(BaseSettings):
    """Serving configuration.

    Read once at import of :func:`get_api_settings` and cached. Anything that
    changes a number the API returns belongs in ``configs/*.yaml`` and carries a
    version string instead of living here -- settings are for *where things are*,
    config files are for *what the numbers mean*.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- service identity ------------------------------------------------
    environment: Literal["local", "ci", "staging", "production"] = "local"
    log_level: str = "INFO"
    #: When 1, every router answers from ``app/mock`` instead of the database.
    #: This is what lets Phase 5 build against the contract before the ETL runs.
    demo_mode: bool = False

    # --- datastores (no defaults on purpose; see module docstring) -------
    database_url: str = Field(
        default="",
        description="SQLAlchemy async URL, e.g. postgresql+psycopg://user:pw@db:5432/narrative",
    )
    #: Alembic and the ETL's COPY path both want a synchronous driver.
    #: Derived from ``database_url`` when unset so there is one source of truth.
    database_url_sync: str = ""
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_statement_timeout_ms: int = 30_000

    redis_url: str = ""
    celery_broker_url: str = ""
    celery_result_backend: str = ""

    # --- auth ------------------------------------------------------------
    #: Server-side pepper mixed into every API key hash. Rotating it invalidates
    #: every issued key, which is the intended emergency lever.
    api_key_pepper: str | None = None
    #: Only used by ``make seed`` / first boot to mint an admin key when the
    #: table is empty. Never referenced on the request path.
    bootstrap_admin_key: str | None = None
    rate_limit_per_minute: int = 600
    rate_limit_burst: int = 60
    #: ``NoDecode`` is load-bearing, not decoration. pydantic-settings treats a
    #: tuple as a "complex" type and JSON-decodes it **in the settings source**,
    #: before any field validator runs -- so the ``_split_csv`` validator below
    #: never saw the value and `CORS_ORIGINS=http://a,http://b` raised
    #: ``SettingsError: error parsing value for field "cors_origins"``.
    #:
    #: That is the exact line shipped in ``.env.example``, which ``make setup``
    #: copies to ``.env``, so a clean clone could not construct settings at all:
    #: the API would not boot and all 24 contract tests errored. ``NoDecode``
    #: turns the source-level decoding off and hands the raw string to the
    #: validator, which is what it was always written to receive.
    cors_origins: Annotated[tuple[str, ...], NoDecode] = (
        "http://localhost:5173",
        "http://localhost:3000",
    )

    # --- embeddings ------------------------------------------------------
    #: pgvector columns are fixed-width, so this pins the deployment. Changing
    #: it is a migration, not a config tweak -- see app/models/embeddings.py.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 64

    # --- model checkpoints ----------------------------------------------
    #: Mounted as a volume in docker-compose. Absent checkpoints degrade the
    #: matching scorer to ``unavailable`` in /readyz; they never block boot.
    models_dir: Path = Path("models")
    checkpoint_embed: str | None = None
    checkpoint_misinfo: str | None = None
    checkpoint_stance: str | None = None
    checkpoint_bot: str | None = None
    checkpoint_deepfake: str | None = None

    # --- corpus / etl ----------------------------------------------------
    data_dir: Path = Path("data")
    #: Row count above which monthly RANGE partitioning of ``posts`` pays for
    #: itself. Documented in docs/data-model.md; the loader warns past it.
    partition_threshold_rows: int = 5_000_000
    etl_copy_batch_rows: int = 50_000

    # --- graph -----------------------------------------------------------
    max_graph_nodes: int = 20_000
    graph_bucket_hours: int = 12
    #: Above this, /network/graph refuses to hand raw edges to the browser and
    #: serves a precomputed layout instead.
    layout_threshold_nodes: int = 1_000

    # --- media -----------------------------------------------------------
    uploads_dir: Path = Path("uploads")
    media_max_bytes: int = 100 * 1024 * 1024
    #: Uploaded media is imagery of real people. Indefinite retention in a
    #: research project is not defensible; the purge beat task enforces this.
    media_retention_hours: int = 24

    # --- llm (compass + summarize) --------------------------------------
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-4-5-20251001"
    compass_prompt_version: str = "v1"
    compass_max_retries: int = 2

    # --- reports ---------------------------------------------------------
    reports_dir: Path = Path("reports")

    # --- validators ------------------------------------------------------
    @field_validator(
        "api_key_pepper",
        "bootstrap_admin_key",
        "anthropic_api_key",
        mode="before",
    )
    @classmethod
    def _blank_secret_is_absent(cls, v: Any) -> Any:
        """``KEY=`` in .env means "not set", matching Phase 1's convention."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator(
        "database_url",
        "database_url_sync",
        "redis_url",
        "celery_broker_url",
        "celery_result_backend",
        mode="before",
    )
    @classmethod
    def _blank_url_is_empty(cls, v: Any) -> Any:
        """A blank URL stays an empty string, not None.

        These are checked at the point of use (``require_database_url``), not at
        import, so the API can boot far enough to serve /healthz and *say* what
        is missing rather than crashing in the container's first millisecond.
        """
        if v is None:
            return ""
        return v.strip() if isinstance(v, str) else v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        """Accept CSV (the documented form) and a JSON array (the reflex form).

        CSV is what ``.env.example`` documents and what 12-factor env vars
        normally look like. JSON is handled too because ``NoDecode`` took away
        the source-level JSON parsing, and without this branch
        ``CORS_ORIGINS='["http://a","http://b"]'`` would split on the comma and
        yield ``('["http://a"', '"http://b"]')`` -- junk, accepted silently,
        which is the same class of bug ``NoDecode`` was added to fix.
        """
        if not isinstance(v, str):
            return v
        text = v.strip()
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    f"CORS_ORIGINS looks like JSON but does not parse: {exc}. "
                    "Use a comma-separated list, e.g. "
                    "CORS_ORIGINS=http://localhost:5173,http://localhost:3000"
                ) from exc
            return tuple(str(part).strip() for part in parsed if str(part).strip())
        return tuple(part.strip() for part in text.split(",") if part.strip())

    @field_validator("data_dir", "models_dir", "uploads_dir", "reports_dir", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    @field_validator("database_url_sync", mode="after")
    @classmethod
    def _derive_sync_url(cls, v: str, info: ValidationInfo) -> str:
        if v:
            return v
        url = info.data.get("database_url") or ""
        # psycopg 3 is both sync and async under one driver name, so the async
        # and sync URLs are identical. Kept as a separate field anyway because
        # a deployment that switches to asyncpg needs them to diverge.
        return url

    @field_validator("celery_broker_url", "celery_result_backend", mode="after")
    @classmethod
    def _default_to_redis(cls, v: str, info: ValidationInfo) -> str:
        return v or (info.data.get("redis_url") or "")

    # --- derived ---------------------------------------------------------
    @property
    def normalized_dir(self) -> Path:
        """Phase 1's corpus. Read-only to Phase 4."""
        return get_ingest_settings().normalized_dir

    @property
    def authors_dir(self) -> Path:
        return self.data_dir / "authors"

    @property
    def scored_dir(self) -> Path:
        """Phase 2/3's scored tables. Read-only to Phase 4."""
        return self.data_dir / "scored"

    @property
    def rejects_dir(self) -> Path:
        return self.data_dir / "rejects"

    @property
    def manifest_path(self) -> Path:
        return get_ingest_settings().manifest_path

    @property
    def embedding_table(self) -> str:
        """One table per dimension. See app/models/embeddings.py for why."""
        return f"post_embeddings_{self.embedding_dim}"

    def require_database_url(self) -> str:
        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL is unset. Copy .env.example to .env and fill it in; "
                "there is deliberately no localhost default."
            )
        return self.database_url

    def require_redis_url(self) -> str:
        if not self.redis_url:
            raise RuntimeError("REDIS_URL is unset. See .env.example.")
        return self.redis_url


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return ApiSettings()


def load_yaml(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``. Returns ``{}`` if absent."""
    import yaml

    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
