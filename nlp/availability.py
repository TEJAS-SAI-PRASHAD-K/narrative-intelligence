"""Which model capabilities this deployment actually has.

Called by /readyz on every probe and by the task layer before it tries to run
inference. Deliberately cheap and import-light: it stats the filesystem and
reads ``configs/models.yaml``. It never loads a model, so the API image can
answer honestly about a capability it could not itself execute.

The rule the whole service follows: **a missing checkpoint is a missing
capability, not a broken service.** The stack boots, /healthz is 200, /readyz
reports the capability as `unavailable`, and only the routes that need it return
503 with the checkpoint path in the message.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from nlp.interfaces import CAPABILITIES, Capability, ModelInfo

log = logging.getLogger(__name__)

#: Capability -> the ``configs/models.yaml`` module key that describes it.
#: Where a capability has no checkpoint of its own (aux scorers pull pretrained
#: weights from the HF cache; compass needs an API key, not a file) the mapping
#: is None and availability is decided by a different probe below.
_MODULE_KEYS: dict[str, str | None] = {
    "embed": "embed",
    "cluster": "cluster",
    "misinfo": "misinfo",
    "stance": "stance",
    "aux": None,
    "bot": "bot",
    "deepfake": "deepfake",
    "summarize": "summarize",
    "compass": None,
}

#: Capabilities that have a Parquet fallback under ``data/scored/``. Phase 2/3
#: already ran these over the corpus and committed the outputs, so Phase 4 can
#: serve real scores with no checkpoint mounted at all. That is a *degraded*
#: state, not a ready one: it can score the existing corpus but not a new post.
_SCORED_TABLE: dict[str, str] = {
    "misinfo": "record_scores",
    "stance": "record_scores",
    "aux": "record_scores",
    "bot": "author_scores",
    "deepfake": "media_scores",
    "cluster": "narratives",
    "summarize": "narratives",
}


def _models_config() -> dict[str, Any]:
    from app.config import load_yaml

    return load_yaml("models")


def _checkpoint_dir(models_dir: Path, module: str, version: str) -> Path:
    """Mirror ``modeling.registry.local_dir`` without importing it.

    Importing modeling/ here would drag torch into the API image. The layout is
    ``models/<module>/<version>/`` and it is asserted by a test so this copy
    cannot drift from the original silently.
    """
    return models_dir / module / version


def _has_weights(path: Path) -> bool:
    """A directory with no weight file in it is a failed download, not a model."""
    if not path.is_dir():
        return False
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.json", "*.joblib", "*.pkl", "*.ubj")
    return any(next(path.rglob(pattern), None) is not None for pattern in patterns)


def _embedding_cache_rows(embeddings_dir: Path) -> int | None:
    """How many vectors Phase 2 left on disk.

    The key index is a JSON list of record ids, so its length is the row count
    and reading it costs nothing compared with memory-mapping the matrix.
    """
    if not embeddings_dir.exists():
        return None
    import json

    total = 0
    for keys_file in embeddings_dir.glob("*.keys.json"):
        try:
            total += len(json.loads(keys_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read embedding key index %s: %s", keys_file, exc)
    return total or None


def _scored_rows(scored_dir: Path, table: str) -> int | None:
    """Row count of a Phase 2/3 scored table, or None if it is absent.

    Reads the Parquet footers only -- metadata, not data -- so this stays cheap
    enough to run on every readiness probe.
    """
    path = scored_dir / table
    if not path.exists():
        return None
    try:
        import pyarrow.dataset as ds

        return ds.dataset(str(path), format="parquet", partitioning="hive").count_rows()
    except Exception as exc:  # pragma: no cover - corrupt parquet is not a probe failure
        log.warning("could not count scored table %s: %s", table, exc)
        return None


#: How long a capability probe stays fresh. Checkpoints do not appear and
#: disappear second to second, and /readyz is polled by an orchestrator on a
#: short interval: re-stat'ing the filesystem and reading Parquet footers on
#: every probe made readiness a 550ms call, which is slow enough to trip a
#: liveness timeout and restart a perfectly healthy container.
PROBE_TTL_SECONDS = 30.0

_last_probe_at: float = 0.0


def probe(*, refresh: bool = False) -> list[Capability]:
    """Report every capability. Never raises.

    ``refresh=True`` respects the TTL rather than forcing a re-scan. A caller
    that genuinely needs to see a checkpoint mounted seconds ago can clear the
    cache directly; nothing in the request path needs that.
    """
    global _last_probe_at
    import time

    now = time.monotonic()
    if refresh and now - _last_probe_at >= PROBE_TTL_SECONDS:
        _probe_cached.cache_clear()
        _last_probe_at = now
    elif not _last_probe_at:
        _last_probe_at = now
    return list(_probe_cached())


@lru_cache(maxsize=1)
def _probe_cached() -> tuple[Capability, ...]:
    from app.config import get_api_settings

    settings = get_api_settings()
    config = _models_config()
    now = datetime.now(timezone.utc)
    out: list[Capability] = []

    for name in CAPABILITIES:
        module_key = _MODULE_KEYS.get(name)
        module_cfg = (config.get(module_key) or {}) if module_key else {}
        version = str(module_cfg.get("version") or "v0.1.0")
        model_name = module_cfg.get("model_name") or module_cfg.get("base_model")

        info = ModelInfo(
            name=str(model_name or name),
            version=version,
            dim=settings.embedding_dim if name == "embed" else None,
            loaded_at=None,
        )

        # 1. A mounted checkpoint is the ready state.
        if module_key:
            ckpt = _checkpoint_dir(settings.models_dir, module_key, version)
            if _has_weights(ckpt):
                out.append(
                    Capability(
                        name=name,
                        available=True,
                        status="ready",
                        detail=f"checkpoint at {ckpt}",
                        model=ModelInfo(
                            name=info.name,
                            version=version,
                            checkpoint_uri=f"file://{ckpt}",
                            dim=info.dim,
                            loaded_at=now,
                        ),
                    )
                )
                continue

        # 2. Compass and summarize need an API key, not a file on disk.
        if name in {"compass", "summarize"} and not settings.anthropic_api_key:
            out.append(
                Capability(
                    name=name,
                    available=False,
                    status="unavailable",
                    detail=(
                        "ANTHROPIC_API_KEY is unset. Narrative labels fall back to "
                        "centroid keywords; Compass Context cannot be generated."
                    ),
                    model=info,
                )
            )
            continue
        if name in {"compass", "summarize"}:
            out.append(
                Capability(
                    name=name,
                    available=True,
                    status="ready",
                    detail=f"LLM configured: {settings.llm_model}",
                    model=ModelInfo(name=settings.llm_model, version=version, loaded_at=now),
                )
            )
            continue

        # 3. Phase 2/3 already scored the corpus into Parquet. That serves every
        #    existing row but cannot score a new one, so it is degraded.
        table = _SCORED_TABLE.get(name)
        rows = _scored_rows(settings.scored_dir, table) if table else None
        if rows:
            out.append(
                Capability(
                    name=name,
                    available=True,
                    status="degraded",
                    detail=(
                        f"no checkpoint mounted; serving {rows} precomputed rows from "
                        f"data/scored/{table}. New posts cannot be scored until a "
                        f"checkpoint is mounted at {settings.models_dir}/{module_key}/{version}."
                    ),
                    model=info,
                )
            )
            continue

        # 4. Embeddings have their own Phase 2 artifact shape: a .npy matrix plus
        #    a key index, not a scored Parquet table.
        if name == "embed":
            cached = _embedding_cache_rows(settings.data_dir / "embeddings")
            if cached:
                out.append(
                    Capability(
                        name=name,
                        available=True,
                        status="degraded",
                        detail=(
                            f"no checkpoint mounted; {cached} precomputed vectors available in "
                            f"data/embeddings. New text cannot be embedded, so /posts/similar "
                            f"accepts post_id but not raw text."
                        ),
                        model=info,
                    )
                )
                continue

        where = (
            f"no checkpoint at {settings.models_dir}/{module_key}/{version}"
            if module_key
            else "no checkpoint configured"
        )
        fallback = (
            f" and no precomputed rows in data/scored/{table}"
            if table
            else " and no precomputed fallback"
        )
        out.append(
            Capability(
                name=name,
                available=False,
                status="unavailable",
                detail=where + fallback + ".",
                model=info,
            )
        )

    return tuple(out)


def get(name: str) -> Capability:
    for capability in probe():
        if capability.name == name:
            return capability
    raise KeyError(f"unknown capability {name!r}; expected one of {CAPABILITIES}")


def require(name: str) -> Capability:
    """Raise the 503 for a route that cannot run without this capability."""
    from app.errors import ScorerUnavailable

    capability = get(name)
    if not capability.available:
        raise ScorerUnavailable(
            f"The '{name}' model is unavailable: {capability.detail}",
            detail={"capability": name, "status": capability.status},
        )
    return capability
