"""Redis handles.

Two clients for the same server: an async one for the request path (rate
limiting, job status reads) and a sync one for the workers. They are separate
because mixing an async client into a prefork Celery child means every task
either spins up an event loop or blocks one, and neither is worth it for a
GETSET.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_api_settings


@lru_cache(maxsize=1)
def get_redis() -> Any:
    """Async client for FastAPI."""
    import redis.asyncio as aioredis

    settings = get_api_settings()
    return aioredis.from_url(
        settings.require_redis_url(),
        decode_responses=True,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )


@lru_cache(maxsize=1)
def get_sync_redis() -> Any:
    """Blocking client for Celery tasks."""
    import redis

    settings = get_api_settings()
    return redis.from_url(
        settings.require_redis_url(),
        decode_responses=True,
        socket_connect_timeout=5,
    )
