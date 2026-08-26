"""Redis handles.

Two clients for the same server: an async one for the request path (rate
limiting, job status reads) and a sync one for the workers. They are separate
because mixing an async client into a prefork Celery child means every task
either spins up an event loop or blocks one, and neither is worth it for a
GETSET.

The async client is cached **per event loop**, not globally. An asyncio
connection pool binds its connections to the loop that created them, so a single
`lru_cache` handing the same pool to a second loop produces
``RuntimeError: Event loop is closed`` on the first call. Under uvicorn there is
one loop and the bug never appears; under a test client, a script, or any process
that replaces the loop, the cached client is permanently broken -- and because
/readyz uses it, that surfaces as a readiness probe that flaps between `ok` and
`down` and restarts a perfectly healthy container.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any
from weakref import WeakKeyDictionary

from app.config import get_api_settings

log = logging.getLogger(__name__)

#: loop -> client. Weak-keyed so a finished loop's client is collected with it
#: rather than pinning a dead pool for the life of the process.
_async_clients: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


def get_redis() -> Any:
    """Async client bound to the running event loop."""
    import redis.asyncio as aioredis

    settings = get_api_settings()
    url = settings.require_redis_url()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop yet: the caller will await this later on whichever loop ends
        # up running. An uncached client is correct here and costs one
        # connection setup.
        return aioredis.from_url(url, decode_responses=True)

    client = _async_clients.get(loop)
    if client is None:
        client = aioredis.from_url(
            url,
            decode_responses=True,
            health_check_interval=30,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        _async_clients[loop] = client
    return client


@lru_cache(maxsize=1)
def get_sync_redis() -> Any:
    """Blocking client for Celery tasks.

    Safe to cache globally: there is no event loop involved, and a prefork child
    builds its own after the fork.
    """
    import redis

    settings = get_api_settings()
    return redis.from_url(
        settings.require_redis_url(),
        decode_responses=True,
        socket_connect_timeout=5,
    )


def reset_async_clients() -> None:
    """Drop every cached async client. Used by the test fixtures."""
    _async_clients.clear()
