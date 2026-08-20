"""Liveness, readiness, jobs and metrics.

``/healthz`` and ``/readyz`` live outside the versioned prefix: an orchestrator
probing a container should not have to know the API version.

The distinction between the two matters operationally and is not cosmetic:

* **/healthz** answers "is this process alive". It touches nothing external and
  is 200 as soon as the event loop is running. If it ever depends on Postgres, a
  database blip restarts every API container and turns a degraded read path into
  a full outage.
* **/readyz** answers "should this container receive traffic", and reports the
  database, Redis and each model checkpoint. A missing checkpoint makes it
  ``degraded``, not ``down``: the narrative API works fine without a deepfake
  model, and taking the container out of rotation for that would be wrong.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Response, status

from app.config import API_VERSION, get_api_settings
from app.schemas.ops import HealthResponse, ReadinessResponse, SubsystemStatus

log = logging.getLogger(__name__)

probe_router = APIRouter(tags=["ops"])

_STARTED_AT = time.time()


@probe_router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="200 while the process is alive. Touches no external service by design.",
)
async def healthz() -> HealthResponse:
    settings = get_api_settings()
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        environment=settings.environment,
        demo_mode=settings.demo_mode,
        uptime_seconds=round(time.time() - _STARTED_AT, 1),
    )


async def _check_database() -> SubsystemStatus:
    from sqlalchemy import text

    from app.db import get_sessionmaker

    started = time.perf_counter()
    try:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            await session.execute(text("SELECT 1"))
            has_vector = (
                await session.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            ).scalar()
            # to_regclass rather than a bare SELECT: an unmigrated database is
            # a legitimate state on a cold start, and an UndefinedTable error
            # here would report the whole database as down while the API
            # container is still running `alembic upgrade head`.
            migrated = (
                await session.execute(text("SELECT to_regclass('alembic_version')"))
            ).scalar()
            revision = None
            if migrated:
                revision = (
                    await session.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
                ).scalar()
        detail = f"pgvector={'yes' if has_vector else 'MISSING'} migration={revision or 'none'}"
        # A reachable database with no pgvector will fail the first kNN query,
        # and one with no schema cannot serve a single route. Both are reachable
        # but not ready, which is exactly what `degraded` means here.
        healthy = bool(has_vector and revision)
        return SubsystemStatus(
            name="database",
            status="up" if healthy else "degraded",
            detail=detail,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        return SubsystemStatus(
            name="database",
            status="down",
            detail=f"{type(exc).__name__}: {exc}"[:400],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


async def _check_redis() -> SubsystemStatus:
    started = time.perf_counter()
    try:
        from app.redis_client import get_redis

        client = get_redis()
        await client.ping()
        return SubsystemStatus(
            name="redis",
            status="up",
            detail="ping ok",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        return SubsystemStatus(
            name="redis",
            status="down",
            detail=f"{type(exc).__name__}: {exc}"[:400],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


@probe_router.get(
    "/readyz",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Reports the database, Redis and every model capability. "
        "`degraded` means the service is usable but one capability is missing or "
        "running from precomputed scores; `down` means a hard dependency failed."
    ),
)
async def readyz(response: Response) -> ReadinessResponse:
    from nlp.availability import probe as probe_capabilities

    settings = get_api_settings()
    subsystems = [await _check_database(), await _check_redis()]
    capabilities = [c.as_dict() for c in probe_capabilities(refresh=True)]

    hard_down = [s for s in subsystems if s.status == "down"]
    if hard_down:
        overall = "down"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif any(s.status == "degraded" for s in subsystems) or any(
        not c["available"] for c in capabilities
    ):
        # Still 200. A degraded readiness that returns 503 takes the container
        # out of rotation for a missing deepfake checkpoint, which would make
        # the whole narrative API unreachable for no reason.
        overall = "degraded"
    else:
        overall = "ok"

    return ReadinessResponse(
        status=overall,
        version=API_VERSION,
        environment=settings.environment,
        demo_mode=settings.demo_mode,
        subsystems=subsystems,
        capabilities=capabilities,
    )


@probe_router.get(
    "/metrics",
    summary="Prometheus metrics",
    include_in_schema=False,
    response_class=Response,
)
async def metrics() -> Response:
    """Optional and unauthenticated, so it stays scrapeable inside the compose
    network. It exposes counters only -- never corpus content."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    except ImportError:
        return Response(
            content="# prometheus_client is not installed; metrics are disabled\n",
            media_type="text/plain",
        )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
