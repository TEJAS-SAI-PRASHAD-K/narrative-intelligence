"""FastAPI application factory.

The app is thin by design: it validates, queries and enqueues. It never
computes. Anything that would take more than roughly 500ms of CPU inside a
request is a design bug and belongs in a Celery task -- see app/tasks/.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_VERSION, get_api_settings

log = logging.getLogger("app")

API_PREFIX = f"/api/{API_VERSION}"

DESCRIPTION = """
Narrative Intelligence Platform -- Phase 4 backend.

Detects, clusters and scores coordinated misinformation narratives across
platforms. Every score this API returns is decomposable: a `components` blob and
a `scoring_version` accompany every number, and every AI-generated string
carries `generated_by`, the model id and a generation timestamp.

**Conventions**

* Auth: `X-API-Key` header. Reads need the `read` scope, mutations `write`,
  key management `admin`.
* Lists are cursor-paginated: `?cursor=&limit=` (default 50, max 200).
* Anything expensive returns `202` with a `job_id`; poll `/api/v1/jobs/{id}`.
* Errors share one envelope: `{"error": {"code", "message", "detail", "request_id"}}`.
"""


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "botocore", "asyncio", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown.

    Startup deliberately does *not* fail on a missing checkpoint or an
    unreachable model. A missing deepfake checkpoint must not stop the narrative
    API from serving; it marks that scorer `unavailable` in /readyz and the
    routes that need it return 503 with a message naming the checkpoint.
    """
    settings = get_api_settings()
    configure_logging(settings.log_level)
    log.info(
        "starting api environment=%s demo_mode=%s embedding=%s/%d",
        settings.environment,
        settings.demo_mode,
        settings.embedding_model,
        settings.embedding_dim,
    )
    yield
    log.info("api shutdown complete")


def create_app() -> FastAPI:
    settings = get_api_settings()

    app = FastAPI(
        title="Narrative Intelligence Platform API",
        description=DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,  # the API is key-authenticated, not cookie-authenticated
        allow_methods=["*"],
        allow_headers=["X-API-Key", "Content-Type", "Accept"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Stamp a request id and log the timing.

        The request id lands in every error envelope, so a screenshot of a
        failed UI call is enough to find the exact log line.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            log.exception(
                "unhandled %s %s request_id=%s elapsed_ms=%.1f",
                request.method,
                request.url.path,
                request_id,
                elapsed,
            )
            raise
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        # Slow requests are a design bug, not a perf nit: log them at WARNING so
        # they are impossible to miss during development.
        level = logging.WARNING if elapsed > 500 else logging.INFO
        log.log(
            level,
            "%s %s -> %s request_id=%s elapsed_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
            elapsed,
        )
        return response

    register_exception_handlers(app)
    register_routers(app)
    return app


def register_exception_handlers(app: FastAPI) -> None:
    from app.errors import install_handlers

    install_handlers(app)


def register_routers(app: FastAPI) -> None:
    from app.routers import ROUTERS, ops

    # Liveness and readiness sit outside the versioned prefix: an orchestrator
    # probing /healthz should not have to know the API version.
    app.include_router(ops.probe_router)
    for router in ROUTERS:
        app.include_router(router, prefix=API_PREFIX)


app = create_app()
