"""Route modules, one per group in the API surface.

``ROUTERS`` is the ordered list mounted under ``/api/v1``. The order sets the
tag order on the docs page, and it is deliberately the order an analyst works
in: pick a project, look at the overview, drill into narratives, then actors,
then drivers, then domains, then the network, then export.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.routers import (
    actors,
    alerts,
    domains,
    drivers,
    ingest,
    jobs,
    keys,
    media,
    narratives,
    network,
    overview,
    posts,
    projects,
    reports,
)

ROUTERS: list[APIRouter] = [
    projects.router,
    ingest.router,
    overview.router,
    narratives.router,
    narratives.compass_router,
    actors.router,
    drivers.router,
    domains.router,
    network.router,
    posts.router,
    media.router,
    alerts.router,
    reports.router,
    jobs.router,
    keys.router,
]
