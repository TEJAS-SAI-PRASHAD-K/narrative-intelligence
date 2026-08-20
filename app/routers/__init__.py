"""Route modules, one per group in the API surface.

``ROUTERS`` is the ordered list mounted under ``/api/v1``. Order matters only
for the OpenAPI tag ordering the docs page renders, which is deliberately the
order an analyst works in: pick a project, look at the overview, drill into
narratives, then actors, then the network, then export.
"""

from __future__ import annotations

from fastapi import APIRouter

ROUTERS: list[APIRouter] = []
