"""DEMO_MODE fixtures backing every route.

Purpose: Phase 5 starts building the dashboard on day one, against the real
OpenAPI schema, before a single row is loaded. Everything after the contract tag
changes *values*, never *shapes*.

Two properties this package must have, and both are load-bearing:

* **Deterministic.** One seed, no clock, no randomness at request time. A
  frontend developer's screenshot has to match the next person's, and a
  snapshot test against these fixtures has to be stable.
* **Referentially coherent.** The narrative ids on a mock post exist as mock
  narratives; a mock author belongs to mock cohorts that appear in /cohorts.
  Fixtures that are individually valid but mutually inconsistent are worse than
  none: the frontend builds navigation that works against them and breaks the
  day real data lands.
"""

from __future__ import annotations

from app.mock.corpus import corpus, demo_project_id, demo_project_slug

__all__ = ["corpus", "demo_project_id", "demo_project_slug"]
