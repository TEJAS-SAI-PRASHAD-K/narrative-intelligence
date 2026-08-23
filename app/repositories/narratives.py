"""Narratives queries.

Placeholder until build step 8. Every function raises the same 501 so an unwired route
fails with a documented, machine-readable code instead of an ImportError that
reads like a crash. The contract for these routes is already published and
served under DEMO_MODE; only the query is missing.
"""

from __future__ import annotations

from typing import Any

from app.errors import NotImplementedYet


def _pending(name: str) -> NotImplementedYet:
    return NotImplementedYet(
        f"{name} is not wired to real data yet; it lands at build step 8. "
        "Set DEMO_MODE=1 to develop against the fixture data.",
        detail={"function": name, "lands_at": "build step 8"},
    )


async def list_narratives(*args: Any, **kwargs: Any):
    raise _pending("narratives.list_narratives")


async def get_narrative(*args: Any, **kwargs: Any):
    raise _pending("narratives.get_narrative")


async def create_manual_narrative(*args: Any, **kwargs: Any):
    raise _pending("narratives.create_manual_narrative")


async def update_narrative(*args: Any, **kwargs: Any):
    raise _pending("narratives.update_narrative")


async def narrative_timeline(*args: Any, **kwargs: Any):
    raise _pending("narratives.narrative_timeline")


async def narrative_posts(*args: Any, **kwargs: Any):
    raise _pending("narratives.narrative_posts")


async def narrative_authors(*args: Any, **kwargs: Any):
    raise _pending("narratives.narrative_authors")


async def narrative_cohorts(*args: Any, **kwargs: Any):
    raise _pending("narratives.narrative_cohorts")


async def narrative_score(*args: Any, **kwargs: Any):
    raise _pending("narratives.narrative_score")


async def clustering_freshness(*args: Any, **kwargs: Any):
    raise _pending("narratives.clustering_freshness")
