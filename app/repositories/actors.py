"""Actors queries.

Placeholder until build step 9. Every function raises the same 501 so an unwired route
fails with a documented, machine-readable code instead of an ImportError that
reads like a crash. The contract for these routes is already published and
served under DEMO_MODE; only the query is missing.
"""

from __future__ import annotations

from typing import Any

from app.errors import NotImplementedYet


def _pending(name: str) -> NotImplementedYet:
    return NotImplementedYet(
        f"{name} is not wired to real data yet; it lands at build step 9. "
        "Set DEMO_MODE=1 to develop against the fixture data.",
        detail={"function": name, "lands_at": "build step 9"},
    )


async def list_authors(*args: Any, **kwargs: Any):
    raise _pending("actors.list_authors")


async def get_author(*args: Any, **kwargs: Any):
    raise _pending("actors.get_author")


async def author_score(*args: Any, **kwargs: Any):
    raise _pending("actors.author_score")


async def author_timeline(*args: Any, **kwargs: Any):
    raise _pending("actors.author_timeline")


async def author_posts(*args: Any, **kwargs: Any):
    raise _pending("actors.author_posts")


async def list_cohorts(*args: Any, **kwargs: Any):
    raise _pending("actors.list_cohorts")


async def cohort_authors(*args: Any, **kwargs: Any):
    raise _pending("actors.cohort_authors")


async def list_author_groups(*args: Any, **kwargs: Any):
    raise _pending("actors.list_author_groups")


async def create_author_group(*args: Any, **kwargs: Any):
    raise _pending("actors.create_author_group")


async def add_group_members(*args: Any, **kwargs: Any):
    raise _pending("actors.add_group_members")
