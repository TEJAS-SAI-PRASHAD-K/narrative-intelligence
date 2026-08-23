"""Reports queries.

Placeholder until build step 13. Every function raises the same 501 so an unwired route
fails with a documented, machine-readable code instead of an ImportError that
reads like a crash. The contract for these routes is already published and
served under DEMO_MODE; only the query is missing.
"""

from __future__ import annotations

from typing import Any

from app.errors import NotImplementedYet


def _pending(name: str) -> NotImplementedYet:
    return NotImplementedYet(
        f"{name} is not wired to real data yet; it lands at build step 13. "
        "Set DEMO_MODE=1 to develop against the fixture data.",
        detail={"function": name, "lands_at": "build step 13"},
    )


async def list_reports(*args: Any, **kwargs: Any):
    raise _pending("reports.list_reports")


async def report_file(*args: Any, **kwargs: Any):
    raise _pending("reports.report_file")
