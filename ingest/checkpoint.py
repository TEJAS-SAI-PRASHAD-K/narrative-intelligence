"""Resumable cursors and the YouTube quota ledger.

A run that dies forty minutes in must resume, not restart. Two pieces make that
true: the store deduplicates on ``id``, and every adapter records where it got
to here. Checkpoints are written atomically after each page, because the
failure this guards against is precisely a process that dies mid-write.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Checkpoint:
    """Per-source cursor state, one JSON file on disk.

    The shape is free-form on purpose: a Mastodon backfill needs ``max_id`` per
    timeline, GDELT needs the last 15-minute file it consumed, YouTube needs a
    page token per query. Forcing one cursor shape on all of them would just
    push the mess into the adapters.
    """

    def __init__(self, source: str, directory: Path):
        self.source = source
        self.path = Path(directory) / f"{source}.json"
        self._state: dict[str, Any] = {}
        if self.path.exists():
            try:
                self._state = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("checkpoint %s is corrupt; starting from scratch", self.path)
                self._state = {}

    # --- cursor ----------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any, *, flush: bool = True) -> None:
        self._state[key] = value
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        if flush:
            self.save()

    def update(self, **values: Any) -> None:
        self._state.update(values)
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def bump(self, key: str, amount: int = 1) -> int:
        value = int(self._state.get(key, 0)) + amount
        self.set(key, value)
        return value

    def mark_done(self, unit: str) -> None:
        """Record a completed unit of work (a subreddit, a feed, a 15-min file)."""
        done = set(self._state.get("completed", []))
        done.add(unit)
        self.set("completed", sorted(done))

    def is_done(self, unit: str) -> bool:
        return unit in set(self._state.get("completed", []))

    def clear(self) -> None:
        self._state = {}
        self.path.unlink(missing_ok=True)

    def save(self) -> None:
        _atomic_write_json(self.path, self._state)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._state)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Checkpoint(source={self.source!r}, keys={sorted(self._state)})"


class QuotaExhausted(RuntimeError):
    """Raised when a day's API budget is spent. Caught by the adapter, not fatal."""


class QuotaLedger:
    """Units spent per UTC day, with a hard stop before exhaustion.

    YouTube gives 10,000 units per UTC day. ``search.list`` costs 100 of them,
    ``videos.list`` and ``commentThreads.list`` cost 1. Nothing wastes a day
    faster than discovering at 11am that a loop spent the whole budget on
    searches, so the budget is enforced before the call, not after.
    """

    def __init__(self, checkpoint: Checkpoint, daily_limit: int, key: str = "quota"):
        self.checkpoint = checkpoint
        self.daily_limit = int(daily_limit)
        self.key = key

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ledger(self) -> dict[str, Any]:
        ledger = self.checkpoint.get(self.key) or {}
        today = self._today()
        if ledger.get("date") != today:
            # New UTC day: the quota resets, and so does the ledger.
            ledger = {"date": today, "units": 0, "calls": {}}
        return ledger

    @property
    def spent(self) -> int:
        return int(self._ledger().get("units", 0))

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self.spent)

    def can_afford(self, units: int) -> bool:
        return units <= self.remaining

    def charge(self, units: int, *, call: str = "unknown") -> int:
        """Spend ``units``, or raise :class:`QuotaExhausted` before spending them."""
        ledger = self._ledger()
        if units > self.daily_limit - int(ledger.get("units", 0)):
            raise QuotaExhausted(
                f"{call} needs {units} units; only {self.remaining} of "
                f"{self.daily_limit} remain for {ledger['date']} (UTC). "
                "Stopping cleanly so the checkpoint stays resumable tomorrow."
            )
        ledger["units"] = int(ledger.get("units", 0)) + units
        calls = ledger.setdefault("calls", {})
        calls[call] = int(calls.get(call, 0)) + 1
        self.checkpoint.set(self.key, ledger)
        log.debug("quota: +%d for %s (%d/%d today)", units, call, ledger["units"], self.daily_limit)
        return ledger["units"]

    def count(self, call: str) -> int:
        """How many times a given call has been made today."""
        return int(self._ledger().get("calls", {}).get(call, 0))

    def summary(self) -> str:
        ledger = self._ledger()
        return (
            f"quota {ledger.get('units', 0)}/{self.daily_limit} units spent on "
            f"{ledger.get('date')} (UTC); calls={ledger.get('calls', {})}"
        )
