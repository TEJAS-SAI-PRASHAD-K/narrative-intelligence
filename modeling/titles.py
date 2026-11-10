"""The title of a corpus record, read the one way every consumer reads it.

Platform-specific fields live in ``raw``, and the title is the ``title`` key of that
blob. The comparison scores titles as ``corpus_title``, and the ``laya`` scoring stage
scores the same value, so both go through :func:`title_of`.
"""

from __future__ import annotations

import json
from typing import Any


def title_of(raw: Any) -> str | None:
    """The stripped ``title`` of a ``raw`` blob (a dict or a JSON string), or ``None``."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    title = raw.get("title") if isinstance(raw, dict) else None
    title = str(title).strip() if title is not None else ""
    return title or None
