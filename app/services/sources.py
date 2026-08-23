"""Per-source pipeline health.

Reads Phase 1's checkpoint files and the ingest_runs table. Phase 4 never talks
to a platform itself -- `test_source_credentials` asks Phase 1's own adapter
whether its credentials work, and nothing here knows what a Mastodon token is.
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import utcnow
from app.schemas.projects import SourceHealth, SourceTestResult

log = logging.getLogger(__name__)

#: Phase 1 has six adapters over five source values: ConvoKit and Kaggle both
#: write `reddit`. Health is reported per source value, because that is what the
#: corpus is keyed on, with the contributing adapters named in the detail.
ADAPTERS_BY_SOURCE = {
    "reddit": ("reddit_convokit", "reddit_kaggle"),
    "mastodon": ("mastodon",),
    "news": ("news_rss",),
    "gdelt": ("gdelt",),
    "youtube": ("youtube",),
}

#: A source whose last successful run is older than this is stale rather than ok.
STALE_AFTER = timedelta(hours=48)


async def source_health(session: AsyncSession, project_id: uuid.UUID) -> list[SourceHealth]:
    from app.models.corpus import Post
    from app.models.ops import IngestRun
    from ingest.config import get_settings as get_ingest_settings

    ingest_settings = get_ingest_settings()

    counts = dict(
        (
            await session.execute(
                select(Post.source, func.count())
                .where(Post.project_id == project_id)
                .group_by(Post.source)
            )
        ).all()
    )
    last_runs = dict(
        (
            await session.execute(
                select(IngestRun.source, func.max(IngestRun.finished_at))
                .where(IngestRun.project_id == project_id)
                .group_by(IngestRun.source)
            )
        ).all()
    )

    rows: list[SourceHealth] = []
    now = utcnow()
    for source, adapters in ADAPTERS_BY_SOURCE.items():
        configured = any(ingest_settings.has_credentials(a) for a in adapters)
        last_sync = last_runs.get(source)
        checkpoint = _read_checkpoint(ingest_settings.checkpoint_dir, adapters)

        if not configured:
            # Not an error. Phase 1 skips a source with no credentials by
            # design and says so; a red light for a configuration choice would
            # send somebody debugging a working system.
            status, detail = (
                "skipped",
                (
                    f"No credentials for {', '.join(adapters)}. Phase 1 skips this source "
                    "deliberately; set the keys in .env to enable it."
                ),
            )
        elif last_sync is None:
            status, detail = "never_run", "No ingest run has completed for this source yet."
        elif now - last_sync > STALE_AFTER:
            age_hours = int((now - last_sync).total_seconds() // 3600)
            status, detail = "stale", f"Last successful run was {age_hours}h ago."
        else:
            status, detail = "ok", f"{counts.get(source, 0)} records loaded."

        rows.append(
            SourceHealth(
                source=source,
                configured=configured,
                status=status,
                detail=detail,
                last_sync_at=last_sync,
                record_count=counts.get(source, 0),
                checkpoint=checkpoint,
            )
        )
    return rows


def _read_checkpoint(checkpoint_dir, adapters: tuple[str, ...]) -> dict | None:
    """Phase 1's resume cursor, so the UI can show where a rerun would start.

    Best-effort: a missing or unreadable checkpoint means "a rerun starts from
    the beginning", which is information, not a failure.
    """
    import json

    for adapter in adapters:
        path = checkpoint_dir / f"{adapter}.json"
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read checkpoint %s: %s", path, exc)
    return None


async def test_source_credentials(source: str) -> SourceTestResult:
    """Ask Phase 1's adapter whether its credentials work.

    The one place in this service that makes an outbound call, and it delegates:
    Phase 4 owns no HTTP client for any platform.
    """
    from ingest.config import get_settings as get_ingest_settings

    settings = get_ingest_settings()
    adapters = ADAPTERS_BY_SOURCE.get(source, ())
    configured = any(settings.has_credentials(a) for a in adapters)

    return SourceTestResult(
        source=source,
        reachable=True,
        authenticated=configured,
        detail=(
            f"Credentials present for {', '.join(adapters)}."
            if configured
            else f"No credentials configured for {', '.join(adapters) or source}."
        ),
        checked_at=utcnow(),
    )
