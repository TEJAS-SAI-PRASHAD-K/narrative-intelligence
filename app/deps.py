"""Shared FastAPI dependencies: auth, scopes, rate limiting, pagination and the
one filter specification the whole API uses.

The filter dependency is the important one architecturally. The UI has a single
sticky filter bar that applies on nearly every page, so every list endpoint
accepts the same parameters. Defining them once here and applying them once in
``app/repositories/filters.py`` is what stops this codebase rotting: the
alternative -- each router re-declaring `date_from` and re-deriving what
`include_shared` means -- guarantees that six months from now two endpoints
disagree about what "toxic" filters to, and nobody can tell which is right.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, Header, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_api_settings
from app.db import get_session
from app.errors import BadRequest, Forbidden, RateLimited, Unauthorized

log = logging.getLogger(__name__)

MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 50


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Only the key *id* is ever logged."""

    key_id: str
    name: str
    scopes: frozenset[str]

    def has(self, scope: str) -> bool:
        # `admin` implies the other two. Three scopes with an implication is
        # simpler to reason about than three independent flags that every key
        # ends up carrying all of anyway.
        return scope in self.scopes or "admin" in self.scopes


async def get_principal(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    """Resolve the X-API-Key header to a principal, or raise 401."""
    settings = get_api_settings()

    if settings.demo_mode:
        # The contract build has to be usable by a frontend developer with no
        # key. This is gated on DEMO_MODE, which is never 1 in production, and
        # the principal it returns is explicitly named so it cannot be mistaken
        # for a real one in a log.
        return Principal(key_id="demo", name="demo-mode", scopes=frozenset({"read", "write"}))

    if not x_api_key:
        raise Unauthorized("Missing X-API-Key header.")

    from app.models.ops import ApiKey
    from app.security import PREFIX_LEN, verify_key

    prefix = x_api_key[:PREFIX_LEN]
    rows = (
        (
            await session.execute(
                select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None))
            )
        )
        .scalars()
        .all()
    )

    for row in rows:
        if verify_key(x_api_key, row.key_hash):
            principal = Principal(
                key_id=str(row.id), name=row.name, scopes=frozenset(row.scopes or ())
            )
            request.state.principal = principal
            await _touch_last_used(session, row)
            return principal

    # One message for "no such prefix" and "wrong key": distinguishing them
    # tells an attacker which prefixes exist.
    log.warning(
        "auth failure prefix=%s path=%s request_id=%s",
        prefix,
        request.url.path,
        getattr(request.state, "request_id", None),
    )
    raise Unauthorized("Invalid API key.")


async def _touch_last_used(session: AsyncSession, row: Any) -> None:
    """Best-effort ``last_used_at``.

    Coarsened to one write per minute per key. Writing on every request would
    serialise the entire API behind a row lock, which is a real outage in
    exchange for a field nobody reads at second resolution.
    """
    now = datetime.now(timezone.utc)
    if row.last_used_at and (now - row.last_used_at).total_seconds() < 60:
        return
    try:
        row.last_used_at = now
        await session.commit()
    except Exception as exc:  # pragma: no cover - never fail a request over this
        log.debug("could not update last_used_at for key %s: %s", row.id, exc)
        await session.rollback()


def require_scope(scope: Literal["read", "write", "admin"]):
    """Dependency factory. Reads need `read`, mutations `write`, keys `admin`."""

    async def dependency(
        principal: Annotated[Principal, Depends(get_principal)],
    ) -> Principal:
        if not principal.has(scope):
            raise Forbidden(
                f"This API key has scopes {sorted(principal.scopes)} and needs '{scope}'.",
                detail={"required_scope": scope, "granted_scopes": sorted(principal.scopes)},
            )
        return principal

    return dependency


RequireRead = Annotated[Principal, Depends(require_scope("read"))]
RequireWrite = Annotated[Principal, Depends(require_scope("write"))]
RequireAdmin = Annotated[Principal, Depends(require_scope("admin"))]


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------
async def rate_limit(
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
) -> None:
    """Per-key token bucket in Redis.

    Fails **open**. A Redis outage should degrade rate limiting, not the API:
    refusing every request because the limiter is unreachable turns a
    nice-to-have into a hard dependency, which is the wrong trade for a
    protection against accidental over-polling by a known-good frontend.
    """
    settings = get_api_settings()
    if settings.demo_mode:
        return

    try:
        from app.redis_client import get_redis

        redis = get_redis()
        window = int(datetime.now(timezone.utc).timestamp() // 60)
        key = f"rl:{principal.key_id}:{window}"
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 120)
        used, _ = await pipe.execute()
    except Exception as exc:
        log.debug("rate limiter unavailable, failing open: %s", exc)
        return

    remaining = max(0, settings.rate_limit_per_minute - int(used))
    request.state.rate_limit_remaining = remaining
    if int(used) > settings.rate_limit_per_minute:
        raise RateLimited(
            f"Rate limit of {settings.rate_limit_per_minute} requests/minute exceeded.",
            detail={"limit_per_minute": settings.rate_limit_per_minute, "retry_after_s": 60},
        )


# ---------------------------------------------------------------------------
# pagination
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Page:
    """Cursor pagination.

    Cursor rather than offset because these tables are large and the UI scrolls:
    ``OFFSET 50000`` makes Postgres walk fifty thousand rows it will discard,
    and a row inserted mid-scroll shifts every subsequent page. An opaque cursor
    encoding the sort key has neither problem.
    """

    cursor: str | None
    limit: int


async def pagination(
    cursor: Annotated[
        str | None,
        Query(description="Opaque cursor from the previous response's `next_cursor`."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
) -> Page:
    return Page(cursor=cursor, limit=limit)


Pagination = Annotated[Page, Depends(pagination)]


# ---------------------------------------------------------------------------
# the shared filter specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterSpec:
    """Everything the UI's sticky filter bar can express, in one object.

    Two subtleties worth stating, because both have bitten this kind of system:

    * ``tz`` never changes *what* is selected -- the corpus is UTC end to end
      and every comparison happens in UTC. It changes only how time series are
      bucketed and how the response labels its interval, and it is echoed back
      so the UI can render "GMT" honestly rather than guessing.
    * ``min_toxicity`` and friends are ``None``-by-default rather than 0. A
      floor of zero and no floor at all are different queries once null scores
      exist: ``toxicity >= 0`` silently drops every post the scorer skipped.
    """

    project_id: str
    date_from: datetime | None = None
    date_to: datetime | None = None
    tz: str = "UTC"
    platform: tuple[str, ...] = ()
    narrative_id: tuple[str, ...] = ()
    cohort_id: tuple[str, ...] = ()
    author_group_id: tuple[str, ...] = ()
    sentiment: tuple[str, ...] = ()
    emotion: tuple[str, ...] = ()
    min_toxicity: float | None = None
    min_risk: float | None = None
    is_anomalous: bool | None = None
    is_bot_like: bool | None = None
    content_type: tuple[str, ...] = ()
    lang: tuple[str, ...] = ()
    q: str | None = None
    #: The UI's "Original data" chip. False means reposts/shares are excluded so
    #: a viral repost swarm cannot inflate every count in the overview.
    include_shared: bool = True

    def as_response_dict(self) -> dict[str, Any]:
        """What every list response echoes back in ``filters_applied``.

        Echoing the filters is not padding: a screenshot of a surprising number
        is only debuggable if the query that produced it travels with it.
        """
        out: dict[str, Any] = {"project_id": self.project_id, "tz": self.tz}
        for name in (
            "date_from",
            "date_to",
            "platform",
            "narrative_id",
            "cohort_id",
            "author_group_id",
            "sentiment",
            "emotion",
            "min_toxicity",
            "min_risk",
            "is_anomalous",
            "is_bot_like",
            "content_type",
            "lang",
            "q",
        ):
            value = getattr(self, name)
            if value in (None, (), ""):
                continue
            out[name] = (
                list(value)
                if isinstance(value, tuple)
                else (value.isoformat() if isinstance(value, datetime) else value)
            )
        out["include_shared"] = self.include_shared
        return out

    @property
    def zoneinfo(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.tz)
        except ZoneInfoNotFoundError:
            return ZoneInfo("UTC")


async def filter_spec(
    project_id: Annotated[str, Query(description="Project uuid or slug. Required everywhere.")],
    date_from: Annotated[datetime | None, Query(description="Inclusive lower bound, UTC.")] = None,
    date_to: Annotated[datetime | None, Query(description="Exclusive upper bound, UTC.")] = None,
    tz: Annotated[
        str,
        Query(description="IANA zone for bucketing and labels only; filtering is always UTC."),
    ] = "UTC",
    platform: Annotated[list[str] | None, Query()] = None,
    narrative_id: Annotated[list[str] | None, Query()] = None,
    cohort_id: Annotated[list[str] | None, Query()] = None,
    author_group_id: Annotated[list[str] | None, Query()] = None,
    sentiment: Annotated[list[str] | None, Query()] = None,
    emotion: Annotated[list[str] | None, Query()] = None,
    min_toxicity: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    min_risk: Annotated[float | None, Query(ge=0.0, le=100.0)] = None,
    is_anomalous: Annotated[bool | None, Query()] = None,
    is_bot_like: Annotated[bool | None, Query()] = None,
    content_type: Annotated[list[str] | None, Query()] = None,
    lang: Annotated[list[str] | None, Query()] = None,
    q: Annotated[str | None, Query(description="Full-text query over post text.")] = None,
    include_shared: Annotated[
        bool, Query(description="False = the UI's 'Original data' chip: exclude reposts.")
    ] = True,
) -> FilterSpec:
    if date_from and date_to and date_from > date_to:
        raise BadRequest(
            "date_from is after date_to.",
            code="invalid_date_range",
            detail={"date_from": date_from.isoformat(), "date_to": date_to.isoformat()},
        )
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise BadRequest(
            f"Unknown timezone {tz!r}. Use an IANA name such as 'UTC' or 'Europe/London'.",
            code="invalid_timezone",
        ) from None

    return FilterSpec(
        project_id=project_id,
        date_from=_as_utc(date_from),
        date_to=_as_utc(date_to),
        tz=tz,
        platform=tuple(platform or ()),
        narrative_id=tuple(narrative_id or ()),
        cohort_id=tuple(cohort_id or ()),
        author_group_id=tuple(author_group_id or ()),
        sentiment=tuple(sentiment or ()),
        emotion=tuple(emotion or ()),
        min_toxicity=min_toxicity,
        min_risk=min_risk,
        is_anomalous=is_anomalous,
        is_bot_like=is_bot_like,
        content_type=tuple(content_type or ()),
        lang=tuple(lang or ()),
        q=q,
        include_shared=include_shared,
    )


def _as_utc(value: datetime | None) -> datetime | None:
    """A naive query parameter means UTC, and is stamped as such explicitly.

    Phase 1 *rejects* naive datetimes at the adapter boundary because a naive
    timestamp there is an adapter bug. Here it is a URL that omitted the offset,
    which is ordinary; assuming UTC is right, but doing it silently is not, so
    the conversion is in one named function that the tests cover.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


Filters = Annotated[FilterSpec, Depends(filter_spec)]
