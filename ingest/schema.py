"""The canonical schema. This is the contract for the entire project.

Every source adapter converges on :class:`Record`. A downstream consumer must
never need to know which platform a record came from in order to process it;
platform-specific fields live in ``raw`` and nowhere else.

Design rules enforced here (not by convention, by validation):

* Timestamps are timezone-aware UTC. Naive datetimes are *rejected*, never
  silently coerced -- a naive Reddit timestamp is a bug in the adapter, and
  silently stamping it UTC hides that bug until the velocity charts look wrong.
* ``engagement`` keys are always present. ``None`` means "the platform does not
  expose this metric"; ``0`` means "measured zero". Conflating the two destroys
  the coordination signal.
* ``id`` and ``author_id`` are namespaced by source, so ids from two platforms
  can never collide when the corpus is unioned.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Source = Literal["reddit", "mastodon", "news", "gdelt", "youtube"]
ContentType = Literal["post", "comment", "article", "video", "video_comment"]

SOURCES: tuple[str, ...] = ("reddit", "mastodon", "news", "gdelt", "youtube")

#: Sentinel used when a platform has removed the author but kept the content.
#: We keep the record (the text is still evidence) but the author is unusable
#: for coordination work, and this makes that explicit rather than implied.
DELETED_AUTHOR = "__deleted__"

_UINT64_MAX = 2**64 - 1
_MAX_TEXT_CHARS = 200_000


class DropReason(str, Enum):
    """Reason codes for records we refuse to emit.

    Every drop is counted and logged. Silent data loss is the failure mode that
    ruins this kind of project: it surfaces three weeks later as an
    unexplainable metric.
    """

    DELETED_TEXT = "deleted_text"
    EMPTY_TEXT = "empty_text"
    MISSING_TIMESTAMP = "missing_timestamp"
    MISSING_ID = "missing_id"
    VALIDATION_ERROR = "validation_error"
    DUPLICATE_ID = "duplicate_id"
    UNSUPPORTED_TYPE = "unsupported_type"
    OUT_OF_RANGE = "out_of_range"


def utcnow() -> datetime:
    """Timezone-aware now. Never use ``datetime.utcnow()`` in this codebase."""
    return datetime.now(timezone.utc)


def make_id(source: str, native_id: str | int) -> str:
    """Namespace a native id: ``make_id("reddit", "t3_abc") -> "reddit:t3_abc"``."""
    return f"{source}:{native_id}"


def _require_aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{field_name} must be timezone-aware; got naive {value!r}. "
            "Convert at the adapter boundary (e.g. "
            "datetime.fromtimestamp(ts, tz=timezone.utc))."
        )
    return value.astimezone(timezone.utc)


class EngagementMetrics(BaseModel):
    """Platform engagement counters.

    All four keys are always present in the serialized form. ``None`` is not the
    same as ``0`` -- see the module docstring.
    """

    model_config = ConfigDict(extra="forbid")

    likes: int | None = None
    shares: int | None = None
    replies: int | None = None
    views: int | None = None

    # Note: negatives are legal. A Reddit score of -12 is real information and
    # must not be clamped to 0 -- that would look like "measured zero".

    @classmethod
    def unavailable(cls) -> EngagementMetrics:
        """Explicitly-unmeasured metrics. Reads better than ``EngagementMetrics()``."""
        return cls()


class Record(BaseModel):
    """One post / comment / article / video / toot, normalized."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = ""  # derived from source + native_id when omitted
    native_id: str
    source: Source
    source_detail: str
    content_type: ContentType
    text: str
    lang: str | None = None
    author_id: str
    author_handle: str | None = None
    timestamp: datetime
    parent_id: str | None = None
    conversation_id: str | None = None
    engagement: EngagementMetrics = Field(default_factory=EngagementMetrics)
    urls: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    media_urls: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    #: 64-bit simhash over word 3-grams. Phase 2 uses this for near-dup work;
    #: Phase 1 computes it and takes no opinion on it.
    simhash: int | None = None
    ingested_at: datetime = Field(default_factory=utcnow)
    raw: dict[str, Any] = Field(default_factory=dict)

    # --- validators ------------------------------------------------------
    @field_validator("timestamp", "ingested_at")
    @classmethod
    def _aware_utc(cls, v: datetime, info) -> datetime:
        return _require_aware_utc(v, info.field_name)

    @field_validator("native_id", "source_detail", "author_id")
    @classmethod
    def _non_empty(cls, v: str, info) -> str:
        v = v.strip()
        if not v:
            raise ValueError(f"{info.field_name} must not be empty")
        return v

    @field_validator("text")
    @classmethod
    def _bounded_text(cls, v: str) -> str:
        if len(v) > _MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds {_MAX_TEXT_CHARS} chars ({len(v)}); truncate upstream")
        return v

    @field_validator("lang")
    @classmethod
    def _iso639_1(cls, v: str | None) -> str | None:
        """Keep the primary subtag only: ``zh-cn`` -> ``zh``, ``EN`` -> ``en``."""
        if v is None:
            return None
        v = v.strip().lower()
        if not v or v in {"und", "unknown", "zxx"}:
            return None
        primary = re.split(r"[-_]", v, maxsplit=1)[0]
        if not re.fullmatch(r"[a-z]{2,3}", primary):
            raise ValueError(f"lang must be an ISO 639-1/3 code, got {v!r}")
        return primary

    @field_validator("simhash")
    @classmethod
    def _uint64(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if not 0 <= v <= _UINT64_MAX:
            raise ValueError("simhash must fit in an unsigned 64-bit integer")
        return v

    @field_validator("urls", "domains", "media_urls", "hashtags", "mentions")
    @classmethod
    def _dedupe_preserving_order(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in v:
            if item is None:
                continue
            item = str(item).strip()
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @model_validator(mode="after")
    def _namespace_ids(self) -> Record:
        prefix = f"{self.source}:"
        if not self.id:
            object.__setattr__(self, "id", f"{prefix}{self.native_id}")
        elif not self.id.startswith(prefix):
            raise ValueError(
                f"id {self.id!r} must be namespaced as '{self.source}:<native_id>'; "
                "use ingest.schema.make_id()"
            )
        if not self.author_id.startswith(prefix):
            raise ValueError(
                f"author_id {self.author_id!r} must be namespaced as "
                f"'{self.source}:<native_author_id>'; use ingest.schema.make_id()"
            )
        for field in ("parent_id", "conversation_id"):
            value = getattr(self, field)
            if value is not None and not value.startswith(prefix):
                raise ValueError(
                    f"{field} {value!r} must be namespaced as '{self.source}:<native_id>' "
                    "so threading survives a cross-source union"
                )
        return self

    # --- helpers ---------------------------------------------------------
    @property
    def date_partition(self) -> str:
        """``YYYY-MM-DD`` in UTC -- the partition key on disk."""
        return self.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def author_is_deleted(self) -> bool:
        return self.author_id.endswith(f":{DELETED_AUTHOR}")


class Author(BaseModel):
    """Per-author roll-up so Phase 2's coordination module has something to score.

    Account age, follower counts and the bot flag are the cheap priors that
    separate "loud human" from "three-day-old account posting 400 times".
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    author_id: str
    source: Source
    handle: str | None = None
    created_at: datetime | None = None
    followers: int | None = None
    following: int | None = None
    post_count: int = 0
    first_seen: datetime
    last_seen: datetime
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at", "first_seen", "last_seen")
    @classmethod
    def _aware_utc(cls, v: datetime | None, info) -> datetime | None:
        if v is None:
            return None
        return _require_aware_utc(v, info.field_name)

    @model_validator(mode="after")
    def _check(self) -> Author:
        if not self.author_id.startswith(f"{self.source}:"):
            raise ValueError(f"author_id {self.author_id!r} must start with '{self.source}:'")
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen precedes first_seen")
        return self
