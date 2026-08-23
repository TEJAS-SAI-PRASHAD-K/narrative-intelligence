"""Posts, threads, search and vector similarity.

Column names mirror ``ingest/schema.py`` field-for-field, with one deliberate
difference: Phase 1's ``engagement`` is a nested struct in Parquet, and it is
flattened here into ``likes``/``shares``/``replies``/``views``. The flattening
preserves the null semantics exactly -- ``null`` means the platform does not
expose that metric, ``0`` means measured zero -- because conflating them is what
destroys the coordination signal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import Camel

ContentType = Literal["post", "comment", "article", "video", "video_comment"]


class Engagement(Camel):
    """Four counters, every one of them nullable, and that is the whole point.

    ``null`` is not ``0``. Reddit exposes no view count, so ``views`` is null
    there and a sum over a mixed corpus must skip it rather than adding zero --
    otherwise "average views per post" silently halves the moment Reddit is
    included in the filter.
    """

    likes: int | None = None
    shares: int | None = None
    replies: int | None = None
    views: int | None = None

    @property
    def total(self) -> int | None:
        """Likes + shares + replies, skipping nulls. Views are excluded on
        purpose: they measure exposure, not engagement, and summing them into
        the same number makes a video corpus look 100x more engaged."""
        parts = [v for v in (self.likes, self.shares, self.replies) if v is not None]
        return sum(parts) if parts else None


class PostScores(Camel):
    """Per-post manipulation signals. Every field nullable: a post the scorer
    skipped (wrong language, too short) carries a reason, not a fabricated
    number."""

    toxicity: float | None = None
    is_toxic: bool | None = None
    anomaly: float | None = None
    is_anomalous: bool | None = None
    misinfo_likelihood: float | None = None
    stance: str | None = None
    sentiment: str | None = None
    sentiment_score: float | None = None
    emotion: str | None = None
    emotion_scores: dict[str, float] | None = None
    scoring_version: str | None = None
    scored_at: datetime | None = None
    skip_reasons: list[str] = Field(
        default_factory=list,
        description="Why a null is null. Empty when the post was scored normally.",
    )


class PostOut(Camel):
    id: str = Field(description="Namespaced as '<source>:<native_id>'.")
    project_id: str
    native_id: str
    source: str
    source_detail: str = Field(description="Subreddit, instance, feed or channel.")
    content_type: ContentType
    text: str
    lang: str | None = None
    author_id: str
    author_handle: str | None = None
    timestamp: datetime
    parent_id: str | None = None
    conversation_id: str | None = None
    engagement: Engagement = Field(default_factory=Engagement)
    urls: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    media_urls: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    scores: PostScores | None = None
    narrative_ids: list[str] = Field(default_factory=list)
    #: Present only on /posts/{id}. Omitted from list responses because the raw
    #: payloads are large and differ per source and per API version.
    raw: dict[str, Any] | None = None


class ThreadNode(Camel):
    """One post in a reconstructed conversation."""

    post: PostOut
    depth: int
    children: list[str] = Field(default_factory=list, description="Child post ids, in time order.")


class ThreadResponse(Camel):
    conversation_id: str | None
    root_id: str | None
    nodes: list[ThreadNode]
    truncated: bool = False
    detail: str | None = Field(
        default=None,
        description=(
            "Set when threading is unavailable. Kaggle-sourced Reddit rows carry no "
            "parent_id, and saying so is better than rendering an orphan as a root."
        ),
    )


class SimilarRequest(Camel):
    """Either a post id or raw text, never both.

    Raw text needs a live embedder. When no embedding checkpoint is mounted the
    route accepts ``post_id`` (whose vector is already in the table) and 503s on
    ``text`` with a message that names the missing checkpoint.
    """

    project_id: str
    post_id: str | None = None
    text: str | None = Field(default=None, max_length=8000)
    limit: int = Field(default=20, ge=1, le=200)
    min_similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity floor. Null returns the top-k regardless of distance.",
    )
    exclude_same_author: bool = Field(
        default=False,
        description="Excludes the query author's own posts, which otherwise dominate the top-k.",
    )


class SimilarPost(Camel):
    post: PostOut
    #: Cosine similarity in [-1, 1]. Reported alongside distance because the UI
    #: shows similarity and the index ranks by distance, and quietly converting
    #: between them in the frontend is how a threshold ends up inverted.
    similarity: float
    distance: float


class SimilarResponse(Camel):
    query_post_id: str | None
    query_text: str | None
    model: str
    dim: int
    metric: Literal["cosine"] = "cosine"
    items: list[SimilarPost]


class HighRiskPost(Camel):
    """A row of the overview's high-risk posts panel."""

    post: PostOut
    risk_score: float
    reasons: list[str] = Field(
        description=(
            "Why this post is here, e.g. ['misinfo_likelihood=0.91', 'author bot_prob=0.88']."
        )
    )
    narrative_id: str | None = None
    narrative_title: str | None = None
