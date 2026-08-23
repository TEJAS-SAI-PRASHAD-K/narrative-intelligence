"""Deepfake checks.

The UI spec is explicit that a bare probability is not an acceptable answer to
"is this video fake", so ``explanation`` is a required field on a completed
check, not an optional nicety. It is a plain-language string produced from the
model's own outputs (frames analysed, faces found, which artefacts fired), and
it is what an analyst quotes in a report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import Camel

Verdict = Literal["likely_manipulated", "possibly_manipulated", "likely_authentic", "inconclusive"]


class MediaCheckOut(Camel):
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    filename: str | None = None
    media_type: str | None = Field(default=None, description="MIME type detected from magic bytes.")
    size_bytes: int | None = None
    submitted_at: datetime
    completed_at: datetime | None = None

    verdict: Verdict | None = None
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Calibrated probability for the reported verdict, not the raw logit.",
    )
    manipulation_type: str | None = Field(
        default=None, description="e.g. 'face_swap', 'reenactment', 'none_detected'."
    )
    frames_analyzed: int | None = None
    face_detected: bool | None = None
    explanation: str | None = Field(
        default=None,
        description=(
            "Plain-language account of what the model saw and how much weight it "
            "deserves. Required on a completed check: a bare number is not an "
            "acceptable answer to this question."
        ),
    )
    model: str | None = None
    model_version: str | None = None
    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Known failure modes bearing on this specific check -- compression, "
            "resolution, no face found, out-of-distribution generator."
        ),
    )
    #: When the upload matched a post's media_url, the post it came from.
    post_id: str | None = None
    error: str | None = None
    retention: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "{'deletes_at', 'retention_hours'}. Uploaded media is imagery of real "
            "people and is purged on a timer; the API states when."
        ),
    )
