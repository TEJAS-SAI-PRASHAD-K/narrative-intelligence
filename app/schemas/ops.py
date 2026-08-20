"""Operational response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    environment: str
    demo_mode: bool
    uptime_seconds: float


class SubsystemStatus(BaseModel):
    name: str
    status: Literal["up", "degraded", "down"]
    detail: str
    latency_ms: float | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    environment: str
    demo_mode: bool
    subsystems: list[SubsystemStatus]
    capabilities: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "One entry per model capability. `status` is ready | degraded | unavailable; "
            "`degraded` means precomputed Phase 2/3 scores are being served because no "
            "checkpoint is mounted."
        ),
    )
