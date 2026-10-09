"""Operational tables.

These answer "what happened, when, and who asked for it", as opposed to the
pillar tables that answer "what is going on in the corpus".

Step 2 introduces :class:`ApiKey`; the jobs, ingest-run, clustering-run, alert
and report tables land with the rest of the schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, uuid_pk

#: The only three scopes this system has. Deliberately not RBAC: a research
#: capstone with one analyst team does not need roles, and three scopes are
#: auditable at a glance.
SCOPES: tuple[str, ...] = ("read", "write", "admin")


class ApiKey(Base):
    """An issued API key.

    The plaintext key exists exactly once, in the response to the call that
    created it. Only a hash is stored. ``prefix`` is the first eight characters,
    kept in the clear so the UI can identify a key in a list without anybody
    being able to reconstruct it.
    """

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    #: The full verifier string, algorithm and parameters included, so a future
    #: parameter bump can rehash on next use without a migration.
    key_hash: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)

    #: The lookup key on every request: the presented key narrows to one
    #: candidate row by prefix, and only then is the deliberately-expensive hash
    #: verified. Scanning every row's hash per request would be both slow and a
    #: timing oracle. The index lives in __table_args__ and is partial -- see
    #: below -- so there is deliberately no `index=True` here to duplicate it.
    prefix: Mapped[str] = mapped_column(String(16), nullable=False)

    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(String(16)), nullable=False, server_default=text("'{read}'::varchar[]")
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    #: Written best-effort, out of band. An UPDATE on every request would
    #: serialise the whole API behind one row's lock.
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Revocation is a tombstone, not a DELETE: the audit trail of which key did
    #: what has to outlive the key.
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Partial index over live keys only. Revoked keys accumulate forever and
        # never appear in an auth lookup, so they have no business in the index
        # that every request hits.
        Index(
            "ix_api_keys_prefix_active",
            "prefix",
            postgresql_where=text("revoked_at IS NULL"),
        ),
    )

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "revoked" if self.revoked_at else "active"
        return f"<ApiKey {self.prefix}… {self.name!r} {state}>"


class Job(Base):
    """One record for **every** async operation.

    Unified rather than per-task so the frontend polls one endpoint shape
    regardless of what it started: one progress component, one polling hook, one
    error surface for ingest, reclustering and deepfake checks alike.
    """

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = uuid_pk()
    #: Celery's own id, kept so a stuck job can be traced into the broker. Not
    #: the primary key: the row exists before the task is dispatched, and a
    #: dispatch that fails must still leave an auditable failed job.
    celery_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    progress: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Cooperative cancellation. The task polls this at its checkpoints and
    #: stops cleanly rather than being killed mid-write, which is what keeps a
    #: cancelled scoring run from leaving half a batch behind.
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_jobs_created", text("created_at DESC")),
        Index("ix_jobs_project_kind", "project_id", "kind", text("created_at DESC")),
        Index(
            "ix_jobs_active",
            "kind",
            postgresql_where=text("status IN ('pending', 'running')"),
        ),
    )


class IngestRun(Base):
    """One source's ingestion, with full reject accounting.

    ``rejection_reasons`` is not optional bookkeeping. Silent data loss between
    Parquet and Postgres poisons every downstream metric and surfaces three
    weeks later as a number nobody can explain, so every row that does not make
    it is counted against a reason code and written to a rejects file.
    """

    __tablename__ = "ingest_runs"

    id: Mapped[uuid.UUID] = uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    mode: Mapped[str | None] = mapped_column(String(8), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")

    records_in: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    records_loaded: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    records_rejected: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    #: Reason code -> count. Must sum to records_rejected; the ETL asserts it.
    rejection_reasons: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    rejects_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    manifest_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # The invariant that makes "zero rows silently dropped" verifiable
        # rather than merely asserted. Enforced by the database because a
        # loader bug should fail at write time, not surface three weeks later
        # as a metric nobody can explain. Created explicitly in the migration
        # too; declared here so autogenerate does not see it as drift.
        CheckConstraint("records_in = records_loaded + records_rejected", name="reject_accounting"),
        Index("ix_ingest_runs_project_started", "project_id", text("started_at DESC")),
    )


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[uuid.UUID] = uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    #: ``{"metric", "op", "value", "scope", "filters"}``. Deliberately not an
    #: expression language: a DSL evaluated against the corpus on a beat
    #: schedule is a small interpreter with an injection surface, and one
    #: metric/operator/value covers every rule the UI spec asks for.
    condition: Mapped[dict] = mapped_column(JSONB, nullable=False)
    channels: Mapped[list[str]] = mapped_column(
        ARRAY(String(16)), nullable=False, server_default=text("'{in_app}'::varchar[]")
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    #: Without a cooldown a narrative hovering on a threshold fires on every
    #: evaluation cycle and the analyst stops reading alerts, which is strictly
    #: worse than having no alerts at all.
    cooldown_minutes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="60")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_evaluated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_triggered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    trigger_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    __table_args__ = (
        Index(
            "ix_alert_rules_enabled",
            "project_id",
            postgresql_where=text("enabled"),
        ),
    )


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = uuid_pk()
    rule_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("alert_rules.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    narrative_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("narratives.id", ondelete="CASCADE"), nullable=True
    )
    subject_type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="narrative"
    )
    subject_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    #: The observed value and the condition that fired, so an alert explains
    #: itself without the reader re-deriving why it exists.
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    __table_args__ = (
        Index("ix_alerts_project_triggered", "project_id", text("triggered_at DESC")),
        Index(
            "ix_alerts_unacknowledged",
            "project_id",
            postgresql_where=text("acknowledged_at IS NULL"),
        ),
        # The cooldown check's exact query: has this rule fired for this subject
        # recently. Indexed so beat evaluation stays cheap as alerts accumulate.
        Index("ix_alerts_rule_subject", "rule_id", "subject_id", text("triggered_at DESC")),
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    template: Mapped[str] = mapped_column(String(32), nullable=False)
    format: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_reports_project_created", "project_id", text("created_at DESC")),)


class MediaCheck(Base):
    """A deepfake check and its retention deadline.

    ``deletes_at`` is a column rather than a computed value because the purge
    task must be able to find expired uploads with an index scan, and because
    the API promises a specific deletion time in every response. A retention
    policy that is only a config value is a policy nobody can verify was applied.
    """

    __tablename__ = "media_checks"

    id: Mapped[uuid.UUID] = uuid_pk()
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    post_id: Mapped[str | None] = mapped_column(String(512), nullable=True)

    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    media_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    manipulation_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    frames_analyzed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    face_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    #: Required on a completed check. A bare probability is not an acceptable
    #: answer to "is this video fake", and the UI spec says so explicitly.
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    limitations: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: When the *file* is deleted. The verdict row survives; the imagery does not.
    deletes_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_media_checks_submitted", text("submitted_at DESC")),
        Index(
            "ix_media_checks_pending_purge",
            "deletes_at",
            postgresql_where=text("purged_at IS NULL AND storage_path IS NOT NULL"),
        ),
    )
