"""drop media_checks

Revision ID: a91f3c5d7b24
Revises: 5082670cff5d
Create Date: 2026-11-10 09:00:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a91f3c5d7b24"
down_revision: str | None = "5082670cff5d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The JobKind literal no longer accepts this kind, so a stored row would make
    # the job listing fail validation. `jobs.kind` has no CHECK constraint or enum,
    # so deleting the rows is the whole change on that table.
    op.execute("DELETE FROM jobs WHERE kind = 'media.deepfake'")
    op.drop_index("ix_media_checks_submitted", table_name="media_checks")
    op.drop_index(
        "ix_media_checks_pending_purge",
        table_name="media_checks",
        postgresql_where=sa.text("purged_at IS NULL AND storage_path IS NOT NULL"),
    )
    op.drop_table("media_checks")


def downgrade() -> None:
    # This recreates the empty table only. The `jobs` rows that upgrade() deleted do
    # not come back, and neither do the rows that the table held.
    op.create_table(
        "media_checks",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=True),
        sa.Column("project_id", sa.UUID(), nullable=True),
        sa.Column("post_id", sa.String(length=512), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=True),
        sa.Column("media_type", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("storage_path", sa.Text(), nullable=True),
        sa.Column("verdict", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("manipulation_type", sa.String(length=48), nullable=True),
        sa.Column("frames_analyzed", sa.Integer(), nullable=True),
        sa.Column("face_detected", sa.Boolean(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("limitations", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("model_version", sa.String(length=32), nullable=True),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deletes_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("purged_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name=op.f("fk_media_checks_job_id_jobs"), ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_media_checks_project_id_projects"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_media_checks")),
    )
    op.create_index(
        "ix_media_checks_pending_purge",
        "media_checks",
        ["deletes_at"],
        unique=False,
        postgresql_where=sa.text("purged_at IS NULL AND storage_path IS NOT NULL"),
    )
    op.create_index(
        "ix_media_checks_submitted",
        "media_checks",
        [sa.literal_column("submitted_at DESC")],
        unique=False,
    )
