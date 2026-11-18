"""narrative scorecard coordinated_author_ratio

Revision ID: c4d8a2f6e193
Revises: b3e7c41d9a52
Create Date: 2026-11-18 09:00:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c4d8a2f6e193"
down_revision: str | None = "b3e7c41d9a52"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # No backfill: NULL means not measured, and a scorecard written before this
    # column existed did not record what fusion read. The next score.fusion run
    # fills it.
    op.add_column(
        "narrative_scorecards",
        sa.Column("coordinated_author_ratio", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("narrative_scorecards", "coordinated_author_ratio")
