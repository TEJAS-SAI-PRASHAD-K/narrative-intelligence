"""narrative engagement_total nullable

Revision ID: b3e7c41d9a52
Revises: a91f3c5d7b24
Create Date: 2026-11-14 09:00:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b3e7c41d9a52"
down_revision: str | None = "a91f3c5d7b24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "narratives",
        "engagement_total",
        existing_type=sa.BigInteger(),
        nullable=True,
        server_default=None,
    )
    # NULL means no member post exposes likes, shares or replies. A stored 0 from
    # before this change cannot tell that case from a measured zero, so it is
    # recomputed from the member posts with the same rule the writers use.
    op.execute(
        """
        UPDATE narratives n SET engagement_total = NULL
        WHERE NOT EXISTS (
            SELECT 1 FROM narrative_posts np JOIN posts p ON p.id = np.post_id
            WHERE np.narrative_id = n.id
              AND (p.likes IS NOT NULL OR p.shares IS NOT NULL OR p.replies IS NOT NULL)
        )
        """
    )


def downgrade() -> None:
    op.execute("UPDATE narratives SET engagement_total = 0 WHERE engagement_total IS NULL")
    op.alter_column(
        "narratives",
        "engagement_total",
        existing_type=sa.BigInteger(),
        nullable=False,
        server_default="0",
    )
