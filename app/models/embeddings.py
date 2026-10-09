"""Post embeddings.

pgvector columns are **fixed-width**: the dimension is part of the column type,
not a runtime property. That single fact drives the whole design here.

* One model per deployment, pinned by ``EMBEDDING_MODEL`` + ``EMBEDDING_DIM``.
* The table is named for its dimension (``post_embeddings_384``). Supporting
  MiniLM (384) and BGE (768) simultaneously means two tables behind one
  repository interface -- never a polymorphic column, and above all never
  padding a 384-vector to 768, which produces neighbours that are wrong in a way
  no test on the index would catch.
* Vectors are L2-normalized on write and compared with cosine distance
  everywhere. Mixing L2 and cosine across a codebase produces subtly wrong
  nearest neighbours that nobody notices for weeks.
* The HNSW index is built **after** the bulk load, in its own migration step.
  Building it first turns a ten-minute load into an hour.

The mapped class is generated at import time from the configured dimension so
that Alembic autogenerate sees the right table. Changing ``EMBEDDING_DIM`` is
therefore a migration, not a restart, and ``docs/data-model.md`` says so.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.config import get_api_settings
from app.models.base import Base

#: Dimensions this deployment is allowed to declare. A typo in EMBEDDING_DIM
#: should fail at import with a readable message, not create a table nobody
#: intended and silently diverge from the checkpoint.
SUPPORTED_DIMS = (384, 768, 1024)


def _configured_dim() -> int:
    dim = get_api_settings().embedding_dim
    if dim not in SUPPORTED_DIMS:
        raise RuntimeError(
            f"EMBEDDING_DIM={dim} is not one of {SUPPORTED_DIMS}. pgvector columns are "
            "fixed-width, so this value defines a table shape; add the dimension to "
            "SUPPORTED_DIMS deliberately rather than by typo."
        )
    return dim


EMBEDDING_DIM = _configured_dim()
EMBEDDING_TABLE = f"post_embeddings_{EMBEDDING_DIM}"
#: Named so ``include_object`` in alembic/env.py can skip it: the operator class
#: and build parameters do not round-trip through autogenerate.
HNSW_INDEX_NAME = f"hnsw_{EMBEDDING_TABLE}_cosine"


class PostEmbedding(Base):
    __tablename__ = EMBEDDING_TABLE

    post_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True
    )
    #: Recorded per row, not assumed from the table name. Two checkpoints of the
    #: same model produce vectors that are the same width and not comparable,
    #: and a mixed table has to be detectable rather than merely wrong.
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=text("now()")
    )

    __table_args__ = (
        Index(f"ix_{EMBEDDING_TABLE}_model", "model"),
        # The HNSW index itself is created in a hand-written migration, after
        # the bulk load. Declaring it here would let autogenerate try to manage
        # it and drop the build parameters.
        {"comment": f"L2-normalized {EMBEDDING_DIM}-d vectors; cosine distance only."},
    )
