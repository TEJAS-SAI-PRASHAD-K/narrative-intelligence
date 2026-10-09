"""SQLAlchemy models, one module per entity group.

``import_all_models`` exists because Alembic autogenerate only sees metadata
that has actually been imported. A model module that nobody imports produces no
migration and no error -- just a table that is missing in production. Calling
this from alembic/env.py makes that failure impossible.
"""

from __future__ import annotations

import importlib

#: Every module holding mapped classes. Adding a model module without adding it
#: here is the one mistake this file exists to prevent.
MODEL_MODULES: tuple[str, ...] = (
    "app.models.core",
    "app.models.corpus",
    "app.models.actors",
    "app.models.narratives",
    "app.models.domains",
    "app.models.network",
    "app.models.compass",
    "app.models.embeddings",
    "app.models.ops",
)


def import_all_models() -> None:
    for module in MODEL_MODULES:
        importlib.import_module(module)
