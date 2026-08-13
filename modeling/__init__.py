"""Phase 2: modeling and scoring.

Reads Phase 1's normalized corpus, produces trained models plus the scored
Parquet tables that Phase 4 will serve. ``ingest/`` is a read-only contract.
"""

__all__ = ["config", "io", "registry"]
