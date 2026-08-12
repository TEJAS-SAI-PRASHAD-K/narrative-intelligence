"""Narrative Intelligence Platform - Phase 1 ingestion layer.

Public surface is deliberately small: everything downstream should depend on
``ingest.schema`` (the contract) and ``ingest.store`` (how to read the corpus),
never on an individual source adapter.
"""

__version__ = "0.1.0"
