"""Phase 2: modeling and scoring.

Reads Phase 1's normalized corpus, produces trained models plus the scored
Parquet tables that Phase 4 will serve. ``ingest/`` is a read-only contract.
"""

# XGBoost must be imported before PyTorch. This is not a style preference.
#
# On macOS both ship their own OpenMP runtime (`libomp.dylib`). Loading torch
# first and XGBoost second puts two OpenMP runtimes in one process, and the
# first `XGBClassifier.fit` then dies with SIGSEGV -- no traceback, no message,
# exit 139. It cost an afternoon to find, because a segfault looks identical to
# a silent success in a piped terminal.
#
# Measured on this project: torch->xgboost segfaults; xgboost->torch is fine,
# with full threading on both. `OMP_NUM_THREADS=1` also avoids it but serialises
# transformer training, and `KMP_DUPLICATE_LIB_OK=TRUE` is the widely-quoted
# workaround that Intel explicitly documents as able to produce *wrong results*
# -- unacceptable in a project whose entire premise is defensible numbers.
#
# So: claim the OpenMP runtime here, at the package root, before anything else
# in this package can import torch. Guarded because xgboost is an optional
# extra; the ingestion layer and the split/IO tests run without it.
try:  # pragma: no cover - import-order guard, not logic
    import xgboost as _xgboost  # noqa: F401
except ImportError:
    pass

__all__ = ["config", "io", "registry"]
