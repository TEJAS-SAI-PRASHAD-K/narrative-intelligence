"""Benchmark loaders.

Importing this package registers every loader, so ``get_dataset("liar")`` works
without the caller knowing which module it lives in.

The invariant every one of these obeys: **a loader takes a local path and never
downloads.** Every benchmark here is access-gated, request-form or crawler-based;
a loader that quietly fetched something would either fail on a grader's machine
or violate a dataset licence. When the data is absent, the loader raises
``DatasetUnavailable`` with the exact manual steps.
"""

# Import for the side effect of registration. Order is alphabetical, not
# meaningful.
from modeling.datasets import (  # noqa: F401  (registration side effect)
    coaid,
    cresci,
    dfdc,
    faceforensics,
    fakenewsnet,
    fnc1,
    liar,
    stance,
    twibot,
)
from modeling.datasets.base import (
    BenchmarkDataset,
    DatasetInfo,
    DatasetUnavailable,
    LoadedDataset,
    all_datasets,
    availability_table,
    get_dataset,
)

__all__ = [
    "BenchmarkDataset",
    "DatasetInfo",
    "DatasetUnavailable",
    "LoadedDataset",
    "all_datasets",
    "availability_table",
    "get_dataset",
]
