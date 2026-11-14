"""SQL expressions that sum engagement without inventing a measured zero.

`None` means the platform does not expose the metric and `0` means measured zero.
A per-row `coalesce(x, 0)` inside a SUM turns an all-NULL group into 0, so these
helpers return NULL for such a group. Inside a group with at least one measured
value, a NULL term counts as not measured and adds nothing.
"""

from __future__ import annotations

from sqlalchemy import case, func


def measured_sum(*columns):
    """Aggregate: the sum of the measured values, or NULL when none was measured."""
    measured = sum(func.count(column) for column in columns[1:]) + func.count(columns[0])
    total = func.sum(sum((func.coalesce(c, 0) for c in columns[1:]), func.coalesce(columns[0], 0)))
    return case((measured == 0, None), else_=total)


def row_total(*columns):
    """Per-row: the sum of the measured columns, or NULL when the row measured none."""
    unmeasured = columns[0].is_(None)
    for column in columns[1:]:
        unmeasured = unmeasured & column.is_(None)
    total = sum((func.coalesce(c, 0) for c in columns[1:]), func.coalesce(columns[0], 0))
    return case((unmeasured, None), else_=total)
