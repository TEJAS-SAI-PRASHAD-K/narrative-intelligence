"""Unsigned-to-signed simhash conversion, in one place.

Phase 1 computes a 64-bit simhash and Arrow stores it as ``uint64``. Postgres
has no unsigned integer type. The options are:

* **NUMERIC** -- lossless but variable-width, and Hamming-distance work over a
  NUMERIC column loses the integer fast path entirely.
* **Clamp or cast naively** -- silently wrong for the upper half of the range,
  which is half of all values. Near-duplicate detection would keep working well
  enough to look fine and quietly miss half its matches.
* **Reinterpret the same 64 bits as signed** -- lossless, fixed-width, and
  ``a # b`` (XOR) followed by a popcount behaves identically because XOR is
  bitwise and does not care about the sign convention.

The third is what this module does. It is two functions rather than an inline
expression precisely because it must be done the same way in the loader, the
query builder and the test.
"""

from __future__ import annotations

_UINT64_MAX = 2**64 - 1
_SIGN_BIT = 2**63


def to_signed(value: int | None) -> int | None:
    """uint64 -> the int64 with the same bit pattern."""
    if value is None:
        return None
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"simhash {value} does not fit in an unsigned 64-bit integer")
    return value - 2**64 if value >= _SIGN_BIT else value


def from_signed(value: int | None) -> int | None:
    """int64 -> the uint64 with the same bit pattern."""
    if value is None:
        return None
    return value + 2**64 if value < 0 else value


def hamming(left: int | None, right: int | None) -> int | None:
    """Bit distance between two simhashes, sign convention irrelevant.

    XOR is bitwise, so this gives the same answer whether the inputs arrived as
    signed or unsigned -- which is the property that makes the storage choice
    above safe.
    """
    if left is None or right is None:
        return None
    return ((from_signed(left) or 0) ^ (from_signed(right) or 0)).bit_count()
