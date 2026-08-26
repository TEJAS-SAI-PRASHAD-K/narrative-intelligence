"""Normalization for fusion sub-scores.

Every sub-score has to arrive in ``[0,1]`` before it can be weighted, and *how*
that mapping happens is a decision that has to be made once and applied
everywhere. Mixing percentile-rank and min-max across a codebase produces scores
that are individually defensible and mutually incomparable, and nobody notices
until two numbers that should agree do not.

The choice is percentile-rank within the project, and the reasoning is in
``configs/fusion.yaml`` next to the setting.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Normalizer:
    """Maps a raw sub-score to ``[0,1]`` against a reference distribution."""

    method: str
    #: Sorted reference values from this project. Empty means no distribution.
    reference: tuple[float, ...] = ()
    min_sample: int = 20
    fallback: str = "raw"

    @classmethod
    def from_values(cls, values, *, method: str = "percentile", min_sample: int = 20):
        clean = sorted(float(v) for v in values if v is not None)
        return cls(method=method, reference=tuple(clean), min_sample=min_sample)

    @property
    def usable(self) -> bool:
        return len(self.reference) >= self.min_sample

    def __call__(self, value: float | None) -> float | None:
        """Normalize. ``None`` in, ``None`` out -- never a zero.

        The null passthrough is the whole discipline: a component that was not
        measured must stay unmeasured all the way to the response, so the
        weights can renormalize over what actually exists.
        """
        if value is None:
            return None
        if self.method == "percentile" and self.usable:
            return self._percentile(float(value))
        if self.method == "minmax" and self.reference:
            low, high = self.reference[0], self.reference[-1]
            if high == low:
                return 0.0
            return _clamp((float(value) - low) / (high - low))
        # Not enough of a distribution to rank against. Clamping the raw value
        # is honest for a score already in [0,1]; the caller records that the
        # fallback was used so the response can say so.
        return _clamp(float(value))

    def _percentile(self, value: float) -> float:
        """Fraction of the reference at or below ``value``.

        ``bisect_left`` over a pre-sorted tuple: the reference is built once per
        scoring run and queried once per narrative, so an O(log n) lookup keeps
        a whole-project rescore linear rather than quadratic.
        """
        position = bisect_left(self.reference, value)
        return _clamp(position / max(len(self.reference) - 1, 1))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def weighted_mean(
    parts: dict[str, float | None], weights: dict[str, float]
) -> tuple[float | None, list[str], bool]:
    """Combine sub-scores, renormalizing the weights over what is present.

    Returns ``(value, missing, renormalized)``.

    The rule this function exists to enforce: **a missing input is not a zero.**
    If the deepfake module never ran, authenticity is absent and the remaining
    weights are scaled up to sum to one, so a narrative is scored on the evidence
    that exists rather than penalised for evidence that does not. Treating
    absence as a zero would systematically under-flag exactly the narratives
    nobody has looked at yet.

    Returns ``None`` when nothing at all is available, because a score computed
    from no inputs is not a low score, it is not a score.
    """
    present = {name: value for name, value in parts.items() if value is not None}
    missing = sorted(name for name in weights if parts.get(name) is None)
    if not present:
        return None, missing, False

    live_weight = sum(weights.get(name, 0.0) for name in present)
    if live_weight <= 0:
        return None, missing, False

    total = sum(value * weights.get(name, 0.0) for name, value in present.items())
    renormalized = bool(missing) and abs(live_weight - 1.0) > 1e-9
    return _clamp(total / live_weight), missing, renormalized


def renormalized_weights(
    parts: dict[str, float | None], weights: dict[str, float]
) -> dict[str, float]:
    """The weights actually applied, for the explainability payload.

    Served verbatim by ``/narratives/{id}/score``. A user reading the drilldown
    must see the weight that was *used*, not the one in the config file, or the
    numbers will not reconcile and the explanation will look wrong.
    """
    present = [name for name, value in parts.items() if value is not None]
    live_weight = sum(weights.get(name, 0.0) for name in present)
    if live_weight <= 0:
        return dict(weights)
    return {
        name: (weights.get(name, 0.0) / live_weight if name in present else weights.get(name, 0.0))
        for name in weights
    }
