"""``macro_f1_binary`` and ``paired_bootstrap_difference`` in modeling/eval/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from modeling.eval.metrics import macro_f1_binary, paired_bootstrap_difference


def test_macro_f1_binary_hand_calculated():
    # class 1: tp 1, fp 0, fn 1 -> F1 = 2/3.   class 0: tp 2, fp 1, fn 0 -> F1 = 4/5.
    # macro = (2/3 + 4/5) / 2 = 11/15
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 0, 0, 0])
    assert macro_f1_binary(y, p) == pytest.approx(11 / 15)


def test_macro_f1_binary_one_class_present_uses_zero_for_the_absent_class():
    # All true 1, all predicted 1: class 1 F1 = 1, class 0 has no support and no
    # prediction, so its F1 is 0 by convention. macro = 0.5.
    y = np.ones(5, dtype=int)
    assert macro_f1_binary(y, y) == 0.5


def test_macro_f1_binary_matches_sklearn_on_random_data():
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.integers(0, 2, 40)
        p = rng.integers(0, 2, 40)
        expected = f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)
        assert macro_f1_binary(y, p) == pytest.approx(expected)


def test_paired_bootstrap_uses_the_same_indices_for_both_models():
    n = 30
    y = np.arange(n) % 2
    pred_a = np.arange(n)
    pred_b = np.arange(n) + 1000
    seen: list[np.ndarray] = []

    def recorder(_y, p):
        seen.append(np.asarray(p).copy())
        return 0.0

    paired_bootstrap_difference(y, pred_a, pred_b, recorder, n_bootstrap=25, seed=1)
    # Calls: one point estimate pair, then one pair per resample. Within a pair
    # the second array is the first shifted by 1000 only if the rows are the same.
    assert len(seen) == 2 * (1 + 25)
    for a, b in zip(seen[0::2], seen[1::2], strict=True):
        assert np.array_equal(b, a + 1000)


def test_identical_models_have_interval_exactly_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.integers(0, 2, 200)
    result = paired_bootstrap_difference(
        y, p, p.copy(), macro_f1_binary, n_bootstrap=300, seed=5
    )
    assert (result.value, result.low, result.high) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_is_deterministic_and_seed_sensitive():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 120)
    a = np.where(rng.random(120) < 0.8, y, 1 - y)
    b = np.where(rng.random(120) < 0.6, y, 1 - y)
    one = paired_bootstrap_difference(y, a, b, macro_f1_binary, n_bootstrap=200, seed=9)
    two = paired_bootstrap_difference(y, a, b, macro_f1_binary, n_bootstrap=200, seed=9)
    other = paired_bootstrap_difference(y, a, b, macro_f1_binary, n_bootstrap=200, seed=10)
    assert one.as_dict() == two.as_dict()
    assert (one.low, one.high) != (other.low, other.high)


def test_paired_bootstrap_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_difference(
            np.zeros(10), np.zeros(10), np.zeros(9), macro_f1_binary, n_bootstrap=5
        )


def test_paired_bootstrap_under_ten_rows_gives_no_interval():
    y = np.array([0, 1, 0, 1])
    result = paired_bootstrap_difference(y, y, 1 - y, macro_f1_binary, n_bootstrap=50)
    assert result.value == pytest.approx(1.0)
    assert np.isnan(result.low) and np.isnan(result.high)
