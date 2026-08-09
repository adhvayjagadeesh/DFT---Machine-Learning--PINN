"""Tests for the significance-testing helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft.evaluation.statistics import (  # noqa: E402
    bootstrap_metric_ci, corrected_repeated_kfold_ttest, fold_level_ttest)


def test_fold_level_ttest_detects_consistent_improvement():
    result = fold_level_ttest([0.021, 0.025, 0.026, 0.012, 0.018])
    assert result.reject_null
    assert result.p_value_one_sided < 0.05
    assert result.n_folds == 5


def test_fold_level_ttest_does_not_flag_zero_centred_differences():
    result = fold_level_ttest([0.02, -0.018, 0.005, -0.011, 0.004])
    assert not result.reject_null
    assert result.p_value_one_sided > 0.05


def test_fold_level_ttest_false_positive_rate_is_near_alpha():
    """Under the null, rejection should occur at roughly the nominal rate.

    A single random draw is a poor test -- five standard normals can easily land
    all-positive -- so this checks the rate across many draws instead.
    """
    rng = np.random.RandomState(0)
    rejections = sum(
        fold_level_ttest(rng.normal(0, 0.02, 5)).reject_null for _ in range(400)
    )
    assert rejections / 400 < 0.12  # one-sided alpha = 0.05, allowing MC slack


def test_nadeau_bengio_is_more_conservative_than_plain_ttest():
    """The variance correction must widen the interval, never narrow it."""
    diffs = [0.021, 0.025, 0.026, -0.004, 0.012]
    plain = fold_level_ttest(diffs)
    corrected = corrected_repeated_kfold_ttest(diffs, n_train=800, n_test=200)
    assert abs(corrected.statistic) < abs(plain.statistic)
    assert corrected.p_value_one_sided > plain.p_value_one_sided


def test_correction_scales_with_test_fraction():
    diffs = [0.02, 0.03, 0.01, 0.02, 0.025]
    small = corrected_repeated_kfold_ttest(diffs, n_train=900, n_test=100)
    large = corrected_repeated_kfold_ttest(diffs, n_train=500, n_test=500)
    assert abs(large.statistic) < abs(small.statistic)


def test_requires_multiple_folds():
    with pytest.raises(ValueError):
        fold_level_ttest([0.01])


def test_bootstrap_ci_brackets_zero_for_identical_models():
    rng = np.random.RandomState(1)
    y = rng.normal(2.5, 1.5, 300)
    pred = y + rng.normal(0, 0.5, 300)
    from sklearn.metrics import r2_score

    out = bootstrap_metric_ci(y, pred, pred.copy(), r2_score, n_boot=200)
    assert out["ci_lower"] <= 0 <= out["ci_upper"]
    assert not out["excludes_zero"]
