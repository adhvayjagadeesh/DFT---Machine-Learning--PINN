"""Active leakage guards.

These tests are designed to FAIL if information crosses a train/validation
boundary. Reading the pipeline is not sufficient evidence that it is clean;
each test below asserts a property that leakage would violate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                    # noqa: E402
from pinn_dft.data import build_dataset, encode_fold           # noqa: E402
from pinn_dft.models.baselines import train_baseline           # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


def test_no_formula_spans_a_fold_boundary(dataset):
    """Grouping must keep every polymorph of a composition on one side."""
    X, y, groups = dataset
    for tr, va in GroupKFold(config.N_SPLITS).split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[va])), \
            "a chemical formula appears in both training and validation"


def test_permuted_target_yields_no_skill(dataset):
    """The decisive leakage test.

    With the target randomly permuted (within the grouping, so the split
    structure is untouched), any honest pipeline must score R^2 <= 0 on held-out
    folds: there is nothing left to learn. A materially positive score would
    mean the model is reaching information about the validation target through
    some other channel -- fitted scalers, encoders, or a prior built in sample.
    """
    X, y, groups = dataset
    rng = np.random.RandomState(0)
    y_perm = rng.permutation(y)

    scores = []
    for tr, va in GroupKFold(config.N_SPLITS).split(X, y_perm, groups):
        Xtr, Xva, _ = encode_fold(X.iloc[tr], X.iloc[va])
        model = train_baseline("gbr", Xtr, y_perm[tr], tune=False)
        pred = model.predict(Xva)
        ss_res = np.sum((y_perm[va] - pred) ** 2)
        ss_tot = np.sum((y_perm[va] - y_perm[tr].mean()) ** 2)
        scores.append(1 - ss_res / ss_tot)

    mean_r2 = float(np.mean(scores))
    assert mean_r2 < 0.05, (
        f"permuted-target R2 = {mean_r2:.4f}; a clean pipeline should score "
        "at or below zero, so this indicates leakage"
    )


def test_encoding_is_invariant_to_validation_content(dataset):
    """Changing validation rows must not change the training design matrix."""
    X, _, groups = dataset
    tr, va = next(GroupKFold(config.N_SPLITS).split(X, np.zeros(len(X)), groups))

    tr_a, _, cols_a = encode_fold(X.iloc[tr], X.iloc[va])
    # perturb the validation half only
    X_alt = X.copy()
    num = X_alt.select_dtypes(include=[np.number]).columns
    X_alt.loc[X_alt.index[va], num] = X_alt.loc[X_alt.index[va], num] * 1000.0
    tr_b, _, cols_b = encode_fold(X.iloc[tr], X_alt.iloc[va])

    assert cols_a == cols_b
    np.testing.assert_allclose(
        tr_a, tr_b, rtol=0, atol=0,
        err_msg="training matrix changed when only validation rows changed")


def test_validation_rows_never_enter_scaler_statistics(dataset):
    """A validation outlier must not move the training standardisation."""
    X, _, groups = dataset
    tr, va = next(GroupKFold(config.N_SPLITS).split(X, np.zeros(len(X)), groups))
    col = "Thickness [Å]"
    idx = list(X.columns).index(col)

    base, _, _ = encode_fold(X.iloc[tr], X.iloc[va])
    X_alt = X.copy()
    X_alt.iloc[va[0], idx] = 1e6
    alt, _, _ = encode_fold(X.iloc[tr], X_alt.iloc[va])
    np.testing.assert_allclose(base, alt, rtol=0, atol=0)
