"""Tests guarding against the leakage and encoding defects fixed in this revision."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                    # noqa: E402
from pinn_dft.data import encode_fold, geometric_indices       # noqa: E402
from pinn_dft.models.hybrid import out_of_fold_prior           # noqa: E402


@pytest.fixture
def toy_frame():
    rng = np.random.RandomState(0)
    n = 60
    return pd.DataFrame({
        "Unit cell area [Å2]": rng.uniform(10, 30, n),
        "Thickness [Å]": rng.uniform(1, 8, n),
        "Energy [eV]": rng.normal(0, 1, n),
        "Magnetic": rng.choice(["Yes", "No"], n),
        "Layer group (not Space group)": rng.choice(["pmmn", "pm2_1n", "p211"], n),
        "Stoichiometry": rng.choice(["AB2", "ABC3"], n),
    })


def test_encoding_uses_training_vocabulary_only(toy_frame):
    """A category seen only in validation must not create a new column."""
    train = toy_frame.iloc[:40].copy()
    valid = toy_frame.iloc[40:].copy()
    valid.iloc[0, valid.columns.get_loc("Layer group (not Space group)")] = "UNSEEN_GROUP"

    tr, va, columns = encode_fold(train, valid)
    assert tr.shape[1] == va.shape[1] == len(columns)
    assert not any("UNSEEN_GROUP" in c for c in columns)


def test_standardisation_statistics_come_from_training_fold(toy_frame):
    """Training columns standardise to ~zero mean; validation is free to differ."""
    train, valid = toy_frame.iloc[:40], toy_frame.iloc[40:]
    tr, _, columns = encode_fold(train, valid)
    idx = columns.index("Energy [eV]")
    assert abs(tr[:, idx].mean()) < 1e-5


def test_geometric_channels_resolve_to_named_geometric_columns(toy_frame):
    """Regression test: indices must not silently fall back to positions 0/1."""
    tr, va, columns = encode_fold(toy_frame.iloc[:40], toy_frame.iloc[40:])
    a, b = geometric_indices(columns)
    assert columns[a] == config.GEOMETRIC_CHANNELS[0]
    assert columns[b] == config.GEOMETRIC_CHANNELS[1]


def test_geometric_lookup_fails_loudly_when_channel_missing():
    with pytest.raises(KeyError):
        geometric_indices(["some", "unrelated", "columns"])


def test_out_of_fold_prior_is_not_in_sample():
    """The OOF prior must be strictly worse on training rows than an in-sample fit.

    This is the property the leakage fix depends on: if the prior were as good as
    an in-sample fit, the residual head would again be trained against a signal
    it never sees at inference.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 6))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.3, 200)

    estimator = GradientBoostingRegressor(n_estimators=100, random_state=0)
    oof = out_of_fold_prior(estimator, X, y, n_splits=5, seed=0)
    in_sample = estimator.fit(X, y).predict(X)

    oof_err = np.mean((y - oof) ** 2)
    in_sample_err = np.mean((y - in_sample) ** 2)
    assert oof_err > in_sample_err, "OOF prior should be less optimistic than in-sample"
