"""Tabular baselines.

Hyperparameters are selected inside each training fold by randomised search over
an inner split, so the proposed model and every baseline receive the same tuning
budget. The previous revision compared a hand-tuned hybrid against baselines
fixed at library defaults, which biases the comparison in the hybrid's favour.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

from .. import config

SEARCH_BUDGET = 12

_GRIDS: dict[str, dict] = {
    "rf": {
        "n_estimators": [300, 600],
        "max_depth": [7, 12, None],
        "min_samples_leaf": [1, 3, 10],
        "max_features": ["sqrt", 0.3, 0.6],
    },
    "gbr": {
        "n_estimators": [200, 400, 800],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "min_samples_leaf": [1, 5, 10],
    },
    "svr": {
        "C": [0.5, 1.0, 10.0, 100.0],
        "epsilon": [0.05, 0.1, 0.3],
        "gamma": ["scale", 0.01, 0.001],
    },
}

_ESTIMATORS = {
    "rf": lambda: RandomForestRegressor(random_state=config.SEED, n_jobs=-1),
    "gbr": lambda: GradientBoostingRegressor(random_state=config.SEED),
    "svr": lambda: SVR(kernel="rbf"),
}


def train_baseline(name: str, X: np.ndarray, y: np.ndarray, tune: bool = True):
    """Fit a baseline, optionally tuning inside the training fold."""
    if name not in _ESTIMATORS:
        raise KeyError(f"unknown baseline {name!r}; expected one of {sorted(_ESTIMATORS)}")

    estimator = _ESTIMATORS[name]()
    if not tune:
        return estimator.fit(X, y.ravel())

    search = RandomizedSearchCV(
        estimator,
        _GRIDS[name],
        n_iter=SEARCH_BUDGET,
        cv=3,
        scoring="neg_mean_squared_error",
        random_state=config.SEED,
        n_jobs=-1,
    )
    search.fit(X, y.ravel())
    return search.best_estimator_
