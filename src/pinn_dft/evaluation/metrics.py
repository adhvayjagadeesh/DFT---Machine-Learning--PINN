"""Point-prediction metrics, REC curves, and quantile calibration.

The calibration functions are new. The architecture predicts three quantiles and
the manuscript describes them as providing uncertainty estimates, but no version
of the code ever checked whether those intervals are calibrated. An interval
that is claimed to cover 50% of outcomes and actually covers 20% is not an
uncertainty estimate, so this is a claim the paper cannot currently support.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "max_error": float(np.max(np.abs(y_true - y_pred))),
        "negative_predictions": int((y_pred < 0).sum()),
    }


def rec_curve(y_true, y_pred, tolerances=None):
    """Regression Error Characteristic: fraction resolved within each tolerance."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if tolerances is None:
        tolerances = np.linspace(0.0, 3.0, 121)
    err = np.abs(y_true - y_pred)
    return np.asarray(tolerances), np.array([(err <= t).mean() for t in tolerances])


def accuracy_within(y_true, y_pred, tolerance: float) -> float:
    """Fraction of predictions within ``tolerance`` eV of the reference value."""
    return float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tolerance).mean())


def quantile_coverage(y_true, q_low, q_high, nominal: float = 0.5) -> dict:
    """Empirical coverage of a predicted interval against its nominal level."""
    y_true = np.asarray(y_true)
    lo = np.minimum(np.asarray(q_low), np.asarray(q_high))
    hi = np.maximum(np.asarray(q_low), np.asarray(q_high))
    covered = float(((y_true >= lo) & (y_true <= hi)).mean())
    return {
        "nominal_coverage": nominal,
        "empirical_coverage": covered,
        "coverage_gap": covered - nominal,
        "mean_interval_width": float(np.mean(hi - lo)),
        "calibrated": bool(abs(covered - nominal) <= 0.05),
    }


def pinball_score(y_true, q_pred, tau: float) -> float:
    """Mean pinball loss; lower is better. Proper scoring rule for quantiles."""
    y_true, q_pred = np.asarray(y_true), np.asarray(q_pred)
    residual = y_true - q_pred
    return float(np.mean(np.maximum(tau * residual, (tau - 1.0) * residual)))


def error_vs_covariate(y_true, y_pred, covariate, n_bins: int = 5) -> list[dict]:
    """Absolute error binned by a covariate (e.g. structural aspect ratio).

    Supports claims of the form 'error stays flat as distortion increases' with
    actual measurements rather than assertion.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    covariate = np.asarray(covariate, dtype=float)
    err = np.abs(y_true - y_pred)

    edges = np.quantile(covariate, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        mask = (covariate >= edges[i]) & (covariate < edges[i + 1])
        if mask.sum() == 0:
            continue
        out.append({
            "bin": i,
            "covariate_low": float(edges[i]),
            "covariate_high": float(edges[i + 1]),
            "n": int(mask.sum()),
            "mae": float(err[mask].mean()),
        })
    return out
