"""Significance testing for cross-validated model comparison.

Why the original test was invalid
---------------------------------
The published p-value (2.54e-3) came from a paired t-test over the 930
*per-sample* squared errors. Those errors are not independent observations of
model performance: they are produced by five model pairs, each evaluated on its
own fold. Treating them as n = 930 inflates the effective sample size by roughly
two orders of magnitude and makes the p-value far smaller than the evidence
supports.

Two correct alternatives are provided.

* :func:`corrected_repeated_kfold_ttest` -- the Nadeau-Bengio variance-corrected
  paired t-test, which accounts for the fact that cross-validation training sets
  overlap and therefore that fold-level differences are positively correlated.
  This is the appropriate test for repeated k-fold comparison.
* :func:`fold_level_ttest` -- the plain paired t-test over fold-level score
  differences. Valid but anti-conservative relative to Nadeau-Bengio.

The helper previously named ``run_dietterich_5x2cv_test`` implemented neither
Dietterich's 5x2cv test nor any variance correction; it was a plain t-test over
whatever differences it was handed, applied to 5-fold results. It has been
removed rather than renamed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value_one_sided: float
    p_value_two_sided: float
    mean_difference: float
    n_folds: int
    reject_null: bool
    effect_size: float

    def as_dict(self) -> dict:
        return asdict(self)


def fold_level_ttest(differences, alpha: float = 0.05) -> TestResult:
    """Plain paired t-test over per-fold score differences."""
    d = np.asarray(differences, dtype=float).ravel()
    n = len(d)
    if n < 2:
        raise ValueError("need at least two folds")

    mean, sd = d.mean(), d.std(ddof=1)
    t = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
    p1 = float(stats.t.sf(abs(t), n - 1))
    return TestResult(
        name="paired t-test (fold level)",
        statistic=float(t),
        p_value_one_sided=p1,
        p_value_two_sided=float(2 * p1),
        mean_difference=float(mean),
        n_folds=n,
        reject_null=bool(p1 < alpha),
        effect_size=float(mean / sd) if sd > 0 else 0.0,
    )


def corrected_repeated_kfold_ttest(
    differences, n_train: int, n_test: int, alpha: float = 0.05
) -> TestResult:
    """Nadeau-Bengio corrected paired t-test.

    The variance of the fold-difference mean is inflated by
    ``1/n + n_test/n_train`` to account for overlapping training sets across
    folds. Reference: Nadeau & Bengio, *Inference for the Generalization Error*,
    Machine Learning 52(3), 2003; see also Bouckaert & Frank (2004).
    """
    d = np.asarray(differences, dtype=float).ravel()
    n = len(d)
    if n < 2:
        raise ValueError("need at least two folds")

    mean, var = d.mean(), d.var(ddof=1)
    correction = (1.0 / n) + (n_test / float(n_train))
    denom = np.sqrt(var * correction)
    t = mean / denom if denom > 0 else 0.0
    p1 = float(stats.t.sf(abs(t), n - 1))
    return TestResult(
        name="Nadeau-Bengio corrected paired t-test",
        statistic=float(t),
        p_value_one_sided=p1,
        p_value_two_sided=float(2 * p1),
        mean_difference=float(mean),
        n_folds=n,
        reject_null=bool(p1 < alpha),
        effect_size=float(mean / np.sqrt(var)) if var > 0 else 0.0,
    )


def bootstrap_metric_ci(
    y_true, pred_a, pred_b, metric, n_boot: int = 5000, seed: int = 42
) -> dict:
    """Paired bootstrap CI for the difference ``metric(b) - metric(a)``.

    Complements the fold-level test: it quantifies uncertainty attributable to
    the finite evaluation sample rather than to the choice of folds.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    n = len(y_true)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        deltas[i] = metric(y_true[idx], pred_b[idx]) - metric(y_true[idx], pred_a[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean_delta": float(deltas.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }
