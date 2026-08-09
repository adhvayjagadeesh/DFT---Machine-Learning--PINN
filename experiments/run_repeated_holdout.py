"""Repeated held-out evaluation: does stacking beat the best single model?

A single held-out split of 179 materials could not separate the stacked
ensemble from the standalone PINN (0.856 vs 0.855). That is a power problem,
not evidence of equivalence: one split of that size carries an error bar far
wider than the difference being tested.

This script draws ``--splits`` independent formula-grouped held-out sets and
evaluates every model on each, giving a paired comparison across splits rather
than a single point estimate. Model selection (stacking weights, early
stopping) happens strictly inside the development partition of each split.

Run::

    python experiments/run_repeated_holdout.py --splits 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                    # noqa: E402
from pinn_dft.data import build_dataset, encode_fold           # noqa: E402
from pinn_dft.evaluation.metrics import regression_metrics     # noqa: E402
from pinn_dft.evaluation.statistics import fold_level_ttest    # noqa: E402
from pinn_dft.models.baselines import train_baseline           # noqa: E402
from pinn_dft.models.hybrid import out_of_fold_prior           # noqa: E402
from pinn_dft.models.neural import predict, train_mlp, train_pinn  # noqa: E402
from pinn_dft.utils import seed_everything                     # noqa: E402

BASE = ["rf", "gbr", "svr", "mlp", "pinn"]
TRAINERS = {"mlp": train_mlp, "pinn": train_pinn}


def run(splits: int, test_fraction: float, quick: bool) -> None:
    seed_everything(config.SEED)
    t0 = time.time()
    epochs = 200 if quick else 1000

    X_df, y, groups = build_dataset()
    records = []

    for split in range(splits):
        dev_idx, test_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=test_fraction,
            random_state=config.SEED + split).split(X_df, y, groups))
        X_dev, y_dev = X_df.iloc[dev_idx], y[dev_idx]
        X_test, y_test = X_df.iloc[test_idx], y[test_idx]
        Xd, Xt, _ = encode_fold(X_dev, X_test)
        seed = config.SEED + split

        preds, oof = {}, {}
        for name in ("rf", "gbr", "svr"):
            est = train_baseline(name, Xd, y_dev, tune=False)
            preds[name] = est.predict(Xt)
            oof[name] = out_of_fold_prior(clone(est), Xd, y_dev,
                                          config.INNER_SPLITS, seed)
        for name, trainer in TRAINERS.items():
            preds[name] = predict(trainer(Xd, y_dev, seed, epochs=epochs), Xt)
            inner = np.zeros(len(Xd))
            for itr, iva in KFold(config.INNER_SPLITS, shuffle=True,
                                  random_state=seed).split(Xd):
                inner[iva] = predict(
                    trainer(Xd[itr], y_dev[itr], seed, epochs=epochs), Xd[iva])
            oof[name] = inner

        OOF = np.column_stack([oof[m] for m in BASE])
        P = np.column_stack([preds[m] for m in BASE])
        meta = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(OOF, y_dev)
        preds["stack_ridge"] = meta.predict(P)
        w, _ = nnls(OOF, y_dev)
        preds["stack_nnls"] = P @ w

        for name, p in preds.items():
            records.append({"split": split, "n_test": len(test_idx),
                            "variant": name, **regression_metrics(y_test, p)})
        print(f"[split {split + 1}/{splits}] n_test={len(test_idx)} "
              f"stack={records[-2]['r2']:.4f} ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(records)
    df.to_csv(config.RESULTS_METRICS / "repeated_holdout.csv", index=False)

    piv = df.pivot_table(index="split", columns="variant", values="mse")
    summary = {
        "n_splits": splits, "test_fraction": test_fraction,
        "mean_n_test": int(df.n_test.mean()),
        # Built explicitly: a MultiIndex .agg().to_dict() yields tuple keys,
        # which json.dump rejects.
        "per_model": {
            str(v): {
                "r2_mean": float(g.r2.mean()), "r2_std": float(g.r2.std(ddof=1)),
                "mae_mean": float(g.mae.mean()), "mae_std": float(g.mae.std(ddof=1)),
            } for v, g in df.groupby("variant")
        },
        "comparisons": {},
    }
    for challenger in ("stack_ridge", "stack_nnls"):
        for incumbent in ("pinn", "mlp", "gbr"):
            d = (piv[incumbent] - piv[challenger]).to_numpy()
            t = fold_level_ttest(d)
            summary["comparisons"][f"{challenger}_vs_{incumbent}"] = {
                "mean_mse_gain": float(d.mean()),
                "splits_won": int((d > 0).sum()),
                "n_splits": len(d),
                "p_one_sided": t.p_value_one_sided,
            }

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(config.RESULTS_METRICS / "repeated_holdout.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== repeated held-out ({splits} splits, "
          f"~{summary['mean_n_test']} materials each) ===")
    for v, g in df.groupby("variant"):
        print(f"  {v:12} R2={g.r2.mean():.4f}+/-{g.r2.std(ddof=1):.4f}  "
              f"MAE={g.mae.mean():.4f}")
    print()
    for k, c in summary["comparisons"].items():
        print(f"  {k:26} gain={c['mean_mse_gain']:+.4f}  "
              f"won {c['splits_won']}/{c['n_splits']}  p={c['p_one_sided']:.4f}")
    print(f"\nruntime {summary['runtime_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--test-fraction", type=float, default=0.20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(args.splits, args.test_fraction, args.quick)
