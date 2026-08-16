"""Head-to-head against the prior models, and a DFT-cost ablation of the features.

Two questions are answered here.

1. **Does the current pipeline beat the earlier one?** The earlier code
   (``submit/ablation.py``) specifies XGBoost, gradient boosting and random
   forest at fixed hyperparameters, optionally with hand-built "direct physics"
   features. Those exact estimators are reproduced here and evaluated under the
   leakage-controlled protocol, alongside the stacked ensemble.

2. **How much of the accuracy depends on having run DFT?** The C2DB descriptors
   are not equally cheap. Energy above hull, heat of formation, total energy,
   vacuum level, magnetic moment and magnetic state are DFT *outputs*; thickness
   and unit-cell area come from a relaxed DFT geometry. Only composition
   statistics, atom count, layer group and stoichiometry are obtainable without
   any electronic-structure calculation on the target material.

   A claim that the model avoids expensive DFT is only meaningful if it holds in
   the DFT-free tier, so all three tiers are measured:

   ``full``            every descriptor (what earlier revisions used)
   ``geometry_only``   composition + symmetry + relaxed geometry, no DFT energies
   ``dft_free``        composition + symmetry only; no DFT of any kind

Run::

    python experiments/run_feature_tiers.py --repeats 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# XGBoost and torch each link their own copy of libomp. On macOS this is fatal
# in two different ways: importing torch first makes every later XGBoost fit
# segfault (exit 139), and leaving OpenMP multi-threaded makes the process
# deadlock instead (0% CPU, no progress, no error). Both are avoided by pinning
# OpenMP to a single thread *before* any import that pulls in libomp, and by
# importing xgboost ahead of torch. Do not reorder or remove these five lines.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import xgboost  # noqa: F401,E402  (import order matters - see comment above)

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                    # noqa: E402
from pinn_dft.data import build_dataset, encode_fold           # noqa: E402
from pinn_dft.evaluation.metrics import regression_metrics     # noqa: E402
from pinn_dft.evaluation.statistics import (                   # noqa: E402
    corrected_repeated_kfold_ttest, fold_level_ttest)
from pinn_dft.models.baselines import train_baseline           # noqa: E402
from pinn_dft.models.hybrid import out_of_fold_prior           # noqa: E402
from pinn_dft.models.neural import predict, train_mlp, train_pinn  # noqa: E402
from pinn_dft.utils import seed_everything                     # noqa: E402

#: DFT outputs: require a converged electronic-structure calculation.
DFT_ENERGY_COLUMNS = [
    "Energy above hull [eV/atom]", "Heat of formation [eV/atom]",
    "Energy [eV]", "Vacuum level [eV]", "Total magnetic moment [μB]", "Magnetic",
]
#: Require a relaxed DFT geometry, but not an energy readout.
DFT_GEOMETRY_COLUMNS = ["Thickness [Å]", "Unit cell area [Å2]",
                        "atomic_density [1/Å2]"]

TIERS = {
    "full": [],
    "geometry_only": DFT_ENERGY_COLUMNS,
    "dft_free": DFT_ENERGY_COLUMNS + DFT_GEOMETRY_COLUMNS,
}

BASE = ["rf", "gbr", "svr", "mlp", "pinn"]
TRAINERS = {"mlp": train_mlp, "pinn": train_pinn}


def prior_estimators():
    """The exact estimators specified by the earlier ablation script."""
    from xgboost import XGBRegressor

    return {
        "prior_xgb": XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2,
            random_state=config.SEED, n_jobs=1),
        "prior_gbr": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.85,
            random_state=config.SEED),
        "prior_rf": RandomForestRegressor(
            n_estimators=400, max_depth=14, max_features=0.7,
            random_state=config.SEED, n_jobs=1),
    }


def run(repeats: int, quick: bool) -> None:
    seed_everything(config.SEED)
    t_start = time.time()
    epochs = 200 if quick else 1000

    X_all, y, groups = build_dataset()
    dev_idx, _ = next(GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_FRACTION,
        random_state=config.SEED).split(X_all, y, groups))
    g_dev = groups[dev_idx]
    y_dev = y[dev_idx]

    records = []
    for tier, dropped in TIERS.items():
        X_dev = X_all.iloc[dev_idx].drop(columns=dropped, errors="ignore")
        print(f"\n=== tier '{tier}': {X_dev.shape[1]} columns "
              f"({len(dropped)} dropped) ===")

        for rep in range(repeats):
            rng = np.random.RandomState(config.SEED + rep)
            uniq = np.unique(g_dev)
            remap = dict(zip(uniq, rng.permutation(len(uniq))))
            g_shuffled = np.array([remap[g] for g in g_dev])

            for fold, (tr, va) in enumerate(GroupKFold(config.N_SPLITS)
                                            .split(X_dev, y_dev, g_shuffled)):
                t0 = time.time()
                Xtr, Xva, _ = encode_fold(X_dev.iloc[tr], X_dev.iloc[va])
                ytr, yva = y_dev[tr], y_dev[va]
                seed = config.SEED + 100 * rep + fold
                tag = {"tier": tier, "repeat": rep, "fold": fold,
                       "n_train": len(tr), "n_valid": len(va)}

                # --- estimators from the earlier pipeline -----------------
                for name, est in prior_estimators().items():
                    p = clone(est).fit(Xtr, ytr).predict(Xva)
                    records.append({**tag, "variant": name,
                                    **regression_metrics(yva, p)})

                # --- current pipeline -------------------------------------
                preds, oof = {}, {}
                for name in ("rf", "gbr", "svr"):
                    est = train_baseline(name, Xtr, ytr, tune=False)
                    preds[name] = est.predict(Xva)
                    oof[name] = out_of_fold_prior(clone(est), Xtr, ytr,
                                                  config.INNER_SPLITS, seed)
                for name, trainer in TRAINERS.items():
                    preds[name] = predict(trainer(Xtr, ytr, seed, epochs=epochs), Xva)
                    inner = np.zeros(len(Xtr))
                    for itr, iva in KFold(config.INNER_SPLITS, shuffle=True,
                                          random_state=seed).split(Xtr):
                        inner[iva] = predict(
                            trainer(Xtr[itr], ytr[itr], seed, epochs=epochs),
                            Xtr[iva])
                    oof[name] = inner
                for name in BASE:
                    records.append({**tag, "variant": name,
                                    **regression_metrics(yva, preds[name])})

                OOF = np.column_stack([oof[m] for m in BASE])
                P = np.column_stack([preds[m] for m in BASE])
                meta = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(OOF, ytr)
                records.append({**tag, "variant": "stack_ridge",
                                **regression_metrics(yva, meta.predict(P))})
                w, _ = nnls(OOF, ytr)
                records.append({**tag, "variant": "stack_nnls",
                                **regression_metrics(yva, P @ w)})

                print(f"  [{tier} rep {rep+1}/{repeats} fold {fold+1}/5] "
                      f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(records)
    df.to_csv(config.RESULTS_METRICS / "feature_tier_metrics.csv", index=False)

    # --- summary + significance -----------------------------------------
    summary = {"repeats": repeats, "tiers": {}}
    n_tr, n_va = int(df.n_train.mean()), int(df.n_valid.mean())

    for tier in TIERS:
        sub = df[df.tier == tier]
        piv = sub.pivot_table(index=["repeat", "fold"], columns="variant",
                              values="mse")
        entry = {"n_features_after_encoding": None, "models": {}}
        for v, g in sub.groupby("variant"):
            entry["models"][v] = {"r2_mean": float(g.r2.mean()),
                                  "r2_std": float(g.r2.std(ddof=1)),
                                  "mae_mean": float(g.mae.mean())}
        # stack against each prior-pipeline estimator
        entry["stack_vs_prior"] = {}
        for opponent in ("prior_xgb", "prior_gbr", "prior_rf"):
            if opponent not in piv:
                continue
            d = (piv[opponent] - piv["stack_ridge"]).to_numpy()
            entry["stack_vs_prior"][opponent] = {
                "mean_mse_gain": float(d.mean()),
                "folds_won": int((d > 0).sum()), "n_folds": len(d),
                "p_one_sided": corrected_repeated_kfold_ttest(
                    d, n_train=n_tr, n_test=n_va).p_value_one_sided,
                "p_one_sided_plain": fold_level_ttest(d).p_value_one_sided,
            }
        summary["tiers"][tier] = entry

    # cost of removing DFT descriptors, for the stacked ensemble
    summary["dft_dependence"] = {}
    full = df[(df.tier == "full") & (df.variant == "stack_ridge")]
    for tier in ("geometry_only", "dft_free"):
        sub = df[(df.tier == tier) & (df.variant == "stack_ridge")]
        summary["dft_dependence"][tier] = {
            "r2_mean": float(sub.r2.mean()),
            "mae_mean": float(sub.mae.mean()),
            "delta_r2_vs_full": float(sub.r2.mean() - full.r2.mean()),
            "delta_mae_vs_full": float(sub.mae.mean() - full.mae.mean()),
        }

    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(config.RESULTS_METRICS / "feature_tiers.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 72)
    for tier in TIERS:
        print(f"\n--- tier: {tier}")
        models = summary["tiers"][tier]["models"]
        for v in sorted(models, key=lambda k: -models[k]["r2_mean"]):
            print(f"    {v:12} R2={models[v]['r2_mean']:.4f}"
                  f"+/-{models[v]['r2_std']:.4f}  MAE={models[v]['mae_mean']:.4f}")
        for opp, c in summary["tiers"][tier]["stack_vs_prior"].items():
            print(f"    stack vs {opp:10} won {c['folds_won']}/{c['n_folds']}  "
                  f"p={c['p_one_sided']:.4f}")
    print(f"\nruntime {summary['runtime_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(args.repeats, args.quick)
