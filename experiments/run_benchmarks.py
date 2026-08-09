"""Full benchmark: seven models, grouped cross-validation, held-out test set.

Run from the repository root::

    python experiments/run_benchmarks.py --splitter group --tune

Outputs (all under ``results/``):
    metrics/predictions_oof.csv     pooled out-of-fold predictions, all models
    metrics/fold_metrics.csv        per-fold R2/MSE/MAE, all models
    metrics/benchmark_summary.json  pooled metrics, significance tests, calibration
    metrics/test_set_metrics.json   held-out test-set metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                    # noqa: E402
from pinn_dft.data import build_dataset, encode_fold, geometric_indices   # noqa: E402
from pinn_dft.evaluation.metrics import (                      # noqa: E402
    quantile_coverage, regression_metrics, accuracy_within, error_vs_covariate)
from pinn_dft.evaluation.statistics import (                   # noqa: E402
    bootstrap_metric_ci, corrected_repeated_kfold_ttest, fold_level_ttest)
from pinn_dft.models.baselines import train_baseline           # noqa: E402
from pinn_dft.models.hybrid import (                           # noqa: E402
    HybridConfig, predict_hybrid, train_hybrid)
from pinn_dft.models.neural import predict, train_mlp, train_pinn   # noqa: E402
from pinn_dft.utils import seed_everything                     # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score       # noqa: E402

MODELS = ["rf", "gbr", "svr", "mlp", "pinn", "hybrid_rf", "hybrid_gbr"]


def make_splitter(kind: str, n_splits: int):
    return GroupKFold(n_splits=n_splits) if kind == "group" else KFold(
        n_splits=n_splits, shuffle=True, random_state=config.SEED)


def run(splitter_kind: str, tune: bool, quick: bool) -> None:
    seed_everything(config.SEED)
    t_start = time.time()

    X_df, y, groups = build_dataset()
    print(f"[data] {X_df.shape[0]} materials x {X_df.shape[1]} raw columns")

    # --- held-out test set, carved before any model selection ---------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_FRACTION,
                                 random_state=config.SEED)
    dev_idx, test_idx = next(splitter.split(X_df, y, groups))
    X_dev, y_dev, g_dev = X_df.iloc[dev_idx], y[dev_idx], groups[dev_idx]
    X_test, y_test = X_df.iloc[test_idx], y[test_idx]
    print(f"[data] development set {len(dev_idx)} | held-out test set {len(test_idx)}")

    cv = make_splitter(splitter_kind, config.N_SPLITS)
    rows, fold_rows = [], []
    quantile_store: dict[int, np.ndarray] = {}

    for fold, (tr, va) in enumerate(cv.split(X_dev, y_dev, g_dev)):
        t0 = time.time()
        Xtr_df, Xva_df = X_dev.iloc[tr], X_dev.iloc[va]
        ytr, yva = y_dev[tr], y_dev[va]

        Xtr, Xva, columns = encode_fold(Xtr_df, Xva_df)
        ch_a, ch_b = geometric_indices(columns)
        seed = config.SEED + fold
        preds: dict[str, np.ndarray] = {}

        for name in ("rf", "gbr", "svr"):
            model = train_baseline(name, Xtr, ytr, tune=tune and not quick)
            preds[name] = model.predict(Xva)

        epochs = 200 if quick else 1200
        preds["mlp"] = predict(train_mlp(Xtr, ytr, seed, epochs=epochs), Xva)
        preds["pinn"] = predict(train_pinn(Xtr, ytr, seed, epochs=epochs), Xva)

        for base_name, key in (("rf", "hybrid_rf"), ("gbr", "hybrid_gbr")):
            base = clone(train_baseline(base_name, Xtr, ytr, tune=False))
            model, va_feats, _ = train_hybrid(
                base, Xtr, ytr, Xva, ch_a, ch_b, HybridConfig(), seed,
                epochs=200 if quick else 1000)
            point, quantiles = predict_hybrid(model, va_feats)
            preds[key] = point
            if key == "hybrid_gbr":
                quantile_store[fold] = quantiles

        aspect = (Xva[:, ch_a] / (Xva[:, ch_b] + 1e-6))
        for i, idx in enumerate(va):
            row = {"fold": fold, "row_index": int(idx), "true": float(yva[i]),
                   "aspect_ratio": float(aspect[i])}
            row.update({m: float(preds[m][i]) for m in MODELS})
            if fold in quantile_store:
                q = quantile_store[fold][i]
                row.update({"hybrid_gbr_q25": float(q[0]),
                            "hybrid_gbr_q50": float(q[1]),
                            "hybrid_gbr_q75": float(q[2])})
            rows.append(row)

        for m in MODELS:
            fold_rows.append({"fold": fold, "model": m, "n_valid": len(va),
                              "n_train": len(tr), **regression_metrics(yva, preds[m])})
        print(f"[fold {fold + 1}/{config.N_SPLITS}] "
              f"gbr R2={r2_score(yva, preds['gbr']):.4f} "
              f"hybrid R2={r2_score(yva, preds['hybrid_gbr']):.4f} "
              f"({time.time() - t0:.0f}s)")

    oof = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    oof.to_csv(config.RESULTS_METRICS / "predictions_oof.csv", index=False)
    folds.to_csv(config.RESULTS_METRICS / "fold_metrics.csv", index=False)

    # --- pooled metrics ------------------------------------------------------
    summary = {
        "protocol": {
            "splitter": splitter_kind, "n_splits": config.N_SPLITS,
            "tuned_baselines": bool(tune and not quick), "seed": config.SEED,
            "n_development": int(len(dev_idx)), "n_test": int(len(test_idx)),
            "n_features_raw": int(X_df.shape[1]),
        },
        "pooled_out_of_fold": {m: regression_metrics(oof["true"], oof[m]) for m in MODELS},
        "fold_mean_std": {
            m: {
                "r2_mean": float(folds[folds.model == m].r2.mean()),
                "r2_std": float(folds[folds.model == m].r2.std(ddof=1)),
                "mae_mean": float(folds[folds.model == m].mae.mean()),
            } for m in MODELS
        },
    }

    # --- significance --------------------------------------------------------
    gbr_mse = folds[folds.model == "gbr"].sort_values("fold").mse.to_numpy()
    hyb_mse = folds[folds.model == "hybrid_gbr"].sort_values("fold").mse.to_numpy()
    diffs = gbr_mse - hyb_mse
    n_tr = int(folds[folds.model == "gbr"].n_train.mean())
    n_va = int(folds[folds.model == "gbr"].n_valid.mean())

    summary["significance"] = {
        "fold_mse_differences": diffs.tolist(),
        "folds_improved": int((diffs > 0).sum()),
        "plain_fold_ttest": fold_level_ttest(diffs).as_dict(),
        "nadeau_bengio_corrected": corrected_repeated_kfold_ttest(
            diffs, n_train=n_tr, n_test=n_va).as_dict(),
        "bootstrap_delta_r2": bootstrap_metric_ci(
            oof["true"], oof["gbr"], oof["hybrid_gbr"], r2_score),
        "bootstrap_delta_mse": bootstrap_metric_ci(
            oof["true"], oof["gbr"], oof["hybrid_gbr"],
            lambda a, b: -mean_squared_error(a, b)),
    }

    # --- REC + calibration + error vs distortion -----------------------------
    summary["accuracy_within_tolerance"] = {
        m: {f"{t}eV": accuracy_within(oof["true"], oof[m], t)
            for t in (0.1, 0.25, 0.5, 1.0)} for m in ("gbr", "hybrid_gbr")
    }
    if "hybrid_gbr_q25" in oof.columns:
        summary["quantile_calibration"] = quantile_coverage(
            oof["true"], oof["hybrid_gbr_q25"], oof["hybrid_gbr_q75"], nominal=0.5)
    summary["error_vs_aspect_ratio"] = {
        m: error_vs_covariate(oof["true"], oof[m], oof["aspect_ratio"])
        for m in ("gbr", "hybrid_gbr")
    }

    # --- held-out test set ---------------------------------------------------
    Xd, Xt, columns = encode_fold(X_dev, X_test)
    ch_a, ch_b = geometric_indices(columns)
    test_metrics = {}
    for name in ("rf", "gbr", "svr"):
        test_metrics[name] = regression_metrics(
            y_test, train_baseline(name, Xd, y_dev, tune=tune and not quick).predict(Xt))
    base = clone(train_baseline("gbr", Xd, y_dev, tune=False))
    model, t_feats, _ = train_hybrid(base, Xd, y_dev, Xt, ch_a, ch_b,
                                     HybridConfig(), config.SEED,
                                     epochs=200 if quick else 1000)
    point, _ = predict_hybrid(model, t_feats)
    test_metrics["hybrid_gbr"] = regression_metrics(y_test, point)
    summary["held_out_test"] = test_metrics

    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(config.RESULTS_METRICS / "benchmark_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== pooled out-of-fold ===")
    for m in MODELS:
        d = summary["pooled_out_of_fold"][m]
        print(f"  {m:11} R2={d['r2']:.4f}  MSE={d['mse']:.4f}  MAE={d['mae']:.4f}")
    nb = summary["significance"]["nadeau_bengio_corrected"]
    print(f"\nNadeau-Bengio corrected: t={nb['statistic']:.3f} "
          f"p={nb['p_value_one_sided']:.4f} (one-sided), folds improved "
          f"{summary['significance']['folds_improved']}/{config.N_SPLITS}")
    print(f"held-out test hybrid R2={test_metrics['hybrid_gbr']['r2']:.4f} "
          f"vs gbr {test_metrics['gbr']['r2']:.4f}")
    print(f"total runtime {summary['runtime_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splitter", choices=["group", "random"], default="group",
                    help="group = never split polymorphs of one formula across folds")
    ap.add_argument("--tune", action="store_true", help="tune baselines inside each fold")
    ap.add_argument("--quick", action="store_true", help="short run for smoke-testing")
    args = ap.parse_args()
    run(args.splitter, args.tune, args.quick)
