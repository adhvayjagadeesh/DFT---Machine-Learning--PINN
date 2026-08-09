"""Why stacking works, and how far the result generalises.

Five analyses, all on the leakage-free protocol:

1. Out-of-fold predictions for every base learner and both meta-learners,
   persisted so every downstream figure derives from one run.
2. Drop-one ensemble ablation -- removing each base learner in turn. This is
   what substantiates a claim about any individual member's contribution;
   a large stacking weight alone does not establish it.
3. Error-correlation structure between base learners, which is the mechanism
   stacking exploits: gains come from decorrelated errors, not from any single
   member being strong.
4. Learning curves over training-set fraction, testing whether the ensemble
   advantage is an artefact of sample size.
5. Held-out test-set confirmation, and permutation importance for the
   strongest single model.

Run::

    python experiments/run_stacking_analysis.py --repeats 3
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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
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

BASE = ["rf", "gbr", "svr", "mlp", "pinn"]
TRAINERS = {"mlp": train_mlp, "pinn": train_pinn}


def _fit_fold(Xtr, ytr, Xva, seed, epochs, need_oof=True):
    """Fit every base learner; return validation predictions and in-fold OOF."""
    preds, oof = {}, {}
    for name in ("rf", "gbr", "svr"):
        est = train_baseline(name, Xtr, ytr, tune=False)
        preds[name] = est.predict(Xva)
        if need_oof:
            oof[name] = out_of_fold_prior(clone(est), Xtr, ytr,
                                          config.INNER_SPLITS, seed)
    for name, trainer in TRAINERS.items():
        preds[name] = predict(trainer(Xtr, ytr, seed, epochs=epochs), Xva)
        if need_oof:
            inner = np.zeros(len(Xtr))
            for itr, iva in KFold(config.INNER_SPLITS, shuffle=True,
                                  random_state=seed).split(Xtr):
                inner[iva] = predict(
                    trainer(Xtr[itr], ytr[itr], seed, epochs=epochs), Xtr[iva])
            oof[name] = inner
    return preds, oof


def _stack(oof_matrix, ytr, pred_matrix, method="ridge"):
    if method == "ridge":
        meta = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(oof_matrix, ytr)
        return meta.predict(pred_matrix), meta.coef_
    weights, _ = nnls(oof_matrix, ytr)
    return pred_matrix @ weights, weights


def run(repeats: int, quick: bool) -> None:
    seed_everything(config.SEED)
    t0 = time.time()
    epochs = 200 if quick else 1000

    X_df, y, groups = build_dataset()
    dev_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_FRACTION,
        random_state=config.SEED).split(X_df, y, groups))
    X_dev, y_dev, g_dev = X_df.iloc[dev_idx], y[dev_idx], groups[dev_idx]
    X_test, y_test = X_df.iloc[test_idx], y[test_idx]

    rows, weight_rows, drop_rows = [], [], []
    oof_frames = []

    for rep in range(repeats):
        rng = np.random.RandomState(config.SEED + rep)
        uniq = np.unique(g_dev)
        remap = dict(zip(uniq, rng.permutation(len(uniq))))
        g_shuffled = np.array([remap[g] for g in g_dev])

        for fold, (tr, va) in enumerate(GroupKFold(config.N_SPLITS)
                                        .split(X_dev, y_dev, g_shuffled)):
            ft = time.time()
            Xtr, Xva, _ = encode_fold(X_dev.iloc[tr], X_dev.iloc[va])
            ytr, yva = y_dev[tr], y_dev[va]
            seed = config.SEED + 100 * rep + fold
            tag = {"repeat": rep, "fold": fold,
                   "n_train": len(tr), "n_valid": len(va)}

            preds, oof = _fit_fold(Xtr, ytr, Xva, seed, epochs)
            OOF = np.column_stack([oof[m] for m in BASE])
            P = np.column_stack([preds[m] for m in BASE])

            for m in BASE:
                rows.append({**tag, "variant": m, **regression_metrics(yva, preds[m])})

            for method, label in (("ridge", "stack_ridge"), ("nnls", "stack_nnls")):
                sp, w = _stack(OOF, ytr, P, method)
                rows.append({**tag, "variant": label, **regression_metrics(yva, sp)})
                weight_rows.append({**tag, "method": label,
                                    **dict(zip(BASE, np.round(w, 4)))})

            # --- drop-one ensemble ablation --------------------------------
            for dropped in BASE:
                keep = [i for i, m in enumerate(BASE) if m != dropped]
                sp, _ = _stack(OOF[:, keep], ytr, P[:, keep], "nnls")
                drop_rows.append({**tag, "dropped": dropped,
                                  **regression_metrics(yva, sp)})

            frame = pd.DataFrame({**{m: preds[m] for m in BASE}, "true": yva})
            frame["stack_nnls"] = _stack(OOF, ytr, P, "nnls")[0]
            frame["repeat"], frame["fold"] = rep, fold
            oof_frames.append(frame)

            print(f"[rep {rep+1}/{repeats} fold {fold+1}/5] ({time.time()-ft:.0f}s)")

    df = pd.DataFrame(rows)
    oof_all = pd.concat(oof_frames, ignore_index=True)
    df.to_csv(config.RESULTS_METRICS / "stacking_fold_metrics.csv", index=False)
    oof_all.to_csv(config.RESULTS_METRICS / "stacking_oof_predictions.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(
        config.RESULTS_METRICS / "stacking_weights.csv", index=False)
    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_csv(config.RESULTS_METRICS / "stacking_drop_one.csv", index=False)

    summary: dict = {"repeats": repeats, "n_fold_estimates": repeats * 5}

    # --- drop-one significance ------------------------------------------
    piv_full = df[df.variant == "stack_nnls"].sort_values(["repeat", "fold"]).mse.to_numpy()
    drop_summary = {}
    for dropped in BASE:
        d = drop_df[drop_df.dropped == dropped].sort_values(["repeat", "fold"]).mse.to_numpy()
        loss = d - piv_full            # positive => removing this model hurts
        # Fold estimates come from repeated CV with overlapping training sets,
        # so the uncorrected paired t-test is anti-conservative here. The
        # Nadeau-Bengio corrected value is the one to report.
        n_tr = int(df.n_train.mean())
        n_va = int(df.n_valid.mean())
        drop_summary[dropped] = {
            "mean_r2_without": float(drop_df[drop_df.dropped == dropped].r2.mean()),
            "mse_penalty_when_removed": float(loss.mean()),
            "folds_worse_without": int((loss > 0).sum()),
            "n_folds": len(loss),
            "test": "Nadeau-Bengio corrected paired t-test",
            "p_one_sided": corrected_repeated_kfold_ttest(
                loss, n_train=n_tr, n_test=n_va).p_value_one_sided,
            "p_one_sided_plain": fold_level_ttest(loss).p_value_one_sided,
        }
    summary["drop_one_ablation"] = drop_summary

    # --- error correlation ----------------------------------------------
    err = pd.DataFrame({m: oof_all["true"] - oof_all[m] for m in BASE})
    summary["error_correlation"] = err.corr().round(4).to_dict()
    summary["mean_pairwise_error_correlation"] = float(
        err.corr().to_numpy()[np.triu_indices(len(BASE), 1)].mean())

    # --- learning curve --------------------------------------------------
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        for fold, (tr, va) in enumerate(GroupKFold(config.N_SPLITS)
                                        .split(X_dev, y_dev, g_dev)):
            rs = np.random.RandomState(config.SEED + fold)
            sub = rs.choice(tr, max(60, int(frac * len(tr))), replace=False)
            Xtr, Xva, _ = encode_fold(X_dev.iloc[sub], X_dev.iloc[va])
            ytr, yva = y_dev[sub], y_dev[va]
            preds, oof = _fit_fold(Xtr, ytr, Xva, config.SEED + fold, epochs)
            OOF = np.column_stack([oof[m] for m in BASE])
            P = np.column_stack([preds[m] for m in BASE])
            sp, _ = _stack(OOF, ytr, P, "nnls")
            curve.append({"fraction": frac, "n_train": len(sub), "fold": fold,
                          "stack_nnls": r2_score(yva, sp),
                          **{m: r2_score(yva, preds[m]) for m in BASE}})
        print(f"[learning curve] fraction {frac} done")
    pd.DataFrame(curve).to_csv(
        config.RESULTS_METRICS / "learning_curve.csv", index=False)

    # --- held-out test ---------------------------------------------------
    Xd, Xt, columns = encode_fold(X_dev, X_test)
    preds, oof = _fit_fold(Xd, y_dev, Xt, config.SEED, epochs)
    OOF = np.column_stack([oof[m] for m in BASE])
    P = np.column_stack([preds[m] for m in BASE])
    test_metrics = {m: regression_metrics(y_test, preds[m]) for m in BASE}
    for method, label in (("ridge", "stack_ridge"), ("nnls", "stack_nnls")):
        sp, w = _stack(OOF, y_dev, P, method)
        test_metrics[label] = regression_metrics(y_test, sp)
        test_metrics[label]["weights"] = dict(zip(BASE, np.round(w, 4).tolist()))
    summary["held_out_test"] = test_metrics

    # --- permutation importance for the strongest single model -----------
    best = max(BASE, key=lambda m: test_metrics[m]["r2"])
    if best in ("rf", "gbr", "svr"):
        est = train_baseline(best, Xd, y_dev, tune=False)
        imp = permutation_importance(est, Xt, y_test, n_repeats=10,
                                     random_state=config.SEED, scoring="r2")
        order = np.argsort(imp.importances_mean)[::-1][:15]
        summary["permutation_importance"] = {
            "model": best,
            "top_features": [{"feature": columns[i],
                              "importance": float(imp.importances_mean[i]),
                              "std": float(imp.importances_std[i])} for i in order],
        }

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(config.RESULTS_METRICS / "stacking_analysis.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== drop-one ensemble ablation (NNLS stack) ===")
    for m, e in sorted(drop_summary.items(),
                       key=lambda kv: -kv[1]["mse_penalty_when_removed"]):
        print(f"  remove {m:5} -> R2 {e['mean_r2_without']:.4f}  "
              f"MSE penalty {e['mse_penalty_when_removed']:+.4f}  "
              f"worse on {e['folds_worse_without']}/{e['n_folds']}  "
              f"p={e['p_one_sided']:.4f}")
    print(f"\nmean pairwise error correlation: "
          f"{summary['mean_pairwise_error_correlation']:.3f}")
    print("\n=== held-out test ===")
    for m, e in sorted(test_metrics.items(), key=lambda kv: -kv[1]["r2"]):
        print(f"  {m:12} R2={e['r2']:.4f}  MAE={e['mae']:.4f}")
    print(f"\nruntime {summary['runtime_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(args.repeats, args.quick)
