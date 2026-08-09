"""Can the hybrid be repaired? Four candidate remedies, measured under repeated CV.

The corrected benchmark shows the hybrid falling below its own gradient-boosted
prior. This script tests the four remedies that follow from *why* that happens,
and uses repeated grouped cross-validation (5 repeats x 5 folds = 25 fold
estimates) because 5 folds gave too little power to separate any two models.

Candidates
----------
gate        Shrinkage-gated hybrid. The correction is scaled by a learned
            scalar initialised at zero, so the model starts as a pass-through of
            the prior and must earn any departure from it.
strong      Hybrid built on the *strongest* base learner rather than a tree.
            The original design assumed trees dominate neural networks on
            tabular materials data; with composition features that premise no
            longer holds here, which inverts the argument for a tree prior.
stack_ridge Proper stacked generalisation: a ridge meta-learner over the
            out-of-fold predictions of all five base models.
stack_nnls  Same, constrained to non-negative weights summing freely -- an
            interpretable ensemble that cannot extrapolate outside the base
            models' span.

Run::

    python experiments/run_recovery_study.py --repeats 5
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                    # noqa: E402
from pinn_dft.data import build_dataset, encode_fold, geometric_indices  # noqa: E402
from pinn_dft.evaluation.metrics import regression_metrics     # noqa: E402
from pinn_dft.evaluation.statistics import (                   # noqa: E402
    corrected_repeated_kfold_ttest, fold_level_ttest)
from pinn_dft.models.baselines import train_baseline           # noqa: E402
from pinn_dft.models.hybrid import (                           # noqa: E402
    HybridConfig, out_of_fold_prior, predict_hybrid, train_hybrid)
from pinn_dft.models.neural import predict, train_mlp, train_pinn  # noqa: E402
from pinn_dft.utils import seed_everything                     # noqa: E402

BASE_MODELS = ["rf", "gbr", "svr", "mlp", "pinn"]


def _stack_weights(oof: np.ndarray, y: np.ndarray, method: str):
    if method == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(oof, y)
        return lambda P: model.predict(P), dict(
            zip(BASE_MODELS, np.round(model.coef_, 4).tolist()))
    weights, _ = nnls(oof, y)
    return lambda P: P @ weights, dict(zip(BASE_MODELS, np.round(weights, 4).tolist()))


def run(repeats: int, quick: bool) -> None:
    seed_everything(config.SEED)
    t_start = time.time()

    X_df, y, groups = build_dataset()
    dev_idx, _ = next(GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_FRACTION,
        random_state=config.SEED).split(X_df, y, groups))
    X_dev, y_dev, g_dev = X_df.iloc[dev_idx], y[dev_idx], groups[dev_idx]

    records: list[dict] = []
    stack_weight_log: list[dict] = []
    epochs = 200 if quick else 1000

    for rep in range(repeats):
        # GroupKFold is deterministic, so shuffle group identity per repeat.
        rng = np.random.RandomState(config.SEED + rep)
        uniq = np.unique(g_dev)
        remap = dict(zip(uniq, rng.permutation(len(uniq))))
        g_shuffled = np.array([remap[g] for g in g_dev])

        cv = GroupKFold(n_splits=config.N_SPLITS)
        for fold, (tr, va) in enumerate(cv.split(X_dev, y_dev, g_shuffled)):
            t0 = time.time()
            Xtr, Xva, columns = encode_fold(X_dev.iloc[tr], X_dev.iloc[va])
            ytr, yva = y_dev[tr], y_dev[va]
            ch_a, ch_b = geometric_indices(columns)
            seed = config.SEED + 100 * rep + fold
            tag = {"repeat": rep, "fold": fold, "n_train": len(tr), "n_valid": len(va)}

            # --- base learners -------------------------------------------
            fitted, preds, oof_cols = {}, {}, {}
            for name in ("rf", "gbr", "svr"):
                est = train_baseline(name, Xtr, ytr, tune=False)
                fitted[name] = est
                preds[name] = est.predict(Xva)
                oof_cols[name] = out_of_fold_prior(
                    clone(est), Xtr, ytr, config.INNER_SPLITS, seed)

            for name, trainer in (("mlp", train_mlp), ("pinn", train_pinn)):
                net = trainer(Xtr, ytr, seed, epochs=epochs)
                preds[name] = predict(net, Xva)
                # inner-split OOF for the neural models, for the stackers
                inner = np.zeros(len(Xtr))
                from sklearn.model_selection import KFold
                for itr, iva in KFold(config.INNER_SPLITS, shuffle=True,
                                      random_state=seed).split(Xtr):
                    inner[iva] = predict(
                        trainer(Xtr[itr], ytr[itr], seed, epochs=epochs), Xtr[iva])
                oof_cols[name] = inner

            for name in BASE_MODELS:
                records.append({**tag, "variant": name,
                                **regression_metrics(yva, preds[name])})

            # --- candidate 1: shrinkage-gated hybrid on the GBR prior -----
            model, va_feats, _ = train_hybrid(
                clone(fitted["gbr"]), Xtr, ytr, Xva, ch_a, ch_b,
                HybridConfig(use_shrinkage_gate=True), seed, epochs=epochs)
            point, _ = predict_hybrid(model, va_feats)
            gate = float(model.correction_gate.detach().item())  # sigmoid-constrained, in (0, 1)
            records.append({**tag, "variant": "hybrid_gated",
                            "gate_value": gate, **regression_metrics(yva, point)})

            # --- candidate 2: ungated hybrid, for reference ---------------
            model, va_feats, _ = train_hybrid(
                clone(fitted["gbr"]), Xtr, ytr, Xva, ch_a, ch_b,
                HybridConfig(), seed, epochs=epochs)
            point, _ = predict_hybrid(model, va_feats)
            records.append({**tag, "variant": "hybrid_ungated",
                            **regression_metrics(yva, point)})

            # --- candidate 3/4: stacked ensembles -------------------------
            OOF = np.column_stack([oof_cols[m] for m in BASE_MODELS])
            P = np.column_stack([preds[m] for m in BASE_MODELS])
            for method, label in (("ridge", "stack_ridge"), ("nnls", "stack_nnls")):
                fn, weights = _stack_weights(OOF, ytr, method)
                records.append({**tag, "variant": label,
                                **regression_metrics(yva, fn(P))})
                stack_weight_log.append({**tag, "method": label, **weights})

            print(f"[rep {rep + 1}/{repeats} fold {fold + 1}/{config.N_SPLITS}] "
                  f"gate={gate:+.3f} ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(records)
    df.to_csv(config.RESULTS_METRICS / "recovery_fold_metrics.csv", index=False)
    pd.DataFrame(stack_weight_log).to_csv(
        config.RESULTS_METRICS / "recovery_stack_weights.csv", index=False)

    # --- comparison against the GBR baseline -----------------------------
    piv = df.pivot_table(index=["repeat", "fold"], columns="variant", values="mse")
    n_tr = int(df.n_train.mean())
    n_va = int(df.n_valid.mean())
    summary = {"n_fold_estimates": len(piv), "repeats": repeats,
               "runtime_seconds": round(time.time() - t_start, 1), "variants": {}}

    for variant in piv.columns:
        d = (piv["gbr"] - piv[variant]).to_numpy()
        entry = {
            "mean_r2": float(df[df.variant == variant].r2.mean()),
            "std_r2": float(df[df.variant == variant].r2.std(ddof=1)),
            "mean_mae": float(df[df.variant == variant].mae.mean()),
            "mean_mse_gain_vs_gbr": float(d.mean()),
            "folds_better_than_gbr": int((d > 0).sum()),
            "n_folds": len(d),
        }
        if variant != "gbr":
            entry["p_plain"] = fold_level_ttest(d).p_value_one_sided
            entry["p_nadeau_bengio"] = corrected_repeated_kfold_ttest(
                d, n_train=n_tr, n_test=n_va).p_value_one_sided
        summary["variants"][variant] = entry

    if "gate_value" in df.columns:
        gates = df[df.variant == "hybrid_gated"].gate_value.dropna()
        summary["shrinkage_gate"] = {
            "mean": float(gates.mean()), "std": float(gates.std(ddof=1)),
            "min": float(gates.min()), "max": float(gates.max()),
        }

    with open(config.RESULTS_METRICS / "recovery_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== recovery study ({len(piv)} fold estimates) ===")
    order = sorted(summary["variants"], key=lambda v: -summary["variants"][v]["mean_r2"])
    for v in order:
        e = summary["variants"][v]
        p = f"p={e['p_nadeau_bengio']:.3f}" if "p_nadeau_bengio" in e else "baseline"
        print(f"  {v:16} R2={e['mean_r2']:.4f}+/-{e['std_r2']:.4f}  "
              f"MAE={e['mean_mae']:.4f}  better/{e['n_folds']}="
              f"{e['folds_better_than_gbr']:2d}  {p}")
    if "shrinkage_gate" in summary:
        g = summary["shrinkage_gate"]
        print(f"\nlearned correction gate: {g['mean']:+.4f} +/- {g['std']:.4f} "
              f"(range {g['min']:+.3f} to {g['max']:+.3f})")
    print(f"runtime {summary['runtime_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(args.repeats, args.quick)
