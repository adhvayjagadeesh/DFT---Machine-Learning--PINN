"""Component ablation for the hybrid architecture.

Every configuration below is trained and evaluated. This replaces the previous
``ablation.py``, which reported ``optimized_hybrid - numpy.random.uniform(...)``
for each variant and raised ``KeyError`` before it could have trained anything;
the numbers it produced were the source of Table 2 in the manuscript draft and
are not measurements.

Run::

    python experiments/run_ablation.py --splitter group
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                     # noqa: E402
from pinn_dft.data import build_dataset, encode_fold, geometric_indices  # noqa: E402
from pinn_dft.evaluation.metrics import regression_metrics      # noqa: E402
from pinn_dft.evaluation.statistics import (                   # noqa: E402
    corrected_repeated_kfold_ttest, fold_level_ttest)
from pinn_dft.models.baselines import train_baseline            # noqa: E402
from pinn_dft.models.hybrid import (                            # noqa: E402
    HybridConfig, predict_hybrid, train_hybrid)
from pinn_dft.utils import seed_everything                      # noqa: E402

#: name -> (config, what the variant tests)
VARIANTS: dict[str, tuple[HybridConfig, str]] = {
    "full": (
        HybridConfig(),
        "complete framework",
    ),
    "no_structural_layer": (
        HybridConfig(use_structural_layer=False),
        "removes the forward geometric coupling layer, keeps its loss term",
    ),
    "no_quantile_heads": (
        HybridConfig(use_quantile_heads=False),
        "point residual head only, no quantile outputs or pinball loss",
    ),
    "no_residual_head": (
        HybridConfig(use_residual_head=False),
        "correction driven solely by the quantile heads",
    ),
    "no_physics_loss": (
        HybridConfig(use_physics_loss=False),
        "removes the non-negativity boundary penalty",
    ),
    "no_anisotropy_loss": (
        HybridConfig(use_anisotropy_loss=False),
        "removes the aspect-ratio coupling penalty",
    ),
    "in_sample_prior": (
        HybridConfig(out_of_fold_prior=False),
        "reverts the leakage fix: trains the head on in-sample tree predictions",
    ),
}


def run(splitter_kind: str, quick: bool, repeats: int = 5) -> None:
    seed_everything(config.SEED)
    t_start = time.time()

    X_df, y, groups = build_dataset()
    dev_idx, _ = next(GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_FRACTION,
        random_state=config.SEED).split(X_df, y, groups))
    X_dev, y_dev, g_dev = X_df.iloc[dev_idx], y[dev_idx], groups[dev_idx]

    cv = (GroupKFold(n_splits=config.N_SPLITS) if splitter_kind == "group"
          else KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED))

    names = list(VARIANTS) + ["gbr_prior_only"]
    per_fold: dict[str, list[dict]] = {name: [] for name in names}
    estimate = -1

    for rep in range(repeats):
        # GroupKFold is deterministic, so permute group identity per repeat to
        # obtain independent partitions -- the same construction used by the
        # other experiments, so all tables rest on one protocol.
        rng = np.random.RandomState(config.SEED + rep)
        uniq = np.unique(g_dev)
        remap = dict(zip(uniq, rng.permutation(len(uniq))))
        g_rep = np.array([remap[g] for g in g_dev])

        for tr, va in cv.split(X_dev, y_dev, g_rep):
            estimate += 1
            t0 = time.time()
            Xtr, Xva, columns = encode_fold(X_dev.iloc[tr], X_dev.iloc[va])
            ytr, yva = y_dev[tr], y_dev[va]
            ch_a, ch_b = geometric_indices(columns)
            seed = config.SEED + 100 * rep + estimate
            tag = {"repeat": rep, "fold": estimate,
                   "n_train": len(tr), "n_valid": len(va)}

            base = clone(train_baseline("gbr", Xtr, ytr, tune=False))
            prior_pred = clone(base).fit(Xtr, ytr.ravel()).predict(Xva)
            per_fold["gbr_prior_only"].append(
                {**tag, **regression_metrics(yva, prior_pred)})

            for name, (cfg, _) in VARIANTS.items():
                model, va_feats, _ = train_hybrid(
                    base, Xtr, ytr, Xva, ch_a, ch_b, cfg, seed,
                    epochs=200 if quick else 1000)
                point, _ = predict_hybrid(model, va_feats)
                per_fold[name].append({**tag, **regression_metrics(yva, point)})

            print(f"[estimate {estimate + 1}/{repeats * config.N_SPLITS}] "
                  f"({time.time() - t0:.0f}s)")

    # --- assemble ------------------------------------------------------------
    frames = {n: pd.DataFrame(v).sort_values("fold") for n, v in per_fold.items()}
    pd.concat([d.assign(variant=n) for n, d in frames.items()]).to_csv(
        config.RESULTS_METRICS / "ablation_fold_metrics.csv", index=False)

    baseline_mse = frames["gbr_prior_only"].mse.to_numpy()
    n_tr = int(frames["gbr_prior_only"].n_train.mean())
    n_va = int(frames["gbr_prior_only"].n_valid.mean())

    rows = []
    for name, df in frames.items():
        entry = {
            "variant": name,
            "description": VARIANTS[name][1] if name in VARIANTS
                           else "tree prior, no neural correction",
            "n_estimates": len(df),
            "r2_mean": float(df.r2.mean()),
            "r2_std": float(df.r2.std(ddof=1)),
            "mae_mean": float(df.mae.mean()),
        }
        if name != "gbr_prior_only":
            diffs = baseline_mse - df.mse.to_numpy()
            entry.update({
                "mean_mse_gain_vs_prior": float(diffs.mean()),
                "estimates_improved_vs_prior": int((diffs > 0).sum()),
                "test": "Nadeau-Bengio corrected paired t-test",
                "p_two_sided_vs_prior": corrected_repeated_kfold_ttest(
                    diffs, n_train=n_tr, n_test=n_va).p_value_two_sided,
            })
        rows.append(entry)

    full_r2 = next(r["r2_mean"] for r in rows if r["variant"] == "full")
    for row in rows:
        row["delta_r2_vs_full"] = row["r2_mean"] - full_r2

    out = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    out.to_csv(config.RESULTS_METRICS / "ablation_results.csv", index=False)
    with open(config.RESULTS_METRICS / "ablation_results.json", "w") as fh:
        json.dump({"runtime_seconds": round(time.time() - t_start, 1),
                   "splitter": splitter_kind, "repeats": repeats,
                   "n_estimates": repeats * config.N_SPLITS,
                   "variants": rows}, fh, indent=2)

    print(f"\n=== ablation, {repeats * config.N_SPLITS} fold estimates ===")
    for _, r in out.iterrows():
        print(f"  {r['variant']:22} R2={r['r2_mean']:.4f}+/-{r['r2_std']:.4f}  "
              f"dR2 vs full={r['delta_r2_vs_full']:+.4f}  MAE={r['mae_mean']:.4f}")
    print(f"\nruntime {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splitter", choices=["group", "random"], default="group")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    run(args.splitter, args.quick, args.repeats)
