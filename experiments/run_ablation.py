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
from pinn_dft.evaluation.statistics import fold_level_ttest     # noqa: E402
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


def run(splitter_kind: str, quick: bool) -> None:
    seed_everything(config.SEED)
    t_start = time.time()

    X_df, y, groups = build_dataset()
    dev_idx, _ = next(GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_FRACTION,
        random_state=config.SEED).split(X_df, y, groups))
    X_dev, y_dev, g_dev = X_df.iloc[dev_idx], y[dev_idx], groups[dev_idx]

    cv = (GroupKFold(n_splits=config.N_SPLITS) if splitter_kind == "group"
          else KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED))

    per_fold: dict[str, list[dict]] = {name: [] for name in VARIANTS}
    per_fold["gbr_prior_only"] = []
    pooled_pred: dict[str, np.ndarray] = {k: np.zeros(len(X_dev)) for k in per_fold}
    y_pooled = np.zeros(len(X_dev))

    for fold, (tr, va) in enumerate(cv.split(X_dev, y_dev, g_dev)):
        t0 = time.time()
        Xtr, Xva, columns = encode_fold(X_dev.iloc[tr], X_dev.iloc[va])
        ytr, yva = y_dev[tr], y_dev[va]
        ch_a, ch_b = geometric_indices(columns)
        seed = config.SEED + fold
        y_pooled[va] = yva

        base = clone(train_baseline("gbr", Xtr, ytr, tune=False))
        prior_pred = clone(base).fit(Xtr, ytr.ravel()).predict(Xva)
        per_fold["gbr_prior_only"].append(
            {"fold": fold, **regression_metrics(yva, prior_pred)})
        pooled_pred["gbr_prior_only"][va] = prior_pred

        for name, (cfg, _) in VARIANTS.items():
            model, va_feats, _ = train_hybrid(
                base, Xtr, ytr, Xva, ch_a, ch_b, cfg, seed,
                epochs=200 if quick else 1000)
            point, _ = predict_hybrid(model, va_feats)
            per_fold[name].append({"fold": fold, **regression_metrics(yva, point)})
            pooled_pred[name][va] = point

        print(f"[fold {fold + 1}/{config.N_SPLITS}] done ({time.time() - t0:.0f}s)")

    # --- assemble ------------------------------------------------------------
    from sklearn.metrics import r2_score

    rows, table = [], {}
    baseline_mse = np.array([r["mse"] for r in per_fold["gbr_prior_only"]])
    for name, folds in per_fold.items():
        df = pd.DataFrame(folds)
        pooled = regression_metrics(y_pooled, pooled_pred[name])
        entry = {
            "variant": name,
            "description": VARIANTS[name][1] if name in VARIANTS else "tree prior, no neural correction",
            "pooled_r2": pooled["r2"],
            "pooled_mse": pooled["mse"],
            "pooled_mae": pooled["mae"],
            "fold_r2_mean": float(df.r2.mean()),
            "fold_r2_std": float(df.r2.std(ddof=1)),
        }
        if name != "gbr_prior_only":
            diffs = baseline_mse - df.sort_values("fold").mse.to_numpy()
            test = fold_level_ttest(diffs)
            entry.update({
                "mean_mse_gain_vs_prior": test.mean_difference,
                "folds_improved_vs_prior": int((diffs > 0).sum()),
                "p_one_sided_vs_prior": test.p_value_one_sided,
            })
        rows.append(entry)
        table[name] = entry

    # deltas relative to the full framework
    full_r2 = table["full"]["pooled_r2"]
    for row in rows:
        row["delta_r2_vs_full"] = row["pooled_r2"] - full_r2

    out = pd.DataFrame(rows).sort_values("pooled_r2", ascending=False)
    out.to_csv(config.RESULTS_METRICS / "ablation_results.csv", index=False)
    with open(config.RESULTS_METRICS / "ablation_results.json", "w") as fh:
        json.dump({"runtime_seconds": round(time.time() - t_start, 1),
                   "splitter": splitter_kind, "variants": rows}, fh, indent=2)

    print("\n=== ablation (measured) ===")
    for _, r in out.iterrows():
        print(f"  {r['variant']:22} R2={r['pooled_r2']:.4f}  "
              f"dR2 vs full={r['delta_r2_vs_full']:+.4f}  MAE={r['pooled_mae']:.4f}")
    print(f"\nruntime {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splitter", choices=["group", "random"], default="group")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(args.splitter, args.quick)
