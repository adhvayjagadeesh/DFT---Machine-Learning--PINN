"""Is the ensemble worth its cost? Four decision-relevant tests beyond R^2.

A gain of Delta-R2 ~ 0.016 over the best single model is not self-evidently
worth a 5x training cost, and R^2 is not the quantity a screening campaign
cares about. This script re-expresses the comparison in terms a practitioner
can act on, reusing the saved out-of-fold predictions so no retraining is
needed except for the timing benchmark.

1. Model-selection risk. The best *single* model is not the same model on
   every fold. Committing to one in advance therefore carries a risk that the
   ensemble removes by construction. We quantify how often each model wins, the
   gap to an oracle that always picks the fold's best model, and the regret
   incurred by committing to one model chosen on other folds.

2. Screening utility. The deployed task is: rank candidate materials, run DFT
   on the top ones, keep those whose true gap falls in a target window. We
   report precision@k and the number of wasted DFT calculations per true hit,
   which converts Delta-R2 into CPU-hours saved.

3. Tail risk. Screening is harmed more by rare large errors than by average
   error. We report the 95th percentile absolute error and the rate of errors
   above 1 eV, which are large enough to move a material between application
   classes.

4. Cost accounting. Measured wall-clock training and inference time for every
   base model and for the ensemble, so the trade-off is stated honestly rather
   than assumed.

Run::

    python experiments/run_utility_study.py
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                    # noqa: E402
from pinn_dft.evaluation.statistics import fold_level_ttest    # noqa: E402

BASE = ["rf", "gbr", "svr", "mlp", "pinn"]
STACK = "stack_nnls"

#: Target windows a screening campaign would actually use.
WINDOWS = {
    "photovoltaic_1.0_2.0eV": (1.0, 2.0),
    "visible_photocatalysis_1.8_3.0eV": (1.8, 3.0),
    "wide_gap_uv_above_3.5eV": (3.5, np.inf),
}


# ---------------------------------------------------------------- 1
def model_selection_risk(oof: pd.DataFrame) -> dict:
    """How risky is committing to a single model chosen in advance?"""
    per_fold = []
    for (rep, fold), g in oof.groupby(["repeat", "fold"]):
        scores = {m: r2_score(g["true"], g[m]) for m in BASE}
        scores[STACK] = r2_score(g["true"], g[STACK])
        best_single = max(BASE, key=lambda m: scores[m])
        per_fold.append({
            "repeat": rep, "fold": fold, "best_single": best_single,
            "oracle_r2": scores[best_single], "stack_r2": scores[STACK],
            **{f"r2_{m}": scores[m] for m in BASE},
        })
    df = pd.DataFrame(per_fold)

    wins = df.best_single.value_counts().to_dict()
    # Committed choice: the model with the highest mean R2 across folds.
    means = {m: df[f"r2_{m}"].mean() for m in BASE}
    committed = max(means, key=means.get)

    # Regret of the committed model relative to the per-fold oracle.
    regret_committed = (df["oracle_r2"] - df[f"r2_{committed}"]).to_numpy()
    regret_stack = (df["oracle_r2"] - df["stack_r2"]).to_numpy()
    # Worst case over any single choice a practitioner might plausibly make.
    worst_choice = min(means, key=means.get)

    return {
        "n_folds": len(df),
        "win_counts": {m: int(wins.get(m, 0)) for m in BASE},
        "mean_r2_per_model": {m: float(means[m]) for m in BASE},
        "committed_model": committed,
        "mean_r2_committed": float(means[committed]),
        "mean_r2_stack": float(df.stack_r2.mean()),
        "mean_regret_committed_vs_oracle": float(regret_committed.mean()),
        "mean_regret_stack_vs_oracle": float(regret_stack.mean()),
        "worst_case_single_choice": worst_choice,
        "mean_r2_worst_choice": float(means[worst_choice]),
        "stack_advantage_over_worst_choice": float(
            df.stack_r2.mean() - means[worst_choice]),
        "folds_stack_beats_committed": int(
            (df.stack_r2 > df[f"r2_{committed}"]).sum()),
        "p_stack_vs_committed": fold_level_ttest(
            (df.stack_r2 - df[f"r2_{committed}"]).to_numpy()).p_value_one_sided,
    }


# ---------------------------------------------------------------- 2
def screening_utility(oof: pd.DataFrame) -> dict:
    """Precision@k and wasted DFT calculations per true hit."""
    out = {}
    for label, (lo, hi) in WINDOWS.items():
        truth = (oof["true"] >= lo) & (oof["true"] <= hi)
        base_rate = float(truth.mean())
        entry = {"window_eV": [lo, None if np.isinf(hi) else hi],
                 "base_rate": base_rate, "n_true_in_window": int(truth.sum()),
                 "models": {}}

        for m in BASE + [STACK]:
            # Rank by distance from window centre; for open-ended windows rank
            # by how far above the lower bound the prediction sits.
            pred = oof[m].to_numpy()
            if np.isinf(hi):
                score = pred - lo            # higher is better
                order = np.argsort(-score)
            else:
                centre = (lo + hi) / 2
                order = np.argsort(np.abs(pred - centre))

            hits = truth.to_numpy()[order]
            model_entry = {}
            for k in (50, 100, 200):
                if k > len(hits):
                    continue
                p_at_k = float(hits[:k].mean())
                model_entry[f"precision@{k}"] = p_at_k
                model_entry[f"enrichment@{k}"] = (
                    float(p_at_k / base_rate) if base_rate > 0 else None)
                # DFT calculations wasted per true hit found in the top k
                n_hits = hits[:k].sum()
                model_entry[f"wasted_dft_per_hit@{k}"] = (
                    float((k - n_hits) / n_hits) if n_hits > 0 else None)
            entry["models"][m] = model_entry
        out[label] = entry
    return out


# ---------------------------------------------------------------- 3
def tail_risk(oof: pd.DataFrame) -> dict:
    out = {}
    for m in BASE + [STACK]:
        err = np.abs(oof["true"] - oof[m])
        out[m] = {
            "mae": float(err.mean()),
            "p95_abs_error": float(np.percentile(err, 95)),
            "p99_abs_error": float(np.percentile(err, 99)),
            "max_abs_error": float(err.max()),
            "rate_error_above_1eV": float((err > 1.0).mean()),
            "rate_error_above_2eV": float((err > 2.0).mean()),
        }
    return out


# ---------------------------------------------------------------- 4
def cost_accounting(repeats: int = 1) -> dict:
    """Measured wall-clock cost of training and inference."""
    from pinn_dft.data import build_dataset, encode_fold
    from pinn_dft.models.baselines import train_baseline
    from pinn_dft.models.hybrid import out_of_fold_prior
    from pinn_dft.models.neural import predict, train_mlp, train_pinn
    from pinn_dft.utils import seed_everything

    seed_everything(config.SEED)
    X_df, y, groups = build_dataset()
    dev_idx, _ = next(GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_FRACTION,
        random_state=config.SEED).split(X_df, y, groups))
    X_dev, y_dev, g_dev = X_df.iloc[dev_idx], y[dev_idx], groups[dev_idx]
    tr, va = next(GroupKFold(config.N_SPLITS).split(X_dev, y_dev, g_dev))
    Xtr, Xva, _ = encode_fold(X_dev.iloc[tr], X_dev.iloc[va])
    ytr = y_dev[tr]

    costs = {}
    for name in ("rf", "gbr", "svr"):
        t = time.time()
        est = train_baseline(name, Xtr, ytr, tune=False)
        fit_s = time.time() - t
        t = time.time()
        est.predict(Xva)
        costs[name] = {"train_s": round(fit_s, 2),
                       "predict_s": round(time.time() - t, 4)}
        t = time.time()
        out_of_fold_prior(clone(est), Xtr, ytr, config.INNER_SPLITS, config.SEED)
        costs[name]["oof_for_stacking_s"] = round(time.time() - t, 2)

    for name, trainer in (("mlp", train_mlp), ("pinn", train_pinn)):
        t = time.time()
        net = trainer(Xtr, ytr, config.SEED, epochs=1000)
        fit_s = time.time() - t
        t = time.time()
        predict(net, Xva)
        costs[name] = {"train_s": round(fit_s, 2),
                       "predict_s": round(time.time() - t, 4)}
        t = time.time()
        for itr, iva in KFold(config.INNER_SPLITS, shuffle=True,
                              random_state=config.SEED).split(Xtr):
            predict(trainer(Xtr[itr], ytr[itr], config.SEED, epochs=1000),
                    Xtr[iva])
        costs[name]["oof_for_stacking_s"] = round(time.time() - t, 2)

    single_best = max(costs, key=lambda k: -costs[k]["train_s"])
    stack_train = sum(c["train_s"] + c["oof_for_stacking_s"] for c in costs.values())
    stack_predict = sum(c["predict_s"] for c in costs.values())
    return {
        "per_model": costs,
        "stack_total_train_s": round(stack_train, 2),
        "stack_total_predict_s": round(stack_predict, 4),
        "cheapest_single_model": single_best,
        "cost_ratio_vs_mlp": round(
            stack_train / max(costs["mlp"]["train_s"], 1e-9), 1),
    }


def main(skip_timing: bool) -> None:
    path = config.RESULTS_METRICS / "stacking_oof_predictions.csv"
    if not path.exists():
        raise SystemExit("run experiments/run_stacking_analysis.py first")
    oof = pd.read_csv(path)

    summary = {
        "model_selection_risk": model_selection_risk(oof),
        "screening_utility": screening_utility(oof),
        "tail_risk": tail_risk(oof),
    }
    if not skip_timing:
        summary["cost"] = cost_accounting()

    with open(config.RESULTS_METRICS / "utility_study.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    r = summary["model_selection_risk"]
    print("=== 1. model-selection risk ===")
    print(f"  best single model varies across {r['n_folds']} folds: {r['win_counts']}")
    print(f"  committing to '{r['committed_model']}' gives R2 {r['mean_r2_committed']:.4f}")
    print(f"  stack gives R2 {r['mean_r2_stack']:.4f} "
          f"(beats committed on {r['folds_stack_beats_committed']}/{r['n_folds']}, "
          f"p={r['p_stack_vs_committed']:.4f})")
    print(f"  regret vs per-fold oracle: committed {r['mean_regret_committed_vs_oracle']:+.4f}"
          f" | stack {r['mean_regret_stack_vs_oracle']:+.4f}")
    print(f"  worst plausible single choice ('{r['worst_case_single_choice']}') "
          f"loses {r['stack_advantage_over_worst_choice']:.4f} R2 vs the stack")

    print("\n=== 2. screening utility (wasted DFT runs per true hit, top 100) ===")
    for label, e in summary["screening_utility"].items():
        row = {m: e["models"][m].get("wasted_dft_per_hit@100")
               for m in BASE + [STACK]}
        best = min((v for v in row.values() if v is not None), default=None)
        print(f"  {label} (base rate {e['base_rate']:.2f}):")
        for m, v in row.items():
            mark = " <-- best" if v == best else ""
            print(f"      {m:11} {v:.3f}{mark}" if v is not None else f"      {m:11} n/a")

    print("\n=== 3. tail risk ===")
    for m, e in summary["tail_risk"].items():
        print(f"  {m:11} p95={e['p95_abs_error']:.3f} eV  "
              f">1eV rate={e['rate_error_above_1eV']:.3%}  max={e['max_abs_error']:.2f}")

    if "cost" in summary:
        c = summary["cost"]
        print("\n=== 4. cost ===")
        for m, e in c["per_model"].items():
            print(f"  {m:11} train {e['train_s']:6.2f}s  "
                  f"+OOF {e['oof_for_stacking_s']:6.2f}s  "
                  f"predict {e['predict_s']:.4f}s")
        print(f"  stack total train {c['stack_total_train_s']}s "
              f"({c['cost_ratio_vs_mlp']}x the MLP alone), "
              f"predict {c['stack_total_predict_s']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    main(args.skip_timing)
