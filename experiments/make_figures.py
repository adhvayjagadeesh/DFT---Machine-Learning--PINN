"""Regenerate every manuscript figure from the committed result files.

Run after ``run_benchmarks.py`` and ``run_ablation.py``::

    python experiments/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_dft import config                                # noqa: E402
from pinn_dft.evaluation.metrics import rec_curve          # noqa: E402

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

ACCENT, NEUTRAL, WARN = "#2E6FD9", "#6B7280", "#C2453B"


def _load():
    oof = pd.read_csv(config.RESULTS_METRICS / "predictions_oof.csv")
    with open(config.RESULTS_METRICS / "benchmark_summary.json") as fh:
        summary = json.load(fh)
    return oof, summary


def figure_parity(oof, summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharex=True, sharey=True)
    lim = (0, max(oof["true"].max(), oof["hybrid_gbr"].max()) * 1.05)
    for ax, model, label in zip(axes, ["gbr", "hybrid_gbr"],
                                ["Gradient boosting baseline", "Hybrid GBR + correction head"]):
        m = summary["pooled_out_of_fold"][model]
        ax.scatter(oof["true"], oof[model], s=9, alpha=.35,
                   color=NEUTRAL if model == "gbr" else ACCENT, edgecolors="none")
        ax.plot(lim, lim, ls="--", lw=1.1, color="#333")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("HSE06 band gap (eV)")
        ax.set_title(f"{label}\n$R^2$={m['r2']:.4f}  MAE={m['mae']:.3f} eV", fontsize=10)
    axes[0].set_ylabel("Predicted band gap (eV)")
    fig.savefig(config.RESULTS_FIGURES / "fig_parity.png")
    plt.close(fig)


def figure_rec(oof):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for model, label, color in (("pinn", "Standalone PINN", "#9CA3AF"),
                                ("gbr", "GBR baseline", NEUTRAL),
                                ("hybrid_gbr", "Hybrid (ours)", ACCENT)):
        tol, acc = rec_curve(oof["true"], oof[model])
        ax.plot(tol, acc, lw=1.8, label=label, color=color)
    ax.set_xlabel("Absolute error tolerance $\\tau$ (eV)")
    ax.set_ylabel("Fraction of materials within $\\tau$")
    ax.set_title("Regression error characteristic", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(config.RESULTS_FIGURES / "fig_rec_curve.png")
    plt.close(fig)


def figure_fold_significance(summary):
    diffs = np.array(summary["significance"]["fold_mse_differences"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(np.arange(1, len(diffs) + 1), diffs, width=.6,
           color=[ACCENT if d > 0 else WARN for d in diffs], edgecolor="white")
    ax.axhline(0, color="#333", lw=1)
    ax.axhline(diffs.mean(), ls="--", lw=1.2, color="#1a7a3c",
               label=f"mean {diffs.mean():+.4f}")
    nb = summary["significance"]["nadeau_bengio_corrected"]
    ax.set_xticks(np.arange(1, len(diffs) + 1),
                  [f"Fold {i}" for i in range(1, len(diffs) + 1)])
    ax.set_ylabel("MSE reduction vs GBR (eV$^2$)")
    ax.set_title(f"Per-fold improvement — Nadeau-Bengio $p$={nb['p_value_one_sided']:.3f} "
                 f"(one-sided)", fontsize=10.5)
    ax.legend(frameon=False)
    fig.savefig(config.RESULTS_FIGURES / "fig_fold_significance.png")
    plt.close(fig)


def figure_ablation():
    path = config.RESULTS_METRICS / "ablation_results.csv"
    if not path.exists():
        print("[figures] ablation_results.csv missing - run run_ablation.py first")
        return
    df = pd.read_csv(path).sort_values("pooled_r2")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = [ACCENT if v == "full" else NEUTRAL for v in df["variant"]]
    ax.barh(np.arange(len(df)), df["pooled_r2"], color=colors, edgecolor="white", height=.68)
    for i, (v, r) in enumerate(zip(df["variant"], df["pooled_r2"])):
        ax.text(r + .003, i, f"{r:.4f}", va="center", fontsize=8.5)
    ax.set_yticks(np.arange(len(df)), df["variant"], fontsize=8.5)
    ax.set_xlabel("Pooled out-of-fold $R^2$")
    ax.set_xlim(0, df["pooled_r2"].max() * 1.12)
    ax.set_title("Component ablation (measured)", fontsize=11)
    fig.savefig(config.RESULTS_FIGURES / "fig_ablation.png")
    plt.close(fig)


def figure_error_vs_aspect(summary):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model, label, color in (("gbr", "GBR baseline", NEUTRAL),
                                ("hybrid_gbr", "Hybrid (ours)", ACCENT)):
        bins = summary["error_vs_aspect_ratio"][model]
        centres = [(b["covariate_low"] + b["covariate_high"]) / 2 for b in bins]
        ax.plot(centres, [b["mae"] for b in bins], "o-", label=label, color=color, lw=1.8)
    ax.set_xlabel("Structural aspect ratio (area / thickness, standardised)")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("Error versus structural distortion", fontsize=11)
    ax.legend(frameon=False)
    fig.savefig(config.RESULTS_FIGURES / "fig_error_vs_aspect.png")
    plt.close(fig)


def main() -> None:
    oof, summary = _load()
    figure_parity(oof, summary)
    figure_rec(oof)
    figure_fold_significance(summary)
    figure_error_vs_aspect(summary)
    figure_ablation()
    print(f"[figures] written to {config.RESULTS_FIGURES}")


if __name__ == "__main__":
    main()
