"""Regenerate every manuscript figure from the committed result files.

Run after run_benchmarks.py, run_ablation.py, run_recovery_study.py and
run_stacking_analysis.py::

    python experiments/make_figures.py

Figures are written to results/figures/ at 300 dpi and are also copied into the
LaTeX tree if one is present.
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

from pinn_dft import config                            # noqa: E402
from pinn_dft.evaluation.metrics import rec_curve      # noqa: E402

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": .6,
    "axes.axisbelow": True, "savefig.bbox": "tight", "figure.facecolor": "white",
})

ACCENT, NEUTRAL, WARN, GOOD = "#2E6FD9", "#6B7280", "#C2453B", "#1A7A3C"
PRETTY = {"rf": "Random forest", "gbr": "GBR", "svr": "SVR", "mlp": "MLP",
          "pinn": "PINN", "stack_ridge": "Stack (ridge)", "stack_nnls": "Stack (NNLS)",
          "hybrid_ungated": "Hybrid residual", "hybrid_gated": "Hybrid + gate"}

M = config.RESULTS_METRICS
F = config.RESULTS_FIGURES


def _exists(*names) -> bool:
    missing = [n for n in names if not (M / n).exists()]
    if missing:
        print(f"[figures] skipping - missing {missing}")
    return not missing


# ---------------------------------------------------------------- figure 1
def fig_model_comparison():
    """Forest plot: mean fold R2 with 95% CI across repeated CV estimates."""
    if not _exists("recovery_fold_metrics.csv"):
        return
    df = pd.read_csv(M / "recovery_fold_metrics.csv")
    stats = df.groupby("variant").r2.agg(["mean", "std", "count"])
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ypos = np.arange(len(stats))
    colors = [ACCENT if "stack" in v else (WARN if "hybrid" in v else NEUTRAL)
              for v in stats.index]
    ax.errorbar(stats["mean"], ypos, xerr=1.96 * stats["se"], fmt="o",
                ecolor="#999", elinewidth=1.2, capsize=3, markersize=0, zorder=1)
    ax.scatter(stats["mean"], ypos, s=46, c=colors, zorder=2, edgecolors="white")

    gbr = stats.loc["gbr", "mean"]
    ax.axvline(gbr, ls="--", lw=1, color=NEUTRAL, zorder=0)
    ax.text(gbr - .004, -0.62, "GBR baseline", fontsize=7.5, color=NEUTRAL,
            ha="right", va="center")
    ax.set_ylim(-1.0, len(stats) - 0.4)

    for i, (v, r) in enumerate(zip(stats.index, stats["mean"])):
        ax.text(r + 1.96 * stats.loc[v, "se"] + .006, i, f"{r:.3f}",
                va="center", fontsize=7.5, color="#444")
    ax.set_yticks(ypos, [PRETTY.get(v, v) for v in stats.index])
    ax.set_xlabel("Cross-validated $R^2$ (mean $\\pm$ 95% CI, 25 fold estimates)")
    ax.set_title("Model comparison under grouped repeated cross-validation",
                 fontsize=10, fontweight="bold")
    fig.savefig(F / "fig1_model_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_drop_one_and_weights():
    """Stacking weights next to the drop-one ablation that actually tests them."""
    if not _exists("stacking_analysis.json", "stacking_weights.csv"):
        return
    with open(M / "stacking_analysis.json") as fh:
        s = json.load(fh)
    w = pd.read_csv(M / "stacking_weights.csv")
    base = [c for c in ["rf", "gbr", "svr", "mlp", "pinn"] if c in w.columns]
    mean_w = w[w.method == "stack_nnls"][base].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    axes[0].bar(range(len(base)), [mean_w[b] for b in base],
                color=NEUTRAL, edgecolor="white", width=.62)
    axes[0].set_xticks(range(len(base)), [PRETTY[b] for b in base],
                       rotation=20, ha="right")
    axes[0].set_ylabel("Mean NNLS weight")
    axes[0].set_title("(a) Ensemble weights", fontsize=9.5, fontweight="bold")

    drop = s["drop_one_ablation"]
    order = sorted(base, key=lambda m: drop[m]["mse_penalty_when_removed"])
    penalties = [drop[m]["mse_penalty_when_removed"] for m in order]
    colors = [GOOD if p > 0 else WARN for p in penalties]
    axes[1].barh(range(len(order)), penalties, color=colors,
                 edgecolor="white", height=.62)
    axes[1].axvline(0, color="#333", lw=1)
    span = max(penalties) - min(penalties)
    axes[1].set_xlim(min(penalties) - 0.12 * span, max(penalties) + 0.34 * span)
    # Annotations sit in a fixed right-hand column so they never collide with
    # the tick labels of near-zero bars.
    label_x = max(penalties) + 0.10 * span
    for i, m in enumerate(order):
        d = drop[m]
        axes[1].text(label_x, i, f"{d['folds_worse_without']}/{d['n_folds']}"
                     f"  p={d['p_one_sided']:.3f}", va="center", ha="left",
                     fontsize=7)
    short = {"rf": "RF", "gbr": "GBR", "svr": "SVR", "mlp": "MLP", "pinn": "PINN"}
    axes[1].set_yticks(range(len(order)), [short[m] for m in order])
    axes[1].set_xlabel("MSE penalty when removed (eV$^2$)")
    axes[1].set_title("(b) Drop-one ablation", fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.28)
    fig.text(0.5, -0.04, "A large weight does not establish a contribution: "
             "collinear members split weight between them. Panel (b) is the test.",
             ha="center", fontsize=7.5, color="#666")
    fig.savefig(F / "fig2_ensemble_contributions.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_error_correlation():
    """Error-correlation heatmap - the diversity stacking exploits."""
    if not _exists("stacking_analysis.json"):
        return
    with open(M / "stacking_analysis.json") as fh:
        s = json.load(fh)
    corr = pd.DataFrame(s["error_correlation"])
    order = ["rf", "gbr", "svr", "mlp", "pinn"]
    corr = corr.loc[order, order]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), [PRETTY[m] for m in order],
                  rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), [PRETTY[m] for m in order], fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.75 else "#222")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.8, label="Pearson correlation of residuals")
    ax.set_title(f"Residual correlation (mean off-diagonal "
                 f"{s['mean_pairwise_error_correlation']:.2f})",
                 fontsize=9.5, fontweight="bold")
    fig.savefig(F / "fig3_error_correlation.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_learning_curve():
    if not _exists("learning_curve.csv"):
        return
    df = pd.read_csv(M / "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for model, color, lw in (("stack_nnls", ACCENT, 2.0), ("mlp", "#7C3AED", 1.4),
                             ("gbr", NEUTRAL, 1.4), ("rf", "#9CA3AF", 1.2)):
        if model not in df.columns:
            continue
        g = df.groupby("n_train")[model].agg(["mean", "std", "count"])
        se = g["std"] / np.sqrt(g["count"])
        ax.plot(g.index, g["mean"], "o-", color=color, lw=lw,
                label=PRETTY.get(model, model), markersize=4)
        ax.fill_between(g.index, g["mean"] - 1.96 * se, g["mean"] + 1.96 * se,
                        color=color, alpha=.12)
    ax.set_xlabel("Training-set size (materials)")
    ax.set_ylabel("Cross-validated $R^2$")
    ax.set_title("Learning curves", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(F / "fig4_learning_curve.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_parity_and_rec():
    if not _exists("stacking_oof_predictions.csv"):
        return
    oof = pd.read_csv(M / "stacking_oof_predictions.csv")
    oof = oof[oof.repeat == oof.repeat.min()]
    from sklearn.metrics import mean_absolute_error, r2_score

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    lim = (0, max(oof["true"].max(), oof["stack_nnls"].max()) * 1.04)
    for ax, model, color in ((axes[0], "gbr", NEUTRAL), (axes[1], "stack_nnls", ACCENT)):
        ax.scatter(oof["true"], oof[model], s=7, alpha=.32, color=color,
                   edgecolors="none")
        ax.plot(lim, lim, ls="--", lw=1, color="#333")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("HSE06 band gap (eV)")
        ax.set_title(f"{PRETTY[model]}\n$R^2$={r2_score(oof['true'], oof[model]):.3f}  "
                     f"MAE={mean_absolute_error(oof['true'], oof[model]):.3f} eV",
                     fontsize=9)
    axes[0].set_ylabel("Predicted band gap (eV)")

    for model, color, lw in (("pinn", "#9CA3AF", 1.2), ("gbr", NEUTRAL, 1.4),
                             ("mlp", "#7C3AED", 1.4), ("stack_nnls", ACCENT, 2.0)):
        tol, acc = rec_curve(oof["true"], oof[model])
        axes[2].plot(tol, acc, lw=lw, color=color, label=PRETTY[model])
    axes[2].set_xlabel("Absolute error tolerance $\\tau$ (eV)")
    axes[2].set_ylabel("Fraction within $\\tau$")
    axes[2].set_title("Regression error characteristic", fontsize=9)
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(F / "fig5_parity_rec.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def fig_ablation():
    if not _exists("ablation_results.csv"):
        return
    df = pd.read_csv(M / "ablation_results.csv").sort_values("pooled_r2")
    labels = {
        "gbr_prior_only": "Tree prior alone", "in_sample_prior": "In-sample prior (leaky)",
        "no_physics_loss": "No boundary penalty", "no_structural_layer": "No coupling layer",
        "no_anisotropy_loss": "No anisotropy penalty", "full": "Full framework",
        "no_quantile_heads": "No quantile heads", "no_residual_head": "No residual head",
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    colors = [ACCENT if v == "full" else NEUTRAL for v in df["variant"]]
    ax.barh(np.arange(len(df)), df["pooled_r2"], color=colors,
            edgecolor="white", height=.66)
    for i, r in enumerate(df["pooled_r2"]):
        ax.text(r + .003, i, f"{r:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(np.arange(len(df)),
                  [labels.get(v, v) for v in df["variant"]], fontsize=8)
    ax.set_xlim(0, df["pooled_r2"].max() * 1.10)
    ax.set_xlabel("Pooled out-of-fold $R^2$")
    ax.set_title("Component ablation (all configurations trained)",
                 fontsize=10, fontweight="bold")
    fig.savefig(F / "fig6_ablation.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 7
def fig_calibration():
    if not _exists("benchmark_summary.json", "predictions_oof.csv"):
        return
    oof = pd.read_csv(M / "predictions_oof.csv")
    if "hybrid_gbr_q25" not in oof.columns:
        return
    lo = np.minimum(oof.hybrid_gbr_q25, oof.hybrid_gbr_q75)
    hi = np.maximum(oof.hybrid_gbr_q25, oof.hybrid_gbr_q75)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    levels = np.linspace(0.05, 0.95, 19)
    centre = (lo + hi) / 2
    half = (hi - lo) / 2
    empirical = [float((np.abs(oof["true"] - centre) <= half * (L / 0.5)).mean())
                 for L in levels]
    axes[0].plot([0, 1], [0, 1], ls="--", color="#333", lw=1, label="ideal")
    axes[0].plot(levels, empirical, "o-", color=WARN, lw=1.8, markersize=3.5,
                 label="measured")
    axes[0].set_xlabel("Nominal coverage"); axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("(a) Interval calibration", fontsize=9.5, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(hi - lo, bins=40, color=NEUTRAL, edgecolor="white")
    axes[1].axvline((hi - lo).mean(), color=WARN, lw=1.6,
                    label=f"mean {float((hi - lo).mean()):.2f} eV")
    axes[1].set_xlabel("Predicted interval width (eV)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) Interval width", fontsize=9.5, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(F / "fig7_calibration.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 8
def fig_repeated_holdout():
    """Per-split held-out performance across independent held-out sets."""
    if not _exists("repeated_holdout.csv", "repeated_holdout.json"):
        return
    df = pd.read_csv(M / "repeated_holdout.csv")
    with open(M / "repeated_holdout.json") as fh:
        s = json.load(fh)

    order = [m for m in ["gbr", "rf", "svr", "mlp", "pinn", "stack_nnls", "stack_ridge"]
             if m in df.variant.unique()]
    jitter = np.random.RandomState(0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for i, m in enumerate(order):
        g = df[df.variant == m]
        color = ACCENT if "stack" in m else NEUTRAL
        axes[0].scatter(np.full(len(g), i) + jitter.uniform(-.12, .12, len(g)),
                        g.r2, s=22, color=color, alpha=.65, edgecolors="none")
        axes[0].hlines(g.r2.mean(), i - .28, i + .28, color=color, lw=2.2)
    axes[0].set_xticks(range(len(order)), [PRETTY[m] for m in order],
                       rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Held-out $R^2$")
    axes[0].set_title(f"(a) {s['n_splits']} independent held-out sets "
                      f"($\\approx${s['mean_n_test']} materials each)",
                      fontsize=9, fontweight="bold")

    piv = df.pivot_table(index="split", columns="variant", values="r2")
    pairs = [(inc, "stack_ridge") for inc in ("pinn", "mlp", "gbr") if inc in piv]
    for j, (inc, ch) in enumerate(pairs):
        delta = piv[ch] - piv[inc]
        axes[1].scatter(np.full(len(delta), j) + jitter.uniform(-.1, .1, len(delta)),
                        delta, s=26, edgecolors="none",
                        color=[GOOD if d > 0 else WARN for d in delta])
        axes[1].hlines(delta.mean(), j - .26, j + .26, color="#333", lw=2)
        c = s["comparisons"].get(f"{ch}_vs_{inc}")
        if c:
            axes[1].annotate(f"{c['splits_won']}/{c['n_splits']}\n"
                             f"p={c['p_one_sided']:.3f}", (j, delta.max()),
                             textcoords="offset points", xytext=(0, 9),
                             ha="center", fontsize=7.5)
    axes[1].axhline(0, color="#333", lw=1)
    axes[1].set_xticks(range(len(pairs)),
                       [f"vs {PRETTY[inc]}" for inc, _ in pairs], fontsize=8)
    axes[1].set_ylabel("$\\Delta R^2$ (stack $-$ single model)")
    axes[1].set_title("(b) Paired per-split difference", fontsize=9, fontweight="bold")
    axes[1].margins(y=0.28)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(F / "fig8_repeated_holdout.png")
    plt.close(fig)


def main() -> None:
    for fn in (fig_model_comparison, fig_drop_one_and_weights, fig_error_correlation,
               fig_learning_curve, fig_parity_and_rec, fig_ablation, fig_calibration,
               fig_repeated_holdout):
        try:
            fn()
        except Exception as exc:  # keep going; report what failed
            print(f"[figures] {fn.__name__} failed: {exc}")

    latex_figs = config.ROOT / "paper" / "ieee_access" / "figures"
    if latex_figs.exists():
        import shutil
        for png in F.glob("fig*.png"):
            shutil.copy(png, latex_figs / png.name)
        print(f"[figures] copied into {latex_figs}")
    print(f"[figures] written to {F}")


if __name__ == "__main__":
    main()
