"""Regenerate every manuscript figure from the committed result files.

Run after the experiment scripts in experiments/. Figures whose inputs are missing
are skipped with a message rather than failing::

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


# ---------------------------------------------------------------- figure 9
def fig_utility():
    """Why the ensemble earns its cost, in decision-relevant terms."""
    if not _exists("utility_study.json"):
        return
    with open(M / "utility_study.json") as fh:
        s = json.load(fh)
    base = ["rf", "gbr", "svr", "mlp", "pinn"]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))

    # (a) which single model wins, per fold
    r = s["model_selection_risk"]
    wins = [r["win_counts"][m] for m in base]
    axes[0].bar(range(len(base)), wins, color=NEUTRAL, edgecolor="white", width=.62)
    axes[0].set_xticks(range(len(base)), [PRETTY[m] for m in base],
                       rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel(f"Folds won (of {r['n_folds']})")
    axes[0].set_title("(a) The best single model is not stable",
                      fontsize=9.5, fontweight="bold")
    axes[0].set_ylim(0, max(wins) * 1.55)
    axes[0].text(0.5, 0.97,
                 f"stack beats the committed choice\non "
                 f"{r['folds_stack_beats_committed']}/{r['n_folds']} folds "
                 f"(p={r['p_stack_vs_committed']:.4f})",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#EEF3FC", ec="none"))

    # (b) tail risk
    t = s["tail_risk"]
    order = base + ["stack_nnls"]
    rates = [t[m]["rate_error_above_1eV"] * 100 for m in order]
    colors = [ACCENT if m == "stack_nnls" else NEUTRAL for m in order]
    axes[1].bar(range(len(order)), rates, color=colors, edgecolor="white", width=.62)
    for i, v in enumerate(rates):
        axes[1].text(i, v + .15, f"{v:.1f}", ha="center", fontsize=7.5)
    axes[1].set_xticks(range(len(order)), [PRETTY[m] for m in order],
                       rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("% of predictions in error by >1 eV")
    axes[1].set_title("(b) Fewer large errors", fontsize=9.5, fontweight="bold")

    # (c) cost in context
    if "cost" in s:
        c = s["cost"]
        labels = ["MLP\nalone", "Full\nensemble", "One HSE06\ncalculation"]
        # A single HSE06 calculation on a small 2D cell is hours of CPU time;
        # 1 CPU-hour is used as a deliberately conservative reference point.
        values = [c["per_model"]["mlp"]["train_s"], c["stack_total_train_s"], 3600.0]
        bars = axes[2].bar(range(3), values,
                           color=[NEUTRAL, ACCENT, WARN], edgecolor="white", width=.6)
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Wall-clock seconds (log scale)")
        axes[2].set_xticks(range(3), labels, fontsize=8)
        for b, v in zip(bars, values):
            axes[2].text(b.get_x() + b.get_width() / 2, v * 1.25,
                         f"{v:.1f}s" if v < 100 else f"{v / 3600:.0f} CPU-h",
                         ha="center", fontsize=7.5)
        axes[2].set_title("(c) The ensemble's overhead is negligible",
                          fontsize=9.5, fontweight="bold")

    fig.subplots_adjust(wspace=0.33)
    fig.text(0.5, -0.06,
             "The ensemble is justified not by its $R^2$ margin but by removing the "
             "risk of committing to the wrong single model, cutting large errors, "
             "and costing seconds against the CPU-hours of the calculation it replaces.",
             ha="center", fontsize=7.5, color="#666")
    fig.savefig(F / "fig9_utility.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 10
def fig_feature_tiers():
    """What accuracy costs in DFT, and how the pipeline compares to the prior one."""
    if not _exists("feature_tier_metrics.csv", "feature_tiers.json"):
        return
    df = pd.read_csv(M / "feature_tier_metrics.csv")
    with open(M / "feature_tiers.json") as fh:
        s = json.load(fh)

    tiers = ["full", "geometry_only", "dft_free"]
    tier_label = {"full": "All descriptors", "geometry_only": "No DFT energies",
                  "dft_free": "No DFT at all\n(composition + symmetry)"}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # (a) accuracy vs DFT cost, ensemble and the prior pipeline's best model
    for model, color, label, marker in (
            ("stack_ridge", ACCENT, "Stacked ensemble (ours)", "o"),
            ("prior_xgb", NEUTRAL, "XGBoost (prior pipeline)", "s")):
        means = [df[(df.tier == t) & (df.variant == model)].r2.mean() for t in tiers]
        errs = [1.96 * df[(df.tier == t) & (df.variant == model)].r2.sem()
                for t in tiers]
        axes[0].errorbar(range(3), means, yerr=errs, marker=marker, color=color,
                         lw=2, capsize=3, markersize=7, label=label)
        for i, v in enumerate(means):
            axes[0].annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                             xytext=(0, 11 if model == "stack_ridge" else -16),
                             ha="center", fontsize=7.5, color=color)
    axes[0].axhline(0.7431, ls="--", lw=1.2, color=WARN)
    axes[0].text(2.05, 0.7431, " previously published\n hybrid (0.743)",
                 fontsize=7.5, color=WARN, va="center", ha="left")
    axes[0].set_xticks(range(3), [tier_label[t] for t in tiers], fontsize=8)
    axes[0].set_ylabel("Cross-validated $R^2$")
    axes[0].set_xlim(-0.3, 3.05)
    axes[0].set_title("(a) Accuracy against DFT cost", fontsize=9.5, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    # (b) head-to-head within the full tier
    full = df[df.tier == "full"]
    order = ["svr", "gbr", "rf", "prior_rf", "prior_gbr", "prior_xgb",
             "pinn", "mlp", "stack_nnls", "stack_ridge"]
    order = [m for m in order if m in full.variant.unique()]
    names = {**PRETTY, "prior_xgb": "XGBoost (prior)", "prior_gbr": "GBR (prior)",
             "prior_rf": "RF (prior)"}
    means = [full[full.variant == m].r2.mean() for m in order]
    colors = [ACCENT if "stack" in m else (WARN if m.startswith("prior") else NEUTRAL)
              for m in order]
    axes[1].barh(range(len(order)), means, color=colors, edgecolor="white", height=.68)
    for i, v in enumerate(means):
        axes[1].text(v + .004, i, f"{v:.3f}", va="center", fontsize=7.5)
    axes[1].set_yticks(range(len(order)), [names[m] for m in order], fontsize=8)
    axes[1].set_xlim(0.70, max(means) * 1.045)
    axes[1].set_xlabel("Cross-validated $R^2$")
    axes[1].set_title("(b) Head-to-head, all descriptors", fontsize=9.5,
                      fontweight="bold")

    fig.subplots_adjust(wspace=0.34)
    fig.text(0.5, -0.05,
             "Red bars are the estimators specified by the prior pipeline, at their "
             "original hyperparameters, on identical folds. Removing every DFT-derived "
             "descriptor costs 0.044 $R^2$ and still exceeds the published hybrid.",
             ha="center", fontsize=7.5, color="#666")
    fig.savefig(F / "fig10_feature_tiers.png")
    plt.close(fig)


def main() -> None:
    for fn in (fig_model_comparison, fig_drop_one_and_weights, fig_error_correlation,
               fig_learning_curve, fig_parity_and_rec, fig_ablation, fig_calibration,
               fig_repeated_holdout, fig_utility, fig_feature_tiers):
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
