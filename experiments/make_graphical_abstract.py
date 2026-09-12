"""Graphical abstract for the IEEE Access submission.

One image, one message: the hybrid's reported advantage exists only under an
in-sample prior; one line of code removes it, and a stacked ensemble is what
actually survives. Every number is read from the committed result artefacts.

Design notes
------------
* Form: emphasis + before/after. The baseline is grey (de-emphasis), the hybrid
  is the accent whose fate the reader follows, the ensemble is a second accent.
  Both accents validated against each other and the surface (dE 24-30, CVD-safe).
* Marks are thin with rounded ends anchored to the baseline; values are direct
  labels in ink colour, never in the series colour.
* Text is kept to what a reader needs in five seconds. The code diff is the
  hinge of the story and sits between the two panels.

Run::

    python experiments/make_graphical_abstract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pinn_dft import config  # noqa: E402

# ---------------------------------------------------------------- data
M = config.RESULTS_METRICS
ab = {v["variant"]: v for v in json.load(open(M / "ablation_results.json"))["variants"]}
rc = pd.read_csv(M / "recovery_fold_metrics.csv")
piv = rc.pivot_table(index=["repeat", "fold"], columns="variant", values="mse")

LEAKY = ab["in_sample_prior"]["r2_mean"]
CLEAN = ab["full"]["r2_mean"]
PRIOR = ab["gbr_prior_only"]["r2_mean"]
STACK = rc[rc.variant == "stack_ridge"].r2.mean()
LEAKY_WINS = ab["in_sample_prior"]["estimates_improved_vs_prior"]
CLEAN_WINS = ab["full"]["estimates_improved_vs_prior"]
STACK_WINS = int((piv["gbr"] > piv["stack_ridge"]).sum())
N_EST = ab["in_sample_prior"]["n_estimates"]
N_MAT = 1169

# ---------------------------------------------------------------- palette
INK, INK2, MUTED = "#1F2328", "#57606A", "#8C959F"
SURFACE, PANEL, RULE = "#FFFFFF", "#F6F8FA", "#D0D7DE"
ACCENT_BAD, ACCENT_GOOD, BASE = "#C2453B", "#2E6FD9", "#8C959F"
CODE_BG, CODE_DEL, CODE_ADD = "#F6F8FA", "#FFEBE9", "#DAFBE1"

mpl.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 100})

W, H = 12.0, 6.75                       # inches; 16:9, rendered at 300 dpi
fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
fig.patch.set_facecolor(SURFACE)


def text(x, y, s, size, color=INK, weight="normal", ha="left", va="center",
         family=None, **kw):
    fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha,
             va=va, family=family, **kw)


def rbox(x, y, w, h, fc, ec="none", r=0.012, lw=0, z=-2):
    # Figure-level patches and axes both default to zorder 0 and the patch
    # wins the tie, so backgrounds must sit strictly below the axes.
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        transform=fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=z))


# ---------------------------------------------------------------- title band
text(0.05, 0.925, "Evaluation Artefacts in Physics-Informed Band-Gap Prediction",
     19, INK, "bold")
text(0.05, 0.875,
     f"A leakage-controlled study of {N_MAT:,} two-dimensional materials  ·  "
     f"C2DB, HSE06 band gaps  ·  {N_EST} fold estimates, grouped by composition",
     10.5, INK2)
fig.add_artist(plt.Line2D([0.05, 0.95], [0.84, 0.84], transform=fig.transFigure,
                          color=RULE, lw=1))


# ---------------------------------------------------------------- panel helper
def panel(x0, w, heading, sub, hybrid_val, hybrid_wins, tone, tag):
    """Diverging bar: the hybrid's R^2 measured from the tree prior.

    The prior is the meaningful zero here -- the whole story is whether the
    corrector lands above or below it -- so the bar is anchored to the prior,
    not to an arbitrary axis origin, and extends right when the hybrid is
    better and left when it is worse.
    """
    rbox(x0, 0.16, w, 0.63, PANEL)
    text(x0 + 0.02, 0.755, heading, 12.5, INK, "bold")
    text(x0 + 0.02, 0.715, sub, 9.5, INK2)

    ax_left, ax_right = x0 + 0.03, x0 + w - 0.03
    half = 0.045                                    # axis half-width in R^2
    ax = fig.add_axes([ax_left, 0.24, ax_right - ax_left, 0.40])
    ax.set_zorder(1)
    ax.set_xlim(PRIOR - half, PRIOR + half)
    ax.set_ylim(-0.9, 1.9)
    ax.axis("off")

    # the prior as a vertical rule, stopped short of the labels above and below
    ax.plot([PRIOR, PRIOR], [-0.05, 1.15], color=BASE, lw=1.8,
            ls=(0, (4, 3)), zorder=1, solid_capstyle="butt")
    ax.text(PRIOR, 1.28, "tree prior alone", ha="center", va="bottom",
            fontsize=9, color=INK2)
    ax.text(PRIOR, 1.62, f"$R^2$ = {PRIOR:.3f}", ha="center", va="bottom",
            fontsize=9.5, color=INK, fontweight="bold")

    # diverging bar from the prior
    delta = hybrid_val - PRIOR
    ax.barh(0.5, delta, left=PRIOR, height=0.55, color=tone,
            edgecolor=SURFACE, lw=2, zorder=2)
    # value at the free end, name at the anchored end
    if delta >= 0:
        ax.text(hybrid_val + 0.003, 0.5, f"{hybrid_val:.3f}", va="center",
                ha="left", fontsize=13, color=INK, fontweight="bold")
        ax.text(PRIOR - 0.003, 0.5, "hybrid", va="center", ha="right",
                fontsize=10, color=INK2, fontweight="bold")
    else:
        ax.text(hybrid_val - 0.003, 0.5, f"{hybrid_val:.3f}", va="center",
                ha="right", fontsize=13, color=INK, fontweight="bold")
        ax.text(PRIOR + 0.003, 0.5, "hybrid", va="center", ha="left",
                fontsize=10, color=INK2, fontweight="bold")

    # verdict, centred under the rule
    sign = "above" if delta > 0 else "below"
    ax.text(PRIOR, -0.45,
            f"{sign} the prior by {abs(delta):.3f}\nwins {hybrid_wins} of {N_EST} folds",
            fontsize=9.5, color=INK2, va="top", ha="center", linespacing=1.5)
    text(x0 + 0.02, 0.19, tag, 9, MUTED, style="italic")


# ---------------------------------------------------------------- left: leaky
panel(0.05, 0.36, "As reported",
      "prior built on the same rows the corrector trains on",
      LEAKY, LEAKY_WINS, ACCENT_BAD,
      "an apparent improvement — every other component held fixed")

# ---------------------------------------------------------------- right: clean
panel(0.59, 0.36, "Under leakage control",
      "prior built out of fold, as at inference",
      CLEAN, CLEAN_WINS, ACCENT_BAD,
      f"the advantage does not survive  ·  p = 0.54")

# ---------------------------------------------------------------- centre: the hinge
cx = 0.50
rbox(cx - 0.065, 0.33, 0.13, 0.30, SURFACE, ec=RULE, lw=1, r=0.01, z=-1)
text(cx, 0.60, "one line", 11, INK, "bold", ha="center")
text(cx, 0.572, "changed", 11, INK, "bold", ha="center")

rbox(cx - 0.058, 0.485, 0.116, 0.052, CODE_DEL, r=0.006, z=0)
text(cx, 0.511, "− fit(X).predict(X)", 8.6, "#82071E", ha="center",
     family="DejaVu Sans Mono")
rbox(cx - 0.058, 0.415, 0.116, 0.052, CODE_ADD, r=0.006, z=0)
text(cx, 0.441, "+ out_of_fold(X)", 8.6, "#116329", ha="center",
     family="DejaVu Sans Mono")
text(cx, 0.372, "how the tree prior\nis constructed", 8, INK2, ha="center")

for x0, x1 in ((0.412, 0.432), (0.568, 0.588)):
    fig.patches.append(FancyArrowPatch(
        (x0, 0.48), (x1, 0.48), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.4, zorder=5))

# ---------------------------------------------------------------- footer: what works
fig.add_artist(plt.Line2D([0.05, 0.95], [0.135, 0.135], transform=fig.transFigure,
                          color=RULE, lw=1))
text(0.05, 0.095, "What survives", 11, INK, "bold")
# Same R^2-per-inch scale as the panels, so the three bars are comparable by
# eye: panel axes span 0.09 R^2 over 0.30 fig-width -> footer at 0.55 width
# spans 0.165 R^2.
fax_w = 0.55
fax = fig.add_axes([0.20, 0.06, fax_w, 0.065])
fax.set_zorder(1)
half = (0.09 / 0.30) * fax_w / 2
fax.set_xlim(PRIOR - half, PRIOR + half); fax.set_ylim(0, 1); fax.axis("off")
fax.plot([PRIOR, PRIOR], [0.02, 0.98], color=BASE, lw=1.6, ls=(0, (4, 3)))
fax.barh(0.5, STACK - PRIOR, left=PRIOR, height=0.62, color=ACCENT_GOOD,
         edgecolor=SURFACE, lw=2)
fax.text(PRIOR - 0.003, 0.5, "stacked ensemble\nof five learners", va="center",
         ha="right", fontsize=9, color=INK2, fontweight="bold", linespacing=1.3)
fax.text(STACK + 0.003, 0.5, f"{STACK:.3f}", va="center", fontsize=13,
         color=INK, fontweight="bold")
text(0.20, 0.03,
     f"beats the prior on {STACK_WINS} of {N_EST} folds  ·  p < 0.001  ·  "
     "MAE 0.363 eV  ·  trains in under nine seconds",
     9.5, INK2, ha="left")

out_png = config.RESULTS_FIGURES / "graphical_abstract.png"
out_pdf = config.RESULTS_FIGURES / "graphical_abstract.pdf"
fig.savefig(out_png, dpi=300, facecolor=SURFACE, bbox_inches=None)
fig.savefig(out_pdf, facecolor=SURFACE, bbox_inches=None)
print(f"written {out_png} ({int(W*300)}x{int(H*300)} px) and {out_pdf}")
