# Band-Gap Prediction for 2D Materials Without DFT

Code and data for *Band-Gap Prediction for Two-Dimensional Materials Without Density
Functional Theory: A Leakage-Controlled Evaluation of Hybrid and Ensemble Models*
(Jagadeesh, Mudalagi, Akl), prepared for **IEEE Access**.

Repository: <https://github.com/adhvayjagadeesh/DFT---Machine-Learning--PINN>

**Headline result.** HSE06 band gaps of rectangular-lattice 2D materials are predicted to
**R² = 0.836, MAE = 0.417 eV from chemical composition and crystal symmetry alone** — no
electronic-structure calculation on the target material. That is more accurate than a
previously published physics-informed hybrid which used the full DFT-derived descriptor
set (R² = 0.743, MAE = 0.555 eV), and unlike it, applies to compounds never calculated.

**Secondary result (negative).** Under the same protocol, a physics-informed hybrid that
corrects a gradient-boosted prior with a neural residual head does *not* improve on that
prior. Reverting one implementation detail — building the prior in-sample rather than
out-of-fold — reproduces the previously reported gain, identifying it as an evaluation
artifact.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

| Experiment | Command | Runtime |
|---|---|---|
| Seven-model benchmark | `python experiments/run_benchmarks.py --splitter group --tune` | ~11 min |
| Component ablation | `python experiments/run_ablation.py --splitter group` | ~11 min |
| Hybrid recovery study | `python experiments/run_recovery_study.py --repeats 5` | ~13 min |
| Stacking analysis | `python experiments/run_stacking_analysis.py --repeats 3` | ~5 min |
| Repeated held-out | `python experiments/run_repeated_holdout.py --splits 8` | ~13 min |
| DFT-cost tiers | `python experiments/run_feature_tiers.py --repeats 3` | ~11 min |
| Ensemble utility | `python experiments/run_utility_study.py` | ~1 min |
| All figures | `python experiments/make_figures.py` | ~30 s |

Add `--quick` to any experiment for a fast smoke test. Results in that mode are not
publication-grade.

---

## Results

Repeated grouped five-fold cross-validation, 25 fold estimates, all descriptors.
*p*-values are Nadeau–Bengio corrected, one-sided, against the GBR baseline.

| Model | R² | MAE (eV) | Beats GBR | *p* |
|---|---|---|---|---|
| **Stacked ensemble (ridge)** | **0.877 ± 0.038** | **0.363** | 25/25 | <0.001 |
| Stacked ensemble (NNLS) | 0.875 ± 0.038 | 0.364 | 25/25 | <0.001 |
| Deep MLP | 0.864 ± 0.039 | 0.378 | 24/25 | 0.002 |
| Standalone PINN | 0.856 ± 0.043 | 0.385 | 22/25 | 0.033 |
| Random forest | 0.827 ± 0.041 | 0.442 | 13/25 | 0.323 |
| GBR (baseline) | 0.823 ± 0.043 | 0.464 | — | — |
| Hybrid GBR + corrector | 0.810 ± 0.047 | 0.445 | 12/25 | 0.271 |
| SVR | 0.774 ± 0.039 | 0.497 | 1/25 | — |

### Accuracy against DFT cost

| Tier | What it needs | R² | MAE (eV) |
|---|---|---|---|
| All descriptors | Converged DFT | 0.880 ± 0.037 | 0.360 |
| No DFT energies | Relaxed geometry | 0.845 ± 0.046 | 0.398 |
| **No DFT at all** | **Formula + symmetry** | **0.836 ± 0.057** | **0.417** |

### Head-to-head against the prior pipeline

The estimators specified by the earlier code, at their original hyperparameters, on
identical folds:

| Model | R² | MAE (eV) | Ensemble wins | *p* |
|---|---|---|---|---|
| XGBoost (prior) | 0.858 | 0.388 | 14/15 | 0.025 |
| GBR (prior) | 0.852 | 0.405 | 14/15 | 0.002 |
| RF (prior) | 0.826 | 0.443 | 15/15 | <0.001 |

---

## Repository layout

```
data/
  raw/c2db_raw.csv               C2DB export (1,169 rows carry an HSE06 label)
  processed/                     generated feature matrices
src/pinn_dft/
  config.py                      paths, schema, protocol constants
  data.py                        cleaning, Magpie featurisation, fold-safe encoding
  utils.py                       seeding
  models/
    baselines.py                 RF / GBR / SVR with in-fold hyperparameter search
    neural.py                    deep MLP and standalone PINN
    structural_layer.py          geometric coupling layer
    hybrid.py                    hybrid residual model + out-of-fold prior
    losses.py                    boundary, pinball, anisotropy terms
  evaluation/
    metrics.py                   point metrics, REC curves, quantile calibration
    statistics.py                fold-level and Nadeau–Bengio corrected tests
experiments/                     one script per experiment, all reproducible
results/
  metrics/                       generated JSON/CSV outputs
  figures/                       generated figures (fig1–fig10)
tests/                           pytest suite
paper/ieee_access/               IEEE Access manuscript sources
archive/exploratory/             superseded scripts, kept for provenance
```

---

## Evaluation protocol

| Aspect | Setting |
|---|---|
| Data | 1,169 C2DB materials with an HSE06 label |
| Held-out evaluation | 8 independent formula-grouped sets (~234 each) |
| Cross-validation | 5 repeats × grouped 5-fold = 25 fold estimates |
| Grouping key | chemical formula |
| Feature scaling / encoding | fitted on the training fold only |
| Stacked inputs | out-of-fold, via an inner 5-fold split |
| Significance | Nadeau–Bengio corrected paired *t*-test |
| Seed | 42 |

**Why grouping matters.** 176 materials share a formula with another entry — polymorphs
distinguished by layer group, whose gaps differ by up to 2.9 eV. Because composition
descriptors are near-identical within such a pair, an ungrouped split lets a model
memorise one polymorph and be scored on another.

---

## Corrections relative to the earlier pipeline

Documented because the numbers in the earlier manuscript predate these fixes.

| Defect | Effect | Fix |
|---|---|---|
| Prior built from **in-sample** tree predictions | Corrector trained against a prior far better than at inference | Out-of-fold construction (inner 5-fold) |
| One-hot encoding fitted on the **full table** | Validation-fold vocabulary visible in training | Encoding inside each fold |
| 1,077 `Formula_` one-hot columns (825 singletons) | Row identifiers, not descriptors | Replaced by 120 Magpie statistics |
| Blanket `dropna()` | Discarded 239 of 1,169 labelled materials | Median imputation + missingness indicator |
| Random CV with 176 polymorph rows | Same-formula polymorphs straddling folds | `GroupKFold` on formula |
| Geometric channels at **positions 0 and 1** | Pointed at two formation energies, not geometry | Resolved by name, asserted in tests |
| Paired *t*-test over 930 per-sample errors | Non-independent; overstated significance | Fold-level, Nadeau–Bengio corrected |
| Ablation via `− np.random.uniform(...)` | Values were not measurements | All 8 configurations trained |
| Baselines at library defaults | Comparison biased toward the hybrid | Equal in-fold search budget |

---

## Known limitations

- **No in-plane lattice vectors.** The anisotropy hypothesis motivating the structural
  layer concerns unequal in-plane vectors *a* and *b*. The current C2DB export contains
  neither, so the layer operates on unit-cell area and thickness and the hypothesis
  cannot be tested here. A full C2DB structure pull is the most valuable extension.
- **Single database, single Bravais class.** All data are C2DB rectangular (*op*).
- **In the DFT-free tier** the ensemble's advantage over a well-tuned XGBoost is not
  statistically significant (*p* = 0.097).
- **No graph-network comparison.** Those require relaxed atomic coordinates, which would
  reintroduce the DFT dependence the DFT-free tier exists to avoid.
- **The structural layer is a soft inductive bias**, not a derivation from elasticity theory.

## Reproducibility note

XGBoost and PyTorch each link their own copy of `libomp`. On macOS, importing torch first
makes later XGBoost fits segfault; leaving OpenMP multi-threaded deadlocks the process
instead. `experiments/run_feature_tiers.py` pins `OMP_NUM_THREADS` before any such import
and loads xgboost first — do not reorder those lines.

## Data provenance

Descriptors and HSE06 band gaps derive from the Computational 2D Materials Database
(C2DB), licensed CC-BY-SA 4.0. Please cite C2DB alongside this repository; see
`paper/ieee_access/references.bib`.

## Citation

```bibtex
@article{jagadeesh2026dftfree,
  title  = {Band-Gap Prediction for Two-Dimensional Materials Without Density
            Functional Theory: A Leakage-Controlled Evaluation of Hybrid and
            Ensemble Models},
  author = {Jagadeesh, Adhvay and Mudalagi, Rutvi and Akl, Marx},
  year   = {2026},
  note   = {Manuscript under preparation}
}
```
