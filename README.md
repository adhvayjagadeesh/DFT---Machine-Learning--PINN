# Physics-Informed Hybrid Residual Learning for Band-Gap Prediction of 2D Materials

Code and data accompanying the manuscript *Physics-Informed Hybrid Residual Learning for
Band-Gap Prediction of Two-Dimensional Materials* (Jagadeesh, Mudalagi, Akl), prepared for
submission to **IEEE Access**.

The pipeline predicts HSE06 band gaps of rectangular-lattice two-dimensional materials from
tabular physicochemical descriptors, combining a gradient-boosted tree prior with a neural
residual correction head.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Reproduce the full benchmark and the ablation:

```bash
python experiments/run_benchmarks.py --splitter group --tune
```

```bash
python experiments/run_ablation.py --splitter group
```

Run the test suite:

```bash
python -m pytest tests/ -q
```

Add `--quick` to either experiment for a fast smoke test (short training schedules, no
hyperparameter search). Results are not publication-grade in that mode.

---

## Repository layout

```
data/
  raw/c2db_raw.csv                 C2DB export, 2,848 rows (1,169 carry an HSE06 label)
  processed/                       generated feature matrices
src/pinn_dft/
  config.py                        paths, schema, protocol constants
  data.py                          cleaning, composition featurisation, fold-safe encoding
  utils.py                         seeding
  models/
    baselines.py                   RF / GBR / SVR with in-fold hyperparameter search
    neural.py                      deep MLP and standalone PINN
    structural_layer.py            geometric coupling layer
    hybrid.py                      hybrid residual model + out-of-fold prior construction
    losses.py                      boundary, pinball, and anisotropy terms
  evaluation/
    metrics.py                     point metrics, REC curves, quantile calibration
    statistics.py                  fold-level and Nadeau-Bengio corrected tests
experiments/
  run_benchmarks.py                seven-model benchmark, grouped CV, held-out test set
  run_ablation.py                  measured component ablation
results/
  metrics/                         generated JSON/CSV outputs
  figures/                         generated figures
tests/                             pytest suite
paper/                             manuscript sources
archive/exploratory/               superseded exploratory scripts, kept for provenance
```

---

## Evaluation protocol

| Aspect | Setting |
|---|---|
| Data | 1,169 C2DB materials with an HSE06 label |
| Held-out test set | 15%, carved by formula group before any model selection |
| Cross-validation | 5-fold `GroupKFold` on the remaining 85% |
| Grouping key | chemical formula |
| Feature scaling | fitted on the training fold only |
| Categorical encoding | vocabulary taken from the training fold only |
| Base-model prior | out-of-fold, via an inner 5-fold split |
| Significance | Nadeau-Bengio corrected paired t-test over fold scores |
| Seed | 42 |

**Why grouping matters.** 176 labelled materials share a chemical formula with at least one
other entry — these are polymorphs, distinguished by layer group, whose band gaps differ by up
to 2.9 eV. Because the feature set includes composition-derived descriptors, an ungrouped split
lets a model see one polymorph in training and be scored on another with near-identical
features. `--splitter random` reproduces the ungrouped protocol for comparison; grouped is the
figure that should be reported.

---

## Corrections made in this revision

This revision fixes several defects that biased the previously reported results. They are
documented here because the numbers in the current manuscript draft predate the fixes.

**Leakage: in-sample base-model prior.** The hybrid built its training features from
`base_model.predict(X_train)` — in-sample predictions of a tree already fitted to those rows.
The residual head therefore trained against a prior far more accurate than the one it meets at
inference. `hybrid.out_of_fold_prior()` replaces this with the standard stacked-generalisation
construction. The `in_sample_prior` ablation variant quantifies the difference.

**Feature matrix dominated by row identifiers.** The raw `Formula` string was one-hot encoded
into 1,077 columns, 825 of which occurred in exactly one material. These are identifiers, not
descriptors. They are replaced by Magpie elemental-property statistics (120 columns after
constant/NaN filtering) plus atom count and packing density.

**Encoding fitted on the full dataset.** One-hot encoding ran before cross-validation, exposing
the category vocabulary of every validation fold. Encoding now happens inside each fold.

**Geometric channels pointed at the wrong columns.** The structural layer was instantiated with
positional indices 0 and 1, which in the assembled matrix were *Energy above hull* and *Heat of
formation* — two formation energies. Channels are now resolved by name (see
`config.GEOMETRIC_CHANNELS`), and a test asserts this.

**239 labelled materials silently discarded.** A blanket `dropna()` removed every row missing
*Total magnetic moment*, cutting 1,169 labelled materials to 930. That column is now imputed
from the training fold with an accompanying missingness indicator.

**Invalid significance test.** The reported p-value came from a paired t-test over 930
per-sample squared errors, which are not independent observations of model performance. The
test is now applied at fold level, with the Nadeau-Bengio variance correction for overlapping
training sets. A helper previously named `run_dietterich_5x2cv_test` implemented neither
Dietterich's test nor any variance correction and has been removed.

**Ablation study reported fabricated numbers.** The previous `submit/ablation.py` computed each
variant as `optimized_hybrid - numpy.random.uniform(...)` and raised `KeyError` before training
anything. `experiments/run_ablation.py` trains and evaluates every configuration.

**Untuned baselines.** Baselines ran at library defaults while the hybrid was hand-tuned.
All baselines now receive an equal in-fold randomised search budget.

---

## Known limitations

- **No true in-plane lattice vectors.** The anisotropy argument in the manuscript concerns
  unequal in-plane lattice vectors *a* and *b*. The current C2DB export contains neither; the
  structural layer therefore operates on unit-cell area and thickness. Recovering *a* and *b*
  requires a re-pull from C2DB and would materially strengthen the physical claim.
- **Single database.** All data are C2DB. No external validation set.
- **Modest dataset.** 1,169 labelled materials, all of rectangular (op) Bravais type.
- **The structural layer is a soft inductive bias**, not a derivation from elasticity theory.

---

## Data provenance

Descriptors and HSE06 band gaps derive from the Computational 2D Materials Database (C2DB).
Please cite C2DB alongside this repository; see `paper/ieee_access_bibliography.bib`.

## Citation

```bibtex
@article{jagadeesh2026hybrid,
  title  = {Physics-Informed Hybrid Residual Learning for Band-Gap Prediction
            of Two-Dimensional Materials},
  author = {Jagadeesh, Adhvay and Mudalagi, Rutvi and Akl, Marx},
  year   = {2026},
  note   = {Manuscript under preparation}
}
```
