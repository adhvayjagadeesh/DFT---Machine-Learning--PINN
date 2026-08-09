"""Central configuration: paths, column names, and protocol constants.

Every path is derived from the repository root so the pipeline runs unchanged on
any machine. No absolute paths appear anywhere else in the codebase.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw" / "c2db_raw.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "c2db_features.csv"
RESULTS = ROOT / "results"
RESULTS_METRICS = RESULTS / "metrics"
RESULTS_FIGURES = RESULTS / "figures"

for _d in (DATA_PROCESSED.parent, RESULTS_METRICS, RESULTS_FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --- dataset schema -------------------------------------------------------
TARGET = "Band gap (HSE06) [eV]"
FORMULA_COL = "Formula"

#: Columns dropped before modelling, with the reason each one is excluded.
DROP_COLUMNS = {
    "2D Bravais type": "constant across the curated subset (all 'Rectangular (op)')",
}

#: Nominal columns that are one-hot encoded *inside* each training fold.
CATEGORICAL_COLUMNS = ["Magnetic", "Layer group (not Space group)", "Stoichiometry"]

#: Continuous descriptors taken directly from C2DB.
NUMERIC_COLUMNS = [
    "Energy above hull [eV/atom]",
    "Heat of formation [eV/atom]",
    "Thickness [Å]",
    "Energy [eV]",
    "Unit cell area [Å2]",
    "Vacuum level [eV]",
    "Total magnetic moment [μB]",
]

#: The two geometric channels fed to the structural coupling layer.
#: These are looked up *by name* at runtime; earlier revisions hard-coded
#: positional indices 0 and 1, which silently pointed at two formation
#: energies rather than at any geometric quantity.
GEOMETRIC_CHANNELS = ("Unit cell area [Å2]", "Thickness [Å]")

# --- evaluation protocol --------------------------------------------------
SEED = 42
N_SPLITS = 5
INNER_SPLITS = 5          # folds used to build out-of-fold base-model priors
TEST_FRACTION = 0.15      # held-out set, carved before any model selection
QUANTILES = (0.25, 0.50, 0.75)
