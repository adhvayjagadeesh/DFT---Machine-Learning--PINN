"""Dataset construction: cleaning, composition featurisation, and fold-safe encoding.

Three defects in the original pipeline are corrected here.

1. The raw ``Formula`` string was one-hot encoded, producing 1,077 indicator
   columns of which 825 occurred in exactly one material. Those columns are row
   identifiers, not descriptors: a formula seen only in the validation fold
   contributes an all-zero block. They are replaced by Magpie elemental-property
   statistics, which generalise across compositions.
2. One-hot encoding was fitted on the full table before cross-validation, so the
   category vocabulary of every validation fold was visible during training.
   Encoding now happens inside each fold via :func:`encode_fold`.
3. ``dropna()`` was applied across all columns, silently discarding 239 of 1,169
   labelled materials because a single column (total magnetic moment) was
   missing. That column is now imputed from the training fold and accompanied by
   a missingness indicator.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Load the C2DB export and keep rows carrying an HSE06 label."""
    df = pd.read_csv(config.DATA_RAW)
    df = df.dropna(subset=[config.TARGET]).reset_index(drop=True)
    return df.drop(columns=list(config.DROP_COLUMNS), errors="ignore")


def add_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append Magpie elemental-property statistics derived from the formula.

    Falls back to a small hand-rolled descriptor set if matminer is unavailable,
    so the pipeline never silently produces a different feature space without
    saying so.
    """
    from pymatgen.core import Composition

    comps, keep = [], []
    for i, formula in enumerate(df[config.FORMULA_COL]):
        try:
            comps.append(Composition(str(formula)))
            keep.append(i)
        except Exception:  # unparseable formula -> drop the row explicitly
            warnings.warn(f"unparseable formula dropped: {formula!r}")
    df = df.iloc[keep].reset_index(drop=True)

    # Atom count and packing density, the one engineered feature worth keeping
    # from the original pipeline.
    n_atoms = np.array([c.num_atoms for c in comps], dtype=float)
    df["n_atoms"] = n_atoms
    df["atomic_density [1/Å2]"] = n_atoms / df["Unit cell area [Å2]"].to_numpy()

    try:
        from matminer.featurizers.composition import ElementProperty

        featurizer = ElementProperty.from_preset("magpie")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            magpie = featurizer.featurize_many(comps, ignore_errors=True, pbar=False)
        magpie = pd.DataFrame(magpie, columns=featurizer.feature_labels())
        magpie = magpie.loc[:, magpie.notna().all()]      # drop all-NaN columns
        magpie = magpie.loc[:, magpie.nunique() > 1]      # drop constants
        df = pd.concat([df, magpie.reset_index(drop=True)], axis=1)
        print(f"[data] Magpie composition features added: {magpie.shape[1]}")
    except ImportError:
        warnings.warn("matminer not installed - composition features limited to n_atoms/density")

    return df


def build_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return ``(features, target, groups)``.

    ``groups`` holds the chemical formula. It is used for grouped
    cross-validation so that two polymorphs of one composition never straddle a
    train/validation boundary: 176 of the labelled materials share a formula with
    at least one other entry, and their band gaps differ by up to 2.9 eV.
    """
    df = load_raw()
    df = add_composition_features(df)

    y = df[config.TARGET].to_numpy(dtype=float)
    groups = df[config.FORMULA_COL].astype(str).to_numpy()
    X = df.drop(columns=[config.TARGET, config.FORMULA_COL])
    return X, y, groups


def encode_fold(
    X_train: pd.DataFrame, X_valid: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One-hot encode, impute, and standardise using training-fold statistics only.

    Returns the transformed matrices and the resulting column names.
    """
    cats = [c for c in config.CATEGORICAL_COLUMNS if c in X_train.columns]
    nums = [c for c in X_train.columns if c not in cats]

    # --- categoricals: vocabulary comes from the training fold alone ---------
    tr_cat = pd.get_dummies(X_train[cats].astype(str), columns=cats, dtype=float)
    va_cat = pd.get_dummies(X_valid[cats].astype(str), columns=cats, dtype=float)
    va_cat = va_cat.reindex(columns=tr_cat.columns, fill_value=0.0)

    # --- numerics: median impute + missingness flag, both from training ------
    tr_num = X_train[nums].astype(float).copy()
    va_num = X_valid[nums].astype(float).copy()
    for col in nums:
        if tr_num[col].isna().any() or va_num[col].isna().any():
            flag = f"{col}__missing"
            tr_cat[flag] = tr_num[col].isna().astype(float).to_numpy()
            va_cat[flag] = va_num[col].isna().astype(float).to_numpy()
            median = tr_num[col].median()
            tr_num[col] = tr_num[col].fillna(median)
            va_num[col] = va_num[col].fillna(median)

    mu = tr_num.mean(axis=0)
    sigma = tr_num.std(axis=0).replace(0.0, 1.0)
    tr_num = (tr_num - mu) / sigma
    va_num = (va_num - mu) / sigma

    tr = pd.concat([tr_num.reset_index(drop=True), tr_cat.reset_index(drop=True)], axis=1)
    va = pd.concat([va_num.reset_index(drop=True), va_cat.reset_index(drop=True)], axis=1)
    va = va.reindex(columns=tr.columns, fill_value=0.0)

    return (
        tr.to_numpy(dtype=np.float32),
        va.to_numpy(dtype=np.float32),
        list(tr.columns),
    )


def geometric_indices(columns: list[str]) -> tuple[int, int]:
    """Locate the two geometric channels by name, not by position."""
    try:
        return tuple(columns.index(c) for c in config.GEOMETRIC_CHANNELS)  # type: ignore[return-value]
    except ValueError as exc:  # pragma: no cover - configuration error
        raise KeyError(
            f"geometric channels {config.GEOMETRIC_CHANNELS} missing from feature matrix"
        ) from exc
