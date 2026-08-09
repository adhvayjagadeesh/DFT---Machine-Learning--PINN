"""Hybrid residual model: tree prior + neural correction head.

LEAKAGE FIX (the substantive change in this revision)
-----------------------------------------------------
The original pipeline built the hybrid's training features from
``base_model.predict(X_train)`` -- in-sample predictions of a tree already fitted
to those exact rows. A gradient-boosted ensemble fits its training set far more
closely than unseen data, so the residual head was trained against a prior of
unrealistically high quality and then evaluated against a materially worse one.
That is a train/serve distribution mismatch of the classic stacking kind: it
inflates the apparent difficulty of the residual task during training and
degrades the correction at test time.

:func:`out_of_fold_prior` replaces those in-sample predictions with out-of-fold
predictions from an inner K-fold, which is the standard stacked-generalisation
construction. The prior the head trains on now has the same error distribution
as the prior it will see at inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import clone
from sklearn.model_selection import KFold

from .. import config
from .losses import hybrid_objective


@dataclass
class HybridConfig:
    """Switches for the ablation study."""

    use_structural_layer: bool = True
    use_quantile_heads: bool = True
    use_residual_head: bool = True
    use_physics_loss: bool = True
    use_anisotropy_loss: bool = True
    out_of_fold_prior: bool = True

    #: Learn a scalar gate on the correction, initialised at zero.
    #:
    #: Without it the head applies a correction unconditionally, so a noisy
    #: correction can only add variance to an already-good prior -- which is what
    #: the measured ablation shows happening. With the gate the model starts as
    #: an identity map onto the prior and must earn any departure from it, so the
    #: hybrid degrades gracefully to its base learner instead of below it.
    use_shrinkage_gate: bool = False


class HybridResidualNet(nn.Module):
    """Neural correction head operating on ``[x, x * prior, prior]``."""

    def __init__(self, raw_dim: int, channel_a: int, channel_b: int,
                 cfg: HybridConfig) -> None:
        super().__init__()
        self.raw_dim = raw_dim
        self.cfg = cfg

        if cfg.use_structural_layer:
            from .structural_layer import StructuralCouplingLayer

            self.structural = StructuralCouplingLayer(raw_dim, channel_a, channel_b)
        else:
            self.structural = None
            self.channel_a, self.channel_b = channel_a, channel_b

        self.encoder = nn.Sequential(
            nn.Linear(raw_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.quantile_heads = nn.Linear(32, 3) if cfg.use_quantile_heads else None
        self.residual_head = (
            nn.Sequential(nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
            if cfg.use_residual_head
            else None
        )
        if self.quantile_heads is None and self.residual_head is None:
            raise ValueError("hybrid needs at least one prediction head")

        # Shrinkage gate, parameterised as sigmoid(raw) and initialised at
        # raw = -3 so the gate opens at ~0.05. A bare unconstrained scalar was
        # tried first and behaved badly: it is free to take negative values, so
        # the correction can flip sign fold to fold (measured range -0.26 to
        # +0.28) and the model is no more stable than the ungated version.
        # Constraining the gate to (0, 1) makes "apply no correction" the
        # default the optimiser must actively move away from.
        self.correction_gate_raw = (
            nn.Parameter(torch.full((1,), -3.0)) if cfg.use_shrinkage_gate else None
        )

    @property
    def correction_gate(self) -> torch.Tensor | None:
        if self.correction_gate_raw is None:
            return None
        return torch.sigmoid(self.correction_gate_raw)

    def forward(self, x_hybrid: torch.Tensor):
        d = self.raw_dim
        x_raw = x_hybrid[:, :d]
        interactions = x_hybrid[:, d : 2 * d]
        prior = x_hybrid[:, -1:]

        if self.structural is not None:
            x_struct, aspect_ratio = self.structural(x_raw)
        else:
            x_struct = x_raw
            a = x_raw[:, self.channel_a : self.channel_a + 1]
            b = x_raw[:, self.channel_b : self.channel_b + 1]
            aspect_ratio = a / (b + 1e-6)

        latent = self.encoder(torch.cat([x_struct, interactions], dim=-1))

        quantiles = (
            self.quantile_heads(latent)
            if self.quantile_heads is not None
            else torch.zeros(len(latent), 3, dtype=latent.dtype)
        )

        correction = torch.zeros_like(prior)
        if self.quantile_heads is not None:
            q25, q50, q75 = quantiles[:, 0:1], quantiles[:, 1:2], quantiles[:, 2:3]
            gate = torch.sigmoid(q75 - q25)
            correction = correction + 0.25 * (q50 + 0.1 * gate * (q75 - q25))
        if self.residual_head is not None:
            correction = correction + 0.75 * self.residual_head(latent)

        gate = self.correction_gate
        if gate is not None:
            correction = correction * gate

        return prior + correction, quantiles, aspect_ratio


def out_of_fold_prior(
    base_estimator, X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> np.ndarray:
    """Out-of-fold predictions of ``base_estimator`` over the training rows.

    This is what the residual head must be trained against; see the module
    docstring for why in-sample predictions bias the hybrid.
    """
    oof = np.zeros(len(X), dtype=float)
    inner = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in inner.split(X):
        model = clone(base_estimator).fit(X[tr], y[tr].ravel())
        oof[va] = model.predict(X[va])
    return oof


def assemble(X: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Build ``[x, x * prior, prior]`` with the prior standardised."""
    prior = prior.reshape(-1, 1)
    return np.hstack([X, X * prior, prior]).astype(np.float32)


def train_hybrid(
    base_estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    channel_a: int,
    channel_b: int,
    cfg: HybridConfig,
    seed: int,
    epochs: int = 1000,
    patience: int = 80,
):
    """Fit the hybrid and return ``(model, valid_features, prior_scaler)``."""
    raw_dim = X_train.shape[1]

    # --- base prior ------------------------------------------------------
    fitted_base = clone(base_estimator).fit(X_train, y_train.ravel())
    if cfg.out_of_fold_prior:
        train_prior = out_of_fold_prior(
            base_estimator, X_train, y_train, config.INNER_SPLITS, seed
        )
    else:
        # Retained only so the ablation can quantify the cost of the original,
        # leaky construction.
        train_prior = fitted_base.predict(X_train)
    valid_prior = fitted_base.predict(X_valid)

    mu, sigma = float(train_prior.mean()), float(train_prior.std()) or 1.0
    train_feats = assemble(X_train, (train_prior - mu) / sigma)
    valid_feats = assemble(X_valid, (valid_prior - mu) / sigma)

    # --- correction head -------------------------------------------------
    model = HybridResidualNet(raw_dim, channel_a, channel_b, cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=30)
    mse = nn.MSELoss()

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_feats))
    n_val = max(40, int(0.15 * len(perm)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xt = torch.tensor(train_feats[tr_idx])
    yt = torch.tensor(y_train[tr_idx], dtype=torch.float32).view(-1, 1)
    Xv = torch.tensor(train_feats[val_idx])
    yv = torch.tensor(y_train[val_idx], dtype=torch.float32).view(-1, 1)

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_loss, stale = float("inf"), 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred, quantiles, aspect = model(Xt)
        data_loss = mse(pred, yt)
        weight = 1e-3 * (data_loss.detach() + 1e-5)
        loss = data_loss + weight * hybrid_objective(
            pred, quantiles, yt, aspect,
            use_quantiles=cfg.use_quantile_heads,
            use_physics=cfg.use_physics_loss,
            use_anisotropy=cfg.use_anisotropy_loss,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(Xv)[0], yv).item()
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, valid_feats, fitted_base


@torch.no_grad()
def predict_hybrid(model: nn.Module, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(point_prediction, quantiles)``."""
    model.eval()
    pred, quantiles, _ = model(torch.tensor(features, dtype=torch.float32))
    return pred.numpy().ravel(), quantiles.numpy()
