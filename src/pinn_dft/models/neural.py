"""Neural baselines: a plain deep MLP and a standalone physics-informed network."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .losses import boundary_physics_penalty


class DeepMLP(nn.Module):
    """Feed-forward baseline with batch normalisation."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    physics_weight: float,
    seed: int,
    patience: int = 80,
) -> nn.Module:
    """Shared training loop with an internal validation split and early stopping.

    The split is drawn from a fold-specific generator; the original code reused a
    single fixed permutation for every fold.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    n_val = max(40, int(0.15 * len(perm)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xt = torch.tensor(X[tr_idx], dtype=torch.float32)
    yt = torch.tensor(y[tr_idx], dtype=torch.float32).view(-1, 1)
    Xv = torch.tensor(X[val_idx], dtype=torch.float32)
    yv = torch.tensor(y[val_idx], dtype=torch.float32).view(-1, 1)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=25)
    mse = nn.MSELoss()

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_loss, stale = float("inf"), 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(Xt)
        loss = mse(pred, yt) + physics_weight * boundary_physics_penalty(pred)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(Xv), yv).item()
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
    return model


def train_mlp(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = 1200) -> nn.Module:
    return _fit(DeepMLP(X.shape[1]), X, y, epochs=epochs, lr=5e-3,
                physics_weight=0.0, seed=seed)


def train_pinn(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = 1500) -> nn.Module:
    """Same backbone as the MLP, trained with the non-negativity penalty."""
    return _fit(DeepMLP(X.shape[1]), X, y, epochs=epochs, lr=5e-3,
                physics_weight=1e-3, seed=seed)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()
