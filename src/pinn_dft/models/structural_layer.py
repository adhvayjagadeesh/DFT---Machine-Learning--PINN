"""Structural coupling layer (formerly 'Poisson-Coupled Elastic Strain Layer').

The layer forms a small set of symmetric, sign-aware interaction terms from two
geometric channels ``(g1, g2)`` and adds a learned projection of them back onto
the feature map::

    s = [g1 * g2,  g1^2 - g2^2,  sqrt(g1^2 + g2^2 + eps),  g1,  g2]
    x' = x + tanh(W2 * SiLU(W1 s + b1) + b2)

Two corrections relative to the original implementation:

* The channels are selected **by name** (see :data:`config.GEOMETRIC_CHANNELS`).
  The previous version used positional indices 0 and 1, which in the assembled
  feature matrix were 'Energy above hull' and 'Heat of formation' -- two
  formation energies, not geometric quantities. Every physical claim made about
  the layer was therefore describing a transform of the wrong inputs.
* The class name no longer asserts a Poisson/elasticity derivation. The terms
  ``g1*g2`` and ``g1^2 - g2^2`` are a generic symmetric/antisymmetric pair; they
  are not derived from a Poisson relation, and calling them one is a claim the
  code does not support.

With the current C2DB export the available geometric channels are unit-cell area
and thickness. Recovering true in-plane lattice vectors ``a`` and ``b`` -- which
is what an anisotropy argument actually requires -- needs a re-pull from C2DB;
see README, "Known limitations".
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StructuralCouplingLayer(nn.Module):
    """Learned low-rank coupling between two geometric channels and the feature map."""

    def __init__(self, raw_dim: int, channel_a: int, channel_b: int, hidden: int = 24) -> None:
        super().__init__()
        self.channel_a = channel_a
        self.channel_b = channel_b
        self.mapping = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.projection = nn.Linear(hidden, raw_dim)

    def forward(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = x_raw[:, self.channel_a : self.channel_a + 1]
        b = x_raw[:, self.channel_b : self.channel_b + 1]

        cross = a * b
        asymmetry = a**2 - b**2
        magnitude = torch.sqrt(a**2 + b**2 + 1e-6)
        aspect_ratio = a / (b + 1e-6)

        descriptor = torch.cat([cross, asymmetry, magnitude, a, b], dim=-1)
        transformed = x_raw + torch.tanh(self.projection(self.mapping(descriptor)))
        return transformed, aspect_ratio
