"""Loss terms: boundary penalty, pinball loss, and the anisotropy regulariser."""
from __future__ import annotations

import torch


def boundary_physics_penalty(y_pred: torch.Tensor) -> torch.Tensor:
    """Soft penalty discouraging negative or near-zero band-gap predictions.

    Band gaps in this dataset are strictly positive (minimum 0.001 eV), so a
    negative prediction is unphysical.
    """
    negative = torch.relu(-y_pred)
    near_zero = torch.exp(torch.relu(0.05 - y_pred)) - 1.0
    return torch.mean(negative + 0.1 * near_zero)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Standard quantile (pinball) loss at level ``tau``."""
    residual = target - pred
    return torch.mean(torch.max(tau * residual, (tau - 1.0) * residual))


def anisotropy_penalty(y_pred: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
    """Penalise predictions falling below a floor set by structural aspect ratio.

    NOTE ON INTERPRETATION: this is a monotonic coupling between a geometric
    ratio and the predicted gap. It is a soft inductive bias, not a derivation
    from elasticity theory, and the manuscript should describe it as such.
    """
    return torch.mean(torch.relu(0.1 * aspect_ratio - y_pred))


def hybrid_objective(
    y_pred: torch.Tensor,
    quantiles: torch.Tensor,
    y_true: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    *,
    use_quantiles: bool = True,
    use_physics: bool = True,
    use_anisotropy: bool = True,
) -> torch.Tensor:
    """Regularisation term of the hybrid objective.

    The three switches exist so the ablation study can disable each component
    without redefining the model.
    """
    total = torch.zeros((), dtype=y_pred.dtype)

    if use_quantiles:
        q25, q50, q75 = quantiles[:, 0:1], quantiles[:, 1:2], quantiles[:, 2:3]
        total = total + 0.5 * (
            pinball_loss(q25, y_true, 0.25) + pinball_loss(q75, y_true, 0.75)
        ) + pinball_loss(q50, y_true, 0.50)

    if use_physics:
        total = total + boundary_physics_penalty(y_pred)

    if use_anisotropy and aspect_ratio is not None:
        total = total + anisotropy_penalty(y_pred, aspect_ratio)

    return total
