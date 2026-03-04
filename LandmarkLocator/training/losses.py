"""Loss functions for heatmap-based landmark detection."""

from typing import Optional

import torch
import torch.nn as nn


class HeatmapMSELoss(nn.Module):
    """MSE loss on heatmaps with optional per-channel masking."""

    def __init__(self) -> None:
        """Initialize loss."""
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked MSE between predicted and target heatmaps.

        Args:
            pred: (B, C, H, W) predicted heatmaps
            target: (B, C, H, W) target heatmaps
            mask: (B, C) optional binary mask (1 = compute loss, 0 = ignore)
        """
        diff_sq = (pred - target) ** 2

        if mask is not None:
            # Expand mask to spatial dims: (B, C) → (B, C, 1, 1)
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            diff_sq = diff_sq * mask

        return diff_sq.mean()
