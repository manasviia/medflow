"""Loss functions for multi-label classification.

Standard BCE works but NIH ChestX-ray14 is heavily imbalanced — "No Finding"
accounts for ~53% of labels. Weighted BCE and Focal Loss both address this.

Focal Loss (Lin et al., 2017) is particularly effective: it down-weights
easy negatives so the model focuses learning on hard, misclassified examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with per-class positive weights.

    Assigns higher loss to positive examples of rare classes, counteracting
    the class imbalance in NIH ChestX-ray14.

    Args:
        pos_weights: Tensor of shape (num_classes,) — ratio of negatives to
            positives per class. Computed from dataset.class_weights.
    """

    def __init__(self, pos_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("pos_weights", pos_weights)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Sigmoid probabilities, shape (B, num_classes).
            targets: Binary labels, shape (B, num_classes).
        """
        return F.binary_cross_entropy_with_logits(
            torch.logit(preds.clamp(1e-6, 1 - 1e-6)),
            targets,
            pos_weight=self.pos_weights,
        )


class FocalLoss(nn.Module):
    """Focal loss for multi-label classification.

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    gamma > 0 reduces loss for well-classified examples, forcing the model
    to focus on hard negatives. gamma=2, alpha=0.25 are standard defaults.

    Args:
        alpha: Weighting factor for positive class.
        gamma: Focusing parameter — higher = more focus on hard examples.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Sigmoid probabilities, shape (B, num_classes).
            targets: Binary labels, shape (B, num_classes).
        """
        bce = F.binary_cross_entropy(preds, targets, reduction="none")
        p_t = preds * targets + (1 - preds) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
