"""Multi-label classification head with Monte Carlo Dropout.

Standard multi-label classification predicts independent binary probabilities
for each class via sigmoid activation (unlike softmax which forces competition).

Monte Carlo Dropout (Gal & Ghahramani, 2016) approximates Bayesian inference
by keeping dropout active at inference time and running multiple forward passes.
The variance across passes gives a calibrated uncertainty estimate — critical
for clinical AI where knowing *when the model is uncertain* is as important
as the predictions themselves.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiLabelHead(nn.Module):
    """MLP classification head for multi-label prediction.

    Architecture: Linear → BN → ReLU → Dropout → Linear → Sigmoid

    Args:
        in_features: Input feature dimension (from backbone).
        num_classes: Number of output classes.
        hidden_dim: Hidden layer width.
        dropout_rate: Dropout probability (also used for MC Dropout at inference).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 14,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize final layer with small weights for stable early training."""
        final_linear = self.classifier[-1]
        nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-class probabilities via sigmoid.

        Args:
            x: Feature tensor of shape (B, in_features).

        Returns:
            Probability tensor of shape (B, num_classes) in [0, 1].
        """
        return torch.sigmoid(self.classifier(x))


class MCDropoutHead(MultiLabelHead):
    """Multi-label head with Monte Carlo Dropout uncertainty estimation.

    At inference time, call predict_with_uncertainty() instead of forward()
    to get both mean predictions and epistemic uncertainty estimates.
    """

    def __init__(self, *args, mc_samples: int = 20, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mc_samples = mc_samples

    def _enable_dropout(self) -> None:
        """Force dropout layers to stay active during inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MC Dropout to estimate prediction uncertainty.

        Performs `mc_samples` stochastic forward passes with dropout active,
        then returns the mean prediction and standard deviation across passes.

        Args:
            x: Feature tensor of shape (B, in_features).

        Returns:
            mean_preds: Mean probabilities, shape (B, num_classes).
            uncertainty: Std deviation across samples, shape (B, num_classes).
                Higher values indicate the model is uncertain about that class.
        """
        self.eval()
        self._enable_dropout()  # Keep dropout on despite eval mode

        samples = torch.stack(
            [torch.sigmoid(self.classifier(x)) for _ in range(self.mc_samples)],
            dim=0,
        )  # (mc_samples, B, num_classes)

        mean_preds = samples.mean(dim=0)  # (B, num_classes)
        uncertainty = samples.std(dim=0)  # (B, num_classes)

        return mean_preds, uncertainty
