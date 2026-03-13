"""Full CXR model: backbone + multi-label head + uncertainty estimation.

This is the top-level model class that wires everything together.
Training code and serving code both import this class — nothing else.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from medflow.models.backbones.efficientnet import EfficientNetBackbone
from medflow.models.heads.multilabel import MCDropoutHead


class ChestXrayModel(nn.Module):
    """End-to-end chest X-ray pathology detection model.

    Combines a pretrained EfficientNet backbone with an uncertainty-aware
    multi-label classification head. Produces both pathology probabilities
    and calibrated uncertainty estimates via Monte Carlo Dropout.

    Args:
        backbone_name: timm model name for the feature extractor.
        pretrained: Load ImageNet pretrained backbone weights.
        num_classes: Number of pathology classes (14 for NIH ChestX-ray14).
        dropout_rate: Dropout rate in classification head.
        mc_samples: Number of MC Dropout forward passes for uncertainty.
        frozen_stages: Backbone stages to freeze during training.

    Example:
        >>> model = ChestXrayModel()
        >>> images = torch.randn(4, 3, 224, 224)
        >>> probs = model(images)                          # (4, 14)
        >>> mean, uncertainty = model.predict(images)     # (4, 14), (4, 14)
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        num_classes: int = 14,
        dropout_rate: float = 0.3,
        mc_samples: int = 20,
        frozen_stages: int = 0,
    ) -> None:
        super().__init__()

        self.backbone = EfficientNetBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
        )

        self.head = MCDropoutHead(
            in_features=self.backbone.output_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            mc_samples=mc_samples,
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass — returns class probabilities.

        Used during training (no uncertainty, faster).

        Args:
            x: Image batch of shape (B, 3, H, W).

        Returns:
            Class probabilities of shape (B, num_classes).
        """
        features = self.backbone(x)
        return self.head(features)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference forward pass with uncertainty estimation.

        Runs MC Dropout to produce both predictions and uncertainty.
        Use this at inference time, not forward().

        Args:
            x: Image batch of shape (B, 3, H, W).

        Returns:
            mean_probs: Mean class probabilities, shape (B, num_classes).
            uncertainty: Epistemic uncertainty per class, shape (B, num_classes).
        """
        features = self.backbone(x)
        return self.head.predict_with_uncertainty(features)

    def count_parameters(self) -> dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
