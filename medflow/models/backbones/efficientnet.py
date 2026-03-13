"""EfficientNet backbone via timm.

Wraps a pretrained EfficientNet-B4 and exposes only the feature extraction
stage — classification head is removed so our custom multi-label head can
be attached downstream.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class EfficientNetBackbone(nn.Module):
    """Pretrained EfficientNet feature extractor.

    Loads weights from timm and strips the classification head,
    returning a feature vector of shape (B, feature_dim).

    Args:
        model_name: Any timm EfficientNet variant (e.g. "efficientnet_b4").
        pretrained: Load ImageNet pretrained weights.
        frozen_stages: Number of early stages to freeze (0 = train all).
            Freezing early stages is useful when fine-tuning on small datasets.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        frozen_stages: int = 0,
    ) -> None:
        super().__init__()

        # num_classes=0 removes the classifier head, returns feature vector
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.feature_dim: int = self.backbone.num_features

        if frozen_stages > 0:
            self._freeze_stages(frozen_stages)

    def _freeze_stages(self, n: int) -> None:
        """Freeze the first n stages of the backbone."""
        stages = list(self.backbone.children())
        for stage in stages[:n]:
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Image tensor of shape (B, 3, H, W).

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        return self.backbone(x)

    @property
    def output_dim(self) -> int:
        return self.feature_dim
