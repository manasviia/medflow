"""Unit tests for model architecture."""

import torch

from medflow.models.heads.multilabel import MCDropoutHead, MultiLabelHead


def test_multilabel_head_output_shape():
    """Head should produce (B, num_classes) probability tensor."""
    head = MultiLabelHead(in_features=256, num_classes=14)
    x = torch.randn(4, 256)
    out = head(x)
    assert out.shape == (4, 14), f"Expected (4, 14), got {out.shape}"


def test_multilabel_head_probabilities_in_range():
    """Sigmoid output should be strictly in (0, 1)."""
    head = MultiLabelHead(in_features=256, num_classes=14)
    x = torch.randn(8, 256)
    out = head(x)
    assert out.min() >= 0.0 and out.max() <= 1.0, "Probabilities must be in [0, 1]"


def test_mc_dropout_uncertainty_shape():
    """MC Dropout should return mean and uncertainty of correct shape."""
    head = MCDropoutHead(in_features=256, num_classes=14, mc_samples=5)
    x = torch.randn(4, 256)
    mean, uncertainty = head.predict_with_uncertainty(x)
    assert mean.shape == (4, 14)
    assert uncertainty.shape == (4, 14)


def test_mc_dropout_uncertainty_is_nonnegative():
    """Uncertainty (std dev) should always be >= 0."""
    head = MCDropoutHead(in_features=256, num_classes=14, mc_samples=10)
    x = torch.randn(4, 256)
    _, uncertainty = head.predict_with_uncertainty(x)
    assert (uncertainty >= 0).all(), "Uncertainty (std) must be non-negative"


def test_mc_dropout_produces_variance():
    """Different MC samples should produce different predictions (dropout is active)."""
    head = MCDropoutHead(in_features=256, num_classes=14, mc_samples=10)
    x = torch.randn(2, 256)
    _, uncertainty = head.predict_with_uncertainty(x)
    # With dropout active, there should be nonzero variance for most samples
    assert uncertainty.mean() > 0, "MC Dropout should produce nonzero uncertainty"
