"""Unit tests for training components."""

import torch

from medflow.training.losses import FocalLoss, WeightedBCELoss
from medflow.training.metrics import AUCMeter

# ── Loss function tests ────────────────────────────────────────────────────────


def test_focal_loss_returns_scalar():
    preds = torch.sigmoid(torch.randn(8, 14))
    targets = torch.randint(0, 2, (8, 14)).float()
    loss = FocalLoss()(preds, targets)
    assert loss.ndim == 0, "Loss should be a scalar"


def test_focal_loss_is_nonnegative():
    preds = torch.sigmoid(torch.randn(8, 14))
    targets = torch.randint(0, 2, (8, 14)).float()
    loss = FocalLoss()(preds, targets)
    assert loss.item() >= 0.0


def test_weighted_bce_returns_scalar():
    pos_weights = torch.ones(14) * 5.0
    preds = torch.sigmoid(torch.randn(8, 14))
    targets = torch.randint(0, 2, (8, 14)).float()
    loss = WeightedBCELoss(pos_weights)(preds, targets)
    assert loss.ndim == 0


def test_focal_loss_perfect_predictions_low_loss():
    """Near-perfect predictions should produce near-zero focal loss."""
    preds = torch.full((8, 14), 0.999)
    targets = torch.ones(8, 14)
    loss = FocalLoss()(preds, targets)
    assert loss.item() < 0.05, f"Expected low loss, got {loss.item():.4f}"


# ── AUC meter tests ────────────────────────────────────────────────────────────


def test_auc_meter_mean_auc_in_range():
    """Mean AUC should be between 0 and 1."""
    meter = AUCMeter()
    preds = torch.rand(50, 14)
    targets = torch.randint(0, 2, (50, 14)).float()
    meter.update(preds, targets)
    results = meter.compute()
    assert 0.0 <= results["mean_auc"] <= 1.0


def test_auc_meter_reset_clears_state():
    """After reset, meter should return fresh results."""
    meter = AUCMeter()
    preds = torch.rand(50, 14)
    targets = torch.randint(0, 2, (50, 14)).float()
    meter.update(preds, targets)
    meter.reset()
    assert len(meter._preds) == 0
    assert len(meter._targets) == 0


def test_auc_meter_perfect_classifier():
    """A perfect classifier should achieve AUC close to 1.0."""
    meter = AUCMeter()
    # Create perfectly separable predictions
    targets = torch.zeros(100, 14)
    targets[:50] = 1.0
    preds = torch.zeros(100, 14)
    preds[:50] = 1.0
    meter.update(preds, targets)
    results = meter.compute()
    assert results["mean_auc"] > 0.99, f"Expected ~1.0 AUC, got {results['mean_auc']:.4f}"
