"""Integration tests for the end-to-end inference pipeline.

These tests exercise the full stack: image → transform → model → response.
They don't require a trained checkpoint — random weights are sufficient to
verify the pipeline is wired correctly.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from medflow.data.transforms.imaging import get_val_transforms
from medflow.models.chestxray_model import ChestXrayModel


@pytest.fixture(scope="module")
def model() -> ChestXrayModel:
    """Instantiate model with random weights (no checkpoint needed)."""
    m = ChestXrayModel(pretrained=False, mc_samples=5)
    m.eval()
    return m


@pytest.fixture
def dummy_image_tensor() -> torch.Tensor:
    """224x224 RGB tensor ready for inference."""
    transform = get_val_transforms(224)
    img = Image.fromarray((torch.rand(224, 224, 3) * 255).byte().numpy(), mode="RGB")
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)


def test_forward_pass_shape(model, dummy_image_tensor):
    """Forward pass should return (1, 14) probability tensor."""
    with torch.no_grad():
        out = model(dummy_image_tensor)
    assert out.shape == (1, 14), f"Expected (1, 14), got {out.shape}"


def test_forward_pass_probabilities_valid(model, dummy_image_tensor):
    """All output values should be valid probabilities in [0, 1]."""
    with torch.no_grad():
        out = model(dummy_image_tensor)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_predict_returns_uncertainty(model, dummy_image_tensor):
    """predict() should return mean probs and uncertainty of correct shape."""
    mean_probs, uncertainty = model.predict(dummy_image_tensor)
    assert mean_probs.shape == (1, 14)
    assert uncertainty.shape == (1, 14)
    assert (uncertainty >= 0).all()


def test_batch_inference(model):
    """Model should handle variable batch sizes correctly."""
    for batch_size in [1, 4, 8]:
        batch = torch.rand(batch_size, 3, 224, 224)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (
            batch_size,
            14,
        ), f"Batch size {batch_size}: expected ({batch_size}, 14), got {out.shape}"


def test_count_parameters(model):
    """Model should report nonzero trainable parameters."""
    counts = model.count_parameters()
    assert counts["total"] > 0
    assert counts["trainable"] > 0
    assert counts["total"] >= counts["trainable"]
