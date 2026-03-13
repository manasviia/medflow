"""Unit tests for image transform pipelines."""

import torch
from PIL import Image

from medflow.data.transforms.imaging import get_train_transforms, get_val_transforms


def _dummy_image(size: int = 256) -> Image.Image:
    """Create a dummy RGB image for testing."""
    import numpy as np
    arr = (torch.rand(size, size, 3) * 255).byte().numpy()
    return Image.fromarray(arr, mode="RGB")


def test_val_transforms_output_shape():
    """Val transform should produce a (3, H, W) tensor."""
    transform = get_val_transforms(image_size=224)
    img = _dummy_image()
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"


def test_train_transforms_output_shape():
    """Train transform should produce a (3, H, W) tensor."""
    transform = get_train_transforms(image_size=224)
    img = _dummy_image()
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"


def test_val_transforms_are_deterministic():
    """Same image through val transform should always give same tensor."""
    transform = get_val_transforms(image_size=224)
    img = _dummy_image()
    t1 = transform(img)
    t2 = transform(img)
    assert torch.allclose(t1, t2), "Val transforms should be deterministic"


def test_transforms_produce_normalized_values():
    """Output values should be roughly in [-3, 3] after ImageNet normalization."""
    transform = get_val_transforms(image_size=224)
    img = _dummy_image()
    tensor = transform(img)
    assert tensor.min() >= -4.0
    assert tensor.max() <= 4.0
