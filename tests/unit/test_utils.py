"""Unit tests for utility modules."""

import torch

from medflow.utils.reproducibility import get_device, seed_everything


def test_seed_everything_is_deterministic():
    """Same seed should produce same random tensors."""
    seed_everything(42)
    t1 = torch.rand(5)

    seed_everything(42)
    t2 = torch.rand(5)

    assert torch.allclose(t1, t2), "seed_everything should produce deterministic outputs"


def test_get_device_returns_torch_device():
    """get_device should always return a valid torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")
