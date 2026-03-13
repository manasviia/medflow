"""Utilities for reproducible experiments."""

import random

import numpy as np
import torch
from loguru import logger


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic ops — slight perf cost but fully reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"All random seeds set to {seed}")


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device


__all__ = ["seed_everything", "get_device"]
