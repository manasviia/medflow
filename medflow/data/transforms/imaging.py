"""Chest X-ray specific image transforms.

Chest X-rays require different augmentation strategies than natural images:
- No aggressive color jitter (grayscale diagnostic images)
- Conservative geometric transforms (pathologies are location-sensitive)
- Histogram normalization (scanner variability across institutions)
"""

from torchvision import transforms as T


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Augmentation pipeline for training.

    Conservative augmentations appropriate for clinical imaging:
    random horizontal flip (anatomically valid), small rotations,
    and slight brightness/contrast shifts to simulate scanner variance.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        # ImageNet stats — reasonable starting point for pretrained backbones
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline for validation and inference.

    No augmentation — only resize and normalize.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


__all__ = ["get_train_transforms", "get_val_transforms"]
