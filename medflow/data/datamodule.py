"""DataModule: manages train/val/test splits and DataLoaders.

Encapsulates all data loading logic in one place so training scripts
stay clean — just call datamodule.train_loader() and go.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from medflow.data.loaders.chestxray import ChestXray14Dataset
from medflow.data.transforms.imaging import get_train_transforms, get_val_transforms
from medflow.utils.logging import logger


class ChestXray14DataModule:
    """Manages dataset splits, transforms, and DataLoader creation.

    Args:
        root_dir: Root directory with images/ subfolder and metadata CSV.
        image_size: Square image resolution.
        batch_size: Samples per batch.
        num_workers: Parallel workers for DataLoader prefetching.
        loader_workers: Thread pool size for parallel image pre-loading.
        preload: Pre-load all images into RAM for faster epoch iteration.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 8,
        loader_workers: int = 8,
        preload: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader_workers = loader_workers
        self.preload = preload

        self._train_dataset: ChestXray14Dataset | None = None
        self._val_dataset: ChestXray14Dataset | None = None
        self._test_dataset: ChestXray14Dataset | None = None

    def setup(self) -> None:
        """Instantiate all three dataset splits."""
        logger.info("Setting up ChestXray14 datasets...")

        split_dir = self.root_dir / "splits"

        self._train_dataset = ChestXray14Dataset(
            root_dir=self.root_dir,
            split_file=split_dir / "train.txt",
            image_size=self.image_size,
            transform=get_train_transforms(self.image_size),
            num_workers=self.loader_workers,
            preload=self.preload,
        )
        self._val_dataset = ChestXray14Dataset(
            root_dir=self.root_dir,
            split_file=split_dir / "val.txt",
            image_size=self.image_size,
            transform=get_val_transforms(self.image_size),
            num_workers=self.loader_workers,
            preload=self.preload,
        )
        self._test_dataset = ChestXray14Dataset(
            root_dir=self.root_dir,
            split_file=split_dir / "test.txt",
            image_size=self.image_size,
            transform=get_val_transforms(self.image_size),
            num_workers=self.loader_workers,
            preload=self.preload,
        )

        logger.info(
            f"Datasets ready — "
            f"train: {len(self._train_dataset)}, "
            f"val: {len(self._val_dataset)}, "
            f"test: {len(self._test_dataset)}"
        )

    def train_loader(self) -> DataLoader:
        assert self._train_dataset is not None, "Call setup() first"
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_loader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup() first"
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_loader(self) -> DataLoader:
        assert self._test_dataset is not None, "Call setup() first"
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def num_classes(self) -> int:
        return 14

    @property
    def class_names(self) -> list[str]:
        assert self._train_dataset is not None, "Call setup() first"
        return self._train_dataset.class_names

    @property
    def class_weights(self):
        assert self._train_dataset is not None, "Call setup() first"
        return self._train_dataset.class_weights
