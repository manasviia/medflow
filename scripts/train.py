"""Training entrypoint.

Usage:
    python scripts/train.py                          # uses configs/base.yaml
    python scripts/train.py --config-name experiment_baseline
    python scripts/train.py training.lr=3e-4 training.epochs=30
"""

import hydra
from loguru import logger
from omegaconf import DictConfig

from medflow.data.datamodule import ChestXray14DataModule
from medflow.models.chestxray_model import ChestXrayModel
from medflow.training.losses import FocalLoss
from medflow.training.trainer import Trainer
from medflow.utils.logging import setup_logger
from medflow.utils.reproducibility import get_device, seed_everything


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logger(log_dir=cfg.logging.log_dir, level=cfg.logging.level)
    seed_everything(cfg.training.seed)
    device = get_device()

    logger.info(f"Config:\n{cfg}")

    # Data
    datamodule = ChestXray14DataModule(
        root_dir=cfg.data.root_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        loader_workers=cfg.data.num_workers,
    )
    datamodule.setup()

    # Model
    model = ChestXrayModel(
        backbone_name=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        dropout_rate=cfg.model.dropout_rate,
        mc_samples=cfg.model.mc_dropout_samples,
    )

    # Loss — focal loss handles class imbalance without needing precomputed weights
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=datamodule.train_loader(),
        val_loader=datamodule.val_loader(),
        loss_fn=loss_fn,
        device=device,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        gradient_clip=cfg.training.gradient_clip,
        mixed_precision=cfg.training.mixed_precision,
        mlflow_experiment=cfg.mlflow.experiment_name,
        mlflow_run_name=cfg.mlflow.get("run_name"),
    )

    trainer.fit()


if __name__ == "__main__":
    main()
