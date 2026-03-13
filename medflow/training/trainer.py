"""Training loop with MLflow experiment tracking and checkpointing.

Encapsulates the full train/eval cycle so train scripts stay thin —
just instantiate Trainer and call .fit().

Design principles:
- All hyperparameters come from config, never hardcoded here
- Every run is logged to MLflow: params, per-epoch metrics, model artifact
- Best checkpoint saved based on val AUC (not loss)
- Mixed precision training via torch.amp for 2x speedup on modern GPUs
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from medflow.training.metrics import AUCMeter


class Trainer:
    """Manages the full training lifecycle for ChestXrayModel.

    Args:
        model: The model to train.
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split.
        loss_fn: Loss function (WeightedBCE or FocalLoss).
        device: torch.device to train on.
        epochs: Total training epochs.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: Linear warmup before cosine decay.
        gradient_clip: Max gradient norm (0 = disabled).
        mixed_precision: Enable torch.amp mixed precision.
        checkpoint_dir: Directory to save best model checkpoint.
        mlflow_experiment: MLflow experiment name.
        mlflow_run_name: MLflow run name (optional).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        gradient_clip: float = 1.0,
        mixed_precision: bool = True,
        checkpoint_dir: str = "checkpoints",
        mlflow_experiment: str = "medflow",
        mlflow_run_name: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
        )
        self.scaler = GradScaler(enabled=mixed_precision)
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)

        self.mlflow_experiment = mlflow_experiment
        self.mlflow_run_name = mlflow_run_name
        self.best_val_auc = 0.0

    def _warmup_step(self, epoch: int) -> None:
        """Apply linear LR warmup for first N epochs."""
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * scale if "initial_lr" in pg else pg["lr"] * scale

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            self.scaler.scale(loss).backward()

            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.debug(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f}"
                )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, dict[str, float]]:
        """Run validation. Returns (mean_loss, auc_results)."""
        self.model.eval()
        total_loss = 0.0
        meter = AUCMeter()

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.mixed_precision):
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            total_loss += loss.item()
            meter.update(preds, labels)

        auc_results = meter.compute()
        return total_loss / len(self.val_loader), auc_results

    def _save_checkpoint(self, epoch: int, val_auc: float) -> None:
        """Save model checkpoint if val AUC improved."""
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_auc": val_auc,
                },
                path,
            )
            logger.info(f"New best checkpoint saved (val_auc={val_auc:.4f}) → {path}")
            mlflow.log_artifact(str(path), artifact_path="checkpoints")

    def fit(self) -> None:
        """Run the full training loop with MLflow logging."""
        mlflow.set_experiment(self.mlflow_experiment)

        with mlflow.start_run(run_name=self.mlflow_run_name):
            # Log hyperparameters
            mlflow.log_params(
                {
                    "epochs": self.epochs,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "warmup_epochs": self.warmup_epochs,
                    "mixed_precision": self.mixed_precision,
                    "gradient_clip": self.gradient_clip,
                }
            )

            param_counts = self.model.count_parameters()
            mlflow.log_params(param_counts)

            logger.info(
                f"Training started — {param_counts['trainable']:,} trainable params, "
                f"{self.epochs} epochs"
            )

            for epoch in range(self.epochs):
                # LR warmup
                if epoch < self.warmup_epochs:
                    self._warmup_step(epoch)

                train_loss = self._train_epoch(epoch)
                val_loss, auc_results = self._val_epoch()

                # Step scheduler after warmup
                if epoch >= self.warmup_epochs:
                    self.scheduler.step()

                mean_auc = auc_results["mean_auc"]
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Log to MLflow
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_mean_auc": mean_auc,
                        "lr": current_lr,
                        **{f"auc_{k}": v for k, v in auc_results.items() if k != "mean_auc"},
                    },
                    step=epoch,
                )

                self._save_checkpoint(epoch, mean_auc)

                logger.info(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_auc={mean_auc:.4f} | "
                    f"lr={current_lr:.2e}"
                )

            logger.info(f"Training complete. Best val AUC: {self.best_val_auc:.4f}")
            mlflow.log_metric("best_val_auc", self.best_val_auc)
