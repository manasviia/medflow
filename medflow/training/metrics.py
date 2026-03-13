"""Evaluation metrics for multi-label chest X-ray classification.

AUC-ROC is the standard metric for this task — it measures discrimination
ability across all thresholds, which matters more than accuracy when classes
are heavily imbalanced (a model predicting all-negative gets 94% accuracy
on NIH ChestX-ray14 but is clinically useless).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PATHOLOGY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


class AUCMeter:
    """Accumulates predictions across batches and computes per-class AUC.

    Usage:
        meter = AUCMeter()
        for batch in loader:
            preds, labels = model(batch)
            meter.update(preds, labels)
        results = meter.compute()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of predictions and ground truth labels."""
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> dict[str, float]:
        """Compute per-class and mean AUC across all accumulated batches.

        Returns:
            Dict with per-class AUC values and 'mean_auc' key.
            Classes with no positive examples are skipped.
        """
        all_preds = torch.cat(self._preds, dim=0).numpy()
        all_targets = torch.cat(self._targets, dim=0).numpy()

        results: dict[str, float] = {}
        valid_aucs: list[float] = []

        for i, class_name in enumerate(PATHOLOGY_CLASSES):
            y_true = all_targets[:, i]
            y_score = all_preds[:, i]

            if y_true.sum() == 0:
                # Cannot compute AUC with no positive examples
                results[class_name] = float("nan")
                continue

            auc = roc_auc_score(y_true, y_score)
            results[class_name] = auc
            valid_aucs.append(auc)

        results["mean_auc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        return results
