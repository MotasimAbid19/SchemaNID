"""Prediction and binary evaluation metrics for SchemaNID."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .model import SchemaNID


def predict_probabilities(
    model: SchemaNID,
    values: np.ndarray,
    present: np.ndarray,
    schema_ids: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("At least one row is required for prediction")
    target = torch.device(device)
    model = model.to(target).eval()
    dataset = TensorDataset(
        torch.from_numpy(values).float(),
        torch.from_numpy(present).float(),
        torch.from_numpy(schema_ids).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch_values, batch_present, batch_schemas in loader:
            logits = model(
                batch_values.to(target),
                batch_present.to(target),
                batch_schemas.to(target),
            )["logits"]
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Compute the main metrics reported in the paper."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be equally sized 1-D arrays")
    if len(labels) == 0:
        raise ValueError("At least one labeled row is required")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")

    predictions = (probabilities >= threshold).astype(np.int64)
    two_classes = len(np.unique(labels)) == 2
    return {
        "rows": int(len(labels)),
        "threshold": float(threshold),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "attack_f1": float(
            f1_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if two_classes
            else float("nan")
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)) if two_classes else float("nan"),
        "auroc": float(roc_auc_score(labels, probabilities)) if two_classes else float("nan"),
        "auprc": (
            float(average_precision_score(labels, probabilities))
            if two_classes
            else float("nan")
        ),
        "brier": float(brier_score_loss(labels, probabilities)),
    }
