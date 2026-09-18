"""Readable training procedure matching the SchemaNID paper settings."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .model import SchemaNID


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 101
    batch_size: int = 256
    epochs: int = 25
    patience: int = 5
    learning_rate: float = 6e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use device='cpu' or 'auto'")
    return torch.device(requested)


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Select the threshold on validation data only."""

    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(np.int64)
        score = float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
        if score > best_score + 1e-12:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score


def _loader(
    values: np.ndarray,
    present: np.ndarray,
    labels: np.ndarray,
    schema_ids: np.ndarray,
    batch_size: int,
    *,
    balanced: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(values).float(),
        torch.from_numpy(present).float(),
        torch.from_numpy(labels).float(),
        torch.from_numpy(schema_ids).long(),
    )
    sampler = None
    if balanced:
        counts = np.bincount(labels.astype(np.int64), minlength=2)
        weights = 1.0 / np.maximum(counts[labels.astype(np.int64)], 1)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)


def _probabilities(
    model: SchemaNID,
    values: np.ndarray,
    present: np.ndarray,
    labels: np.ndarray,
    schema_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = _loader(
        values,
        present,
        labels,
        schema_ids,
        batch_size,
        balanced=False,
        seed=0,
    )
    with torch.no_grad():
        for x, mask, _y, domains in loader:
            logits = model(x.to(device), mask.to(device), domains.to(device))["logits"]
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def train(
    model: SchemaNID,
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, object]:
    """Initialize and train using the paper's seeded selection procedure."""

    set_seed(config.seed)
    model.reset_parameters()
    device = choose_device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    train_loader = _loader(
        *train_data,
        config.batch_size,
        balanced=True,
        seed=config.seed,
    )

    best_state = None
    best_score = -1.0
    best_threshold = 0.5
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for values, present, labels, schema_ids in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                values.to(device), present.to(device), schema_ids.to(device)
            )["logits"]
            loss = criterion(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_probability = _probabilities(
            model, *validation_data, config.batch_size, device
        )
        threshold, score = select_threshold(
            validation_data[2], validation_probability
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_macro_f1": score,
                "threshold": threshold,
            }
        )
        if score > best_score + 1e-5:
            best_state = copy.deepcopy(model.state_dict())
            best_score = score
            best_threshold = threshold
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "threshold": best_threshold,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "history": history,
    }
