"""Shared training loop for the DL methods (implementation_plan.md, Setup >
Training protocol).

Wraps native-size OCTDataset samples onto the fixed canvas (canvas.py) and
rasterizes harmonized boundaries into a per-pixel class target: 0 for
background, 1-5 for HARMONIZED_LAYER_NAMES in order.

Two kinds of pixel carry no usable label and are both set to IGNORE_INDEX so
CrossEntropyLoss skips them:
  - canvas padding, which is 43% of the canvas for an HC-MS scan;
  - columns where any of the 6 boundaries is unannotated. DUKE-DME traces a
    fovea window and has a complete set on 64.6% of columns; HC-MS on 100%.

The optimizer, LR, augmentation, early-stopping rule and seed handling live
here rather than per method, so every method trains under the same protocol.
Only the model and its loss vary, per the shared/per-method table.

5e regresses boundary positions rather than pixel classes, so it needs a
(n_boundaries, width) target this module does not yet build.
"""

import copy
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.s1_data.dataset import OCTDataset
from src.s1_data.labels import N_HARMONIZED_BOUNDARIES, boundaries_to_layer_masks
from src.s3_methods.m5_deep_learning.canvas import pad_to_canvas

# torch.nn.CrossEntropyLoss's own default, so a caller who builds the loss
# without arguments still skips these pixels.
IGNORE_INDEX = -100

# 5 harmonized layers plus background (above ILM and below BM).
N_CLASSES = N_HARMONIZED_BOUNDARIES

# Augmentation set fixed by implementation_plan.md: horizontal flip, small
# horizontal translation, mild intensity/contrast jitter. No vertical flip or
# rotation, both of which destroy boundary ordering.
MAX_SHIFT_COLUMNS = 16
JITTER = 0.1


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def boundaries_to_class_map(boundaries: np.ndarray, height: int) -> np.ndarray:
    """Rasterize boundaries into a per-pixel class target.

    Args:
        boundaries: (6, width) harmonized boundary row positions, NaN where
            unannotated.
        height: image height to rasterize into.
    Returns:
        (height, width) int64 array. 0 is background, k + 1 is the k-th
        harmonized layer, IGNORE_INDEX marks columns without a complete
        boundary set.
    """
    masks = boundaries_to_layer_masks(boundaries, height=height)
    target = np.zeros((height, boundaries.shape[1]), dtype=np.int64)
    for layer, mask in enumerate(masks):
        target[mask.astype(bool)] = layer + 1

    annotated = ~np.isnan(boundaries).any(axis=0)
    target[:, ~annotated] = IGNORE_INDEX
    return target


def _shift_columns(array: np.ndarray, shift: int, fill: float) -> np.ndarray:
    """Translate an array horizontally, filling the vacated columns."""
    out = np.full_like(array, fill)
    if shift > 0:
        out[:, shift:] = array[:, :-shift]
    elif shift < 0:
        out[:, :shift] = array[:, -shift:]
    else:
        out[:] = array
    return out


def augment(image: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply the shared augmentation set to a native-size image and target.

    Runs before padding: flipping or shifting the canvas instead would move
    the padding off the bottom-right and break the coordinate invariant
    canvas.py relies on.

    Args:
        image: (height, width) float32 in [0, 1].
        target: (height, width) int64 class map.
        rng: seeded generator, so a run is reproducible.
    Returns:
        (image, target) augmented together, same shapes.
    """
    if rng.random() < 0.5:
        image = image[:, ::-1]
        target = target[:, ::-1]

    shift = int(rng.integers(-MAX_SHIFT_COLUMNS, MAX_SHIFT_COLUMNS + 1))
    if shift:
        image = _shift_columns(image, shift, 0.0)
        target = _shift_columns(target, shift, IGNORE_INDEX)

    scale = 1.0 + rng.uniform(-JITTER, JITTER)
    offset = rng.uniform(-JITTER, JITTER)
    image = np.clip(image * scale + offset, 0.0, 1.0, dtype=np.float32)

    return np.ascontiguousarray(image), np.ascontiguousarray(target)


class CanvasDataset(Dataset):
    """torch Dataset over one or more OCTDatasets, padded onto the canvas.

    Takes several OCTDatasets so DUKE-DME and HC-MS can train as one set
    while each stays a separate directory on disk.
    """

    def __init__(self, datasets: Sequence[OCTDataset], augment_samples: bool = False, seed: int = 0) -> None:
        """
        Args:
            datasets: OCTDatasets already filtered to a fold's patient ids.
            augment_samples: apply the shared augmentation set. Training
                folds only, never validation.
            seed: seeds this dataset's augmentation RNG.
        """
        self.datasets = list(datasets)
        self.augment_samples = augment_samples
        self.rng = np.random.default_rng(seed)
        self.index = [
            (dataset_idx, sample_idx)
            for dataset_idx, dataset in enumerate(self.datasets)
            for sample_idx in range(len(dataset))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (image, target): image is (1, 224, 1024) float32, target is
            (224, 1024) int64.
        """
        dataset_idx, sample_idx = self.index[idx]
        image, _, boundaries = self.datasets[dataset_idx][sample_idx]
        target = boundaries_to_class_map(boundaries, height=image.shape[0])

        if self.augment_samples:
            image, target = augment(image, target, self.rng)

        image = pad_to_canvas(image, value=0.0)
        target = pad_to_canvas(target, value=IGNORE_INDEX)
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(target)


def _run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> float:
    """Run one pass over a loader. Trains when optimizer is given, else evaluates.

    Returns:
        Mean loss per sample over the pass.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.shape[0]
            total_samples += images.shape[0]

    return total_loss / total_samples


def train(model: nn.Module, train_dataset: Dataset, val_dataset: Dataset, max_epochs: int = 100, batch_size: int = 4, lr: float = 1e-3, criterion: nn.Module | None = None, patience: int = 10, seed: int | None = None, device: str | torch.device | None = None, checkpoint_path: str | Path | None = None) -> list[dict[str, float]]:
    """Train a segmentation model, keeping the best-validation-loss weights.

    The model must map (B, 1, 224, 1024) to (B, N_CLASSES, 224, 1024).

    Args:
        model: the network to train, modified in place.
        train_dataset, val_dataset: CanvasDatasets over a fold's train and
            val patients. Only the training one should augment.
        max_epochs: ceiling on epochs; early stopping usually ends the run
            first.
        batch_size: samples per step.
        lr: Adam learning rate.
        criterion: loss. Defaults to CrossEntropyLoss ignoring IGNORE_INDEX.
            5e overrides it, per the shared/per-method table.
        patience: stop after this many epochs without a validation
            improvement.
        seed: seeds python/numpy/torch when given.
        device: torch device; defaults to CUDA when available.
        checkpoint_path: if given, the best weights are written here.
    Returns:
        list of {"epoch", "train_loss", "val_loss"}, one per epoch run. On
        return, model holds the best-validation-loss weights, not the last.
    """
    if seed is not None:
        set_seed(seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return history
