"""Inference adapter turning a trained DL model into a segment_fn
(evaluation_protocol.md, Boundary adapter).

run_experiment calls segment_fn(bscan) and expects native-size
(layer_masks, boundaries), the same contract the classical methods meet. This
wraps the canvas round trip and the class-map-to-boundary reduction so the
eval harness cannot tell a padded network from a per-column threshold.

Boundary positions are the canonical representation: layer masks are derived
from the ordered boundaries rather than from the raw argmax, so the region and
boundary metrics always describe the same segmentation.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from src.s1_data.labels import N_HARMONIZED_BOUNDARIES, boundaries_to_layer_masks
from src.s3_methods.m5_deep_learning.canvas import crop_from_canvas, pad_to_canvas
from src.s4_postprocessing.ordering import enforce_non_crossing


def class_map_to_boundaries(class_map: np.ndarray, n_boundaries: int = N_HARMONIZED_BOUNDARIES) -> np.ndarray:
    """Reduce a per-pixel class map to per-boundary row positions.

    Class k + 1 is the k-th layer (0 is background), so boundary k is the
    first row of layer k + 1 and the last boundary is the bottom of the final
    layer. A column where a layer is absent yields NaN for its boundary: the
    model declined there, which coverage records rather than the distance
    metrics.

    Args:
        class_map: (height, width) int array of predicted class indices.
        n_boundaries: harmonized boundary count.
    Returns:
        (n_boundaries, width) float array of y-row positions, NaN where the
        bounding layer is absent in that column.
    """
    height, width = class_map.shape
    boundaries = np.full((n_boundaries, width), np.nan)

    for layer in range(n_boundaries - 1):
        mask = class_map == layer + 1
        present = mask.any(axis=0)
        boundaries[layer, present] = np.argmax(mask, axis=0)[present]

    # The outermost boundary is the bottom edge of the last layer, not the top
    # of a further one.
    last = class_map == n_boundaries - 1
    present = last.any(axis=0)
    bottom = (height - 1) - np.argmax(last[::-1], axis=0)
    boundaries[n_boundaries - 1, present] = bottom[present] + 1

    return boundaries


def make_segment_fn(model: nn.Module, device: str | torch.device | None = None) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Wrap a trained model as the segment_fn run_experiment expects.

    Args:
        model: network mapping (B, 1, 224, 1024) to (B, n_classes, 224, 1024).
        device: torch device; defaults to CUDA when available.
    Returns:
        segment_fn(bscan) -> (layer_masks, boundaries) at the B-scan's native
        size, with boundaries ordered non-crossing.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    def segment_fn(bscan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        native_shape = bscan.shape

        canvas = pad_to_canvas(np.asarray(bscan, dtype=np.float32), value=0.0)
        batch = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(batch)

        # Crop before reducing to boundaries: tracing across the padded columns
        # would emit positions for columns the B-scan does not have.
        logits = crop_from_canvas(logits, native_shape)
        class_map = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        boundaries = enforce_non_crossing(class_map_to_boundaries(class_map))
        layer_masks = boundaries_to_layer_masks(boundaries, height=native_shape[0])
        return layer_masks, boundaries

    return segment_fn
