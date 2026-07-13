"""Boundary-to-mask conversion for layer segmentation ground truth."""

import json
from pathlib import Path

import numpy as np


def boundaries_to_layer_masks(boundaries: np.ndarray, height: int) -> np.ndarray:
    """
    Args:
        boundaries: (n_boundaries, width) array of y-row positions: layer k
            spans rows [boundaries[k], boundaries[k+1]) at each column.
            Columns may be NaN where a boundary wasn't annotated.
        height: image height (number of rows) to rasterize into.
    Returns:
        (n_boundaries - 1, height, width) uint8 array, one binary mask per
        layer. A column is left at 0 for a layer if either of its
        boundaries is NaN there.
    """
    n_boundaries, width = boundaries.shape
    masks = np.zeros((n_boundaries - 1, height, width), dtype=np.uint8)
    rows = np.arange(height)[:, None]
    for k in range(n_boundaries - 1):
        top, bottom = boundaries[k], boundaries[k + 1]
        valid = ~np.isnan(top) & ~np.isnan(bottom)
        top_r = np.round(np.nan_to_num(top)).astype(np.int64)
        bottom_r = np.round(np.nan_to_num(bottom)).astype(np.int64)
        col_mask = (rows >= top_r[None, :]) & (rows < bottom_r[None, :])
        masks[k] = col_mask & valid[None, :]
    return masks


def load_boundaries_json(label_path: str | Path) -> np.ndarray:
    """Load boundaries from MATLAB-generated JSON label file.

    Args:
        label_path: path to label/*.txt (JSON file).
    Returns:
        (n_boundaries, width) array of y-row positions. Converts from
        MATLAB 1-indexed to Python 0-indexed by subtracting 1.
    """
    with open(label_path) as f:
        data = json.load(f)
    boundaries = np.array(data["bds"], dtype=np.float64)
    boundaries -= 1
    return boundaries
