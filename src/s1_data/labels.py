"""Boundary-to-mask conversion for layer segmentation ground truth.

MATLAB-generated datasets (generate_dme_train.m, generate_hc_train.m) store
layer ground truth as raw boundary coordinates (y-pixel row per A-scan column),
not rasterized masks. This fills between consecutive boundaries to produce
per-layer binary masks.
"""

import json
from pathlib import Path

import numpy as np


def boundaries_to_layer_masks(boundaries, height):
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


def load_boundaries_json(label_path):
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


def is_annotated(boundaries):
    """
    True if at least one column has a full, non-NaN boundary set.

    MATLAB-generated datasets always have complete annotations (no NaN),
    so this always returns True. Kept for API compatibility.
    """
    return not np.isnan(boundaries).all()
