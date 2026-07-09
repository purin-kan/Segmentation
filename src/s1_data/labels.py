"""Boundary-to-mask conversion for layer segmentation ground truth.

BOE.py saves layer ground truth as raw boundary coordinates (y-pixel row
per A-scan column), not rasterized masks — see temp/BOE_usage.md. This
fills between consecutive boundaries to produce per-layer binary masks.
"""

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


def is_annotated(boundaries):
    """
    True if at least one column has a full, non-NaN boundary set.

    Needed because BOE.py writes a layers/*.npy for every B-scan, but Chiu
    DME only hand-annotated 11 of 61 B-scans per subject — the other 50
    are entirely NaN. Filtering on this is what takes OCTDataset from 610
    slices down to the "110 annotated B-scans total" in
    implementation_plan.md.
    """
    return not np.isnan(boundaries).all()
