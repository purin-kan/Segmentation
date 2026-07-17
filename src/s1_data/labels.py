"""Boundary-to-mask conversion for layer segmentation ground truth."""

import json
from pathlib import Path

import numpy as np

# The 6-boundary/5-layer harmonized scheme (implementation_plan.md,
# Setup > label harmonization). Keyed by each dataset's native boundary
# count, which identifies it unambiguously: DUKE-DME ships 8, HC-MS 9.
HARMONIZED_INDICES = {
    8: [0, 1, 2, 3, 4, 7],  # DUKE-DME: drops ONL-ISM/ISE, ISE/OS-RPE
    9: [0, 1, 2, 3, 4, 8],  # HC-MS: drops ELM, IS-OS, OS-RPE
}

HARMONIZED_BOUNDARY_NAMES = ("ILM", "RNFL/GCL", "IPL/INL", "INL/OPL", "OPL/ONL", "BM")
HARMONIZED_LAYER_NAMES = ("RNFL", "GCL+IPL", "INL", "OPL", "ONL-BM")

N_HARMONIZED_BOUNDARIES = len(HARMONIZED_BOUNDARY_NAMES)


def harmonize_boundaries(boundaries: np.ndarray) -> np.ndarray:
    """Reduce a dataset's native boundaries to the shared 6.

    Args:
        boundaries: (n_boundaries, width) array, 8 rows for DUKE-DME or 9 for
            HC-MS. Already-harmonized (6-row) input is returned unchanged.
    Returns:
        (6, width) array ordered as HARMONIZED_BOUNDARY_NAMES.
    Raises:
        ValueError: on any other boundary count.
    """
    n_boundaries = boundaries.shape[0]
    if n_boundaries == N_HARMONIZED_BOUNDARIES:
        return boundaries
    if n_boundaries not in HARMONIZED_INDICES:
        raise ValueError(
            f"Expected {sorted(HARMONIZED_INDICES)} or {N_HARMONIZED_BOUNDARIES} "
            f"boundaries, got {n_boundaries}."
        )
    return boundaries[HARMONIZED_INDICES[n_boundaries]]


def annotated_columns_per_layer(boundaries: np.ndarray) -> np.ndarray:
    """Columns where a layer has ground truth, i.e. both bounding boundaries
    are annotated. Same layer order as boundaries_to_layer_masks.

    Validity is per layer, not global: DUKE-DME's OPL/ONL is annotated over
    fewer columns than its other boundaries, so the layers it bounds have a
    smaller valid set.

    Args:
        boundaries: (n_boundaries, width) array of y-row positions, NaN where
            unannotated.
    Returns:
        (n_boundaries - 1, width) bool array.
    """
    annotated = ~np.isnan(boundaries)
    return annotated[:-1] & annotated[1:]


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
