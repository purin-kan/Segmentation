"""
Boundary metrics (docs/implementation_plan.md, Setup > Metrics): MAD and
RMSE, computed per boundary, then averaged.

Reuses Contour_based_metrics.mad and
PixelError_based_metrics.root_mean_squared_error. Both expect 1D
per-boundary row-position arrays, not 2D masks — feeding masks silently
turns mad into a pixel-overlap metric instead (temp/AUDIT.md, B2).
"""
import os
import sys
from collections.abc import Sequence

import numpy as np

_METRICS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "external",
        "Retinal_OCT_Image_Segmentation_via_Deep_Learning",
        "Metrics",
    )
)
sys.path.insert(0, _METRICS_DIR)

from Contour_based_metrics import mad  # noqa: E402
from PixelError_based_metrics import root_mean_squared_error  # noqa: E402


def boundary_metrics(y_true_boundaries: Sequence[np.ndarray], y_pred_boundaries: Sequence[np.ndarray]) -> dict[str, float]:
    """
    Compute MAD and RMSE per boundary (row position per A-scan column),
    then average across boundaries.

    Args:
        y_true_boundaries: sequence of per-boundary ground-truth row
            positions, one 1D array per boundary — not a binary mask.
        y_pred_boundaries: predicted per-boundary row positions, same
            boundary count and order as y_true_boundaries.
    Returns:
        dict with "mad" and "rmse", each the mean over boundaries.
    """
    mad_scores = []
    rmse_scores = []
    for y_true, y_pred in zip(y_true_boundaries, y_pred_boundaries):
        mad_scores.append(mad(y_true, y_pred))
        rmse_scores.append(root_mean_squared_error(y_true, y_pred))

    return {
        "mad": float(np.mean(mad_scores)),
        "rmse": float(np.mean(rmse_scores)),
    }
