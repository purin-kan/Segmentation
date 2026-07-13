"""
Boundary-based metrics (docs/implementation_plan.md, Setup > Metrics): MAD
and RMSE, computed per boundary, then averaged across boundaries.

Reuses:
  - Contour_based_metrics.mad — mean(|y_true - y_pred|). This is a generic
    elementwise formula: it is a boundary-localization error ONLY when
    given 1D per-boundary row-position arrays (e.g. Chiu manualLayers1/2,
    shape (num_boundaries, W, Z)) — feeding it 2D binarized masks instead
    silently turns it back into a pixel-overlap metric (temp/AUDIT.md, B2).
  - PixelError_based_metrics.root_mean_squared_error — same row-position
    convention.
"""
import os
import sys

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


def boundary_metrics(y_true_boundaries, y_pred_boundaries):
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
