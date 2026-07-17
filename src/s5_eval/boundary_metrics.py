"""
Boundary metrics (docs/implementation_plan.md, Setup > Metrics): MAD and
RMSE, computed per boundary, then averaged.

Reuses Contour_based_metrics.mad and
PixelError_based_metrics.root_mean_squared_error. Both expect 1D
per-boundary row-position arrays, not 2D masks — feeding masks silently
turns mad into a pixel-overlap metric instead.

Both also use np.mean, not np.nanmean, so unannotated columns must be
dropped before calling them: DUKE-DME leaves ~31% of columns unlabeled and
a single NaN would take the whole score to NaN.
"""
import os
import sys
from collections.abc import Sequence

import numpy as np

from src.s1_data.labels import HARMONIZED_BOUNDARY_NAMES

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

from Contour_based_metrics import mad  # type: ignore # noqa: E402
from PixelError_based_metrics import root_mean_squared_error  # type: ignore # noqa: E402


def boundary_metrics(y_true_boundaries: Sequence[np.ndarray], y_pred_boundaries: Sequence[np.ndarray], names: Sequence[str] | None = None) -> dict[str, float]:
    """
    Compute MAD and RMSE per boundary (row position per A-scan column),
    scored only over columns the graders annotated, then average across
    boundaries.

    NaN in y_true marks an unannotated column and is skipped. NaN in y_pred
    is a method failure and is left to propagate, so that boundary scores NaN
    rather than being silently excluded.

    Args:
        y_true_boundaries: sequence of per-boundary ground-truth row
            positions, one 1D array per boundary — not a binary mask.
        y_pred_boundaries: predicted per-boundary row positions, same
            boundary count and order as y_true_boundaries.
        names: per-boundary labels for the breakdown keys. Defaults to
            HARMONIZED_BOUNDARY_NAMES when the count matches, else indices.
    Returns:
        dict with "mad" and "rmse" (means over boundaries, NaN boundaries
        skipped), plus "mad_<name>"/"rmse_<name>" per boundary. A boundary
        with no annotated column scores NaN.
    """
    if names is None:
        names = (
            HARMONIZED_BOUNDARY_NAMES
            if len(y_true_boundaries) == len(HARMONIZED_BOUNDARY_NAMES)
            else [str(i) for i in range(len(y_true_boundaries))]
        )

    mad_scores = []
    rmse_scores = []
    for y_true, y_pred in zip(y_true_boundaries, y_pred_boundaries):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        annotated = ~np.isnan(y_true)
        if not annotated.any():
            mad_scores.append(np.nan)
            rmse_scores.append(np.nan)
            continue
        mad_scores.append(mad(y_true[annotated], y_pred[annotated]))
        rmse_scores.append(root_mean_squared_error(y_true[annotated], y_pred[annotated]))

    results = {
        "mad": float(np.nanmean(mad_scores)),
        "rmse": float(np.nanmean(rmse_scores)),
    }
    for name, mad_score, rmse_score in zip(names, mad_scores, rmse_scores):
        results[f"mad_{name}"] = float(mad_score)
        results[f"rmse_{name}"] = float(rmse_score)
    return results
