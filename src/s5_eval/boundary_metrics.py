"""
Boundary metrics (docs/implementation_plan.md, Setup > Metrics): MAD and
RMSE, computed per boundary, then averaged, plus per-boundary coverage.

Reuses Contour_based_metrics.mad and
PixelError_based_metrics.root_mean_squared_error. Both expect 1D
per-boundary row-position arrays, not 2D masks — feeding masks silently
turns mad into a pixel-overlap metric instead.

Two separate NaN sources, handled differently:
  - NaN in y_true is an unannotated column (DUKE-DME leaves ~31% unlabeled).
    Dropped from scoring, never charged against a method.
  - NaN in y_pred is a column the method declined to segment. MAD/RMSE are
    measured only over the columns actually predicted, so a gap no longer
    NaN-poisons the whole boundary; the unpredicted fraction is reported
    separately as coverage rather than folded into the distance.
Both metric functions use np.mean, so the masked slices passed to them must
be NaN-free.
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
    Compute MAD, RMSE, and coverage per boundary (row position per A-scan
    column), then average across boundaries.

    MAD/RMSE are scored over the columns that are both annotated in y_true
    and predicted in y_pred, i.e. the method's accuracy where it committed.
    Coverage is the fraction of annotated columns the method actually
    predicted, reporting failures-to-predict as their own number rather than
    folding them into the distance.

    Two empty cases are kept distinct: a boundary with no annotated column at
    all scores NaN on every metric (undefined); a boundary that is annotated
    but which the method predicted nowhere scores coverage 0.0 (it covered
    nothing, a real measurement) with MAD/RMSE NaN (no column to measure).

    Args:
        y_true_boundaries: sequence of per-boundary ground-truth row
            positions, one 1D array per boundary — not a binary mask.
        y_pred_boundaries: predicted per-boundary row positions, same
            boundary count and order as y_true_boundaries. NaN marks a column
            the method declined to segment.
        names: per-boundary labels for the breakdown keys. Defaults to
            HARMONIZED_BOUNDARY_NAMES when the count matches, else indices.
    Returns:
        dict with "mad", "rmse", "coverage" (means over boundaries, NaN
        boundaries skipped), plus "mad_<name>"/"rmse_<name>"/"coverage_<name>"
        per boundary.
    """
    if names is None:
        names = (
            HARMONIZED_BOUNDARY_NAMES
            if len(y_true_boundaries) == len(HARMONIZED_BOUNDARY_NAMES)
            else [str(i) for i in range(len(y_true_boundaries))]
        )

    mad_scores = []
    rmse_scores = []
    coverage_scores = []
    for y_true, y_pred in zip(y_true_boundaries, y_pred_boundaries):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        annotated = ~np.isnan(y_true)
        if not annotated.any():
            mad_scores.append(np.nan)
            rmse_scores.append(np.nan)
            coverage_scores.append(np.nan)
            continue
        predicted = annotated & ~np.isnan(y_pred)
        coverage_scores.append(float(predicted.sum() / annotated.sum()))
        if not predicted.any():
            mad_scores.append(np.nan)
            rmse_scores.append(np.nan)
            continue
        mad_scores.append(mad(y_true[predicted], y_pred[predicted]))
        rmse_scores.append(root_mean_squared_error(y_true[predicted], y_pred[predicted]))

    results = {
        "mad": float(np.nanmean(mad_scores)),
        "rmse": float(np.nanmean(rmse_scores)),
        "coverage": float(np.nanmean(coverage_scores)),
    }
    for name, mad_score, rmse_score, coverage_score in zip(names, mad_scores, rmse_scores, coverage_scores):
        results[f"mad_{name}"] = float(mad_score)
        results[f"rmse_{name}"] = float(rmse_score)
        results[f"coverage_{name}"] = float(coverage_score)
    return results
