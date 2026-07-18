"""
Boundary ordering / non-crossing enforcement for deep-learning outputs.

implementation_plan.md, Setup > Postprocessing: DL methods need this step to
be compared fairly against Graph Search/DP/Graph-Cut, which enforce
non-crossing boundaries inherently. A fairness patch, not a method.

Ordering only. No smoothing and no gap interpolation: interpolating would
manufacture coverage the model did not earn.
"""
import numpy as np


def enforce_non_crossing(boundaries: np.ndarray) -> np.ndarray:
    """Enforce y[k] >= y[k-1] per column by a running max down the boundary axis.

    Cummax rather than a per-column sort: sort assumes only the boundary-index
    assignment was permuted, so one misplaced boundary shifts the others in
    that column. Cummax treats the ordering as authoritative and pins the
    offender against the boundary above it, keeping the damage local.

    NaN marks a column the method declined, and coverage counts non-NaN
    columns, so NaNs must survive. np.maximum.accumulate would propagate them
    downward and destroy earned coverage; the running max here skips them and
    carries across the gap instead.

    Args:
        boundaries: (n_boundaries, width) array of y-row positions, ordered
            inner to outer, NaN where unpredicted.
    Returns:
        (n_boundaries, width) array, same NaN pattern. Columns that were
        already ordered come back unchanged.
    Raises:
        ValueError: if the input is not 2D.
    """
    ordered = np.array(boundaries, dtype=float, copy=True)
    if ordered.ndim != 2:
        raise ValueError(f"Expected (n_boundaries, width), got shape {ordered.shape}.")

    running_max = np.full(ordered.shape[1], -np.inf)
    for k in range(ordered.shape[0]):
        predicted = ~np.isnan(ordered[k])
        ordered[k, predicted] = np.maximum(ordered[k, predicted], running_max[predicted])
        running_max[predicted] = ordered[k, predicted]
    return ordered
