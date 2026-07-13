"""
Intensity normalization (implementation_plan.md, Setup > Preprocessing).
"""

import numpy as np


def normalize(bscan: np.ndarray, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    """Normalize a single B-scan via percentile clipping and min-max scaling.

    bscan: grayscale image as a uint8 or float array (any range).
    lo_pct, hi_pct: percentiles used as the clip bounds before scaling, to
        limit the influence of outlier pixels (e.g. specular reflections).
    Returns a float32 array of the same shape, scaled to [0, 1].
    """
    img_float = bscan.astype(np.float32)

    lo, hi = np.percentile(img_float, [lo_pct, hi_pct])
    clipped = np.clip(img_float, lo, hi)
    normalizedScan = (clipped - lo) / (hi - lo)

    return normalizedScan
