"""
Region metrics (docs/implementation_plan.md, Setup > Metrics): Dice and
IoU, computed per layer, then averaged.

Reuses dice_coefficient/iou_score from
external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/Metrics/Region_based_metrics.py.

Scored only over columns with ground truth. DUKE-DME leaves ~31% of columns
unannotated, and a method predicts across the full width regardless, so
scoring those columns charges it false positives for pixels it was never
given labels for: a perfect prediction scores Dice 0.83 instead of 1.0.
"""
import os
import sys
from collections.abc import Sequence

import numpy as np

from src.s1_data.labels import HARMONIZED_LAYER_NAMES

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

from Region_based_metrics import dice_coefficient, iou_score  # type: ignore # noqa: E402


def region_metrics(y_true_layers: Sequence[np.ndarray], y_pred_layers: Sequence[np.ndarray], valid_columns: Sequence[np.ndarray] | None = None, names: Sequence[str] | None = None) -> dict[str, float]:
    """
    Compute Dice and IoU per layer, scored only over columns with ground
    truth, then average across layers.

    Args:
        y_true_layers: sequence of per-layer ground-truth binary masks,
            one (H, W) array per layer. The harmonized scheme gives 5.
        y_pred_layers: predicted per-layer binary masks, same layer count
            and order as y_true_layers.
        valid_columns: optional per-layer 1D bool arrays marking columns with
            ground truth, from labels.annotated_columns_per_layer. Columns
            outside are dropped from both masks. None scores every column,
            which deflates any dataset with unannotated columns.
        names: per-layer labels for the breakdown keys. Defaults to
            HARMONIZED_LAYER_NAMES when the count matches, else indices.
    Returns:
        dict with "dice" and "iou" (means over layers, NaN layers skipped),
        plus "dice_<name>"/"iou_<name>" per layer. A layer with no annotated
        column scores NaN.
    """
    if names is None:
        names = (
            HARMONIZED_LAYER_NAMES
            if len(y_true_layers) == len(HARMONIZED_LAYER_NAMES)
            else [str(i) for i in range(len(y_true_layers))]
        )

    dice_scores = []
    iou_scores = []
    for layer, (y_true, y_pred) in enumerate(zip(y_true_layers, y_pred_layers)):
        if valid_columns is not None:
            annotated = np.asarray(valid_columns[layer], dtype=bool)
            if not annotated.any():
                dice_scores.append(np.nan)
                iou_scores.append(np.nan)
                continue
            y_true = y_true[:, annotated]
            y_pred = y_pred[:, annotated]
        dice_scores.append(dice_coefficient(y_true, y_pred))
        iou_scores.append(iou_score(y_true, y_pred))

    results = {
        "dice": float(np.nanmean(dice_scores)),
        "iou": float(np.nanmean(iou_scores)),
    }
    for name, dice_score, iou_score_ in zip(names, dice_scores, iou_scores):
        results[f"dice_{name}"] = float(dice_score)
        results[f"iou_{name}"] = float(iou_score_)
    return results
