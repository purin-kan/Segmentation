"""
Region-based metrics (docs/implementation_plan.md, Setup > Metrics): Dice
and IoU, computed per layer as a separate binary mask, then averaged
across layers.

Reuses external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/Metrics/
Region_based_metrics.py — dice_coefficient/iou_score are generic
set-overlap formulas, so they apply unchanged to a single layer's binary
mask.
"""
import os
import sys

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

from Region_based_metrics import dice_coefficient, iou_score  # noqa: E402


def region_metrics(y_true_layers, y_pred_layers):
    """
    Compute Dice and IoU per layer, then average across layers.

    Args:
        y_true_layers: sequence of per-layer ground-truth binary masks,
            one (H, W) array per layer — number of layers is set by
            "Label definition" (implementation_plan.md, To Be Decided).
        y_pred_layers: predicted per-layer binary masks, same layer count
            and order as y_true_layers.
    Returns:
        dict with "dice" and "iou", each the mean over layers.
    """
    raise NotImplementedError
