"""
Wrapper that runs every metric implemented in
external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/Metrics/
(Region_based_metrics.py, ConfusionMatrix_based_metrics.py,
Contour_based_metrics.py, PixelError_based_metrics.py,
Biomarker_based_metrics.py) against one prediction, a whole method's test
set, or several methods at once.
"""
import csv
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

from Region_based_metrics import dice_coefficient, iou_score, precision, recall
from ConfusionMatrix_based_metrics import accuracy, sensitivity, specificity, auc_score
from Contour_based_metrics import hausdorff_distance, hausdorff_distance_95, assd, mad
from PixelError_based_metrics import mean_squared_error, root_mean_squared_error
from Biomarker_based_metrics import thickness_difference, vascularity_index

# Contour metrics need a contour crossing the 0.5 level in both masks;
# an empty or fully-filled mask raises IndexError inside skimage's
# find_contours, so those are caught and reported as NaN instead of crashing
# a whole-dataset run.
_CONTOUR_METRICS = (
    ("hausdorff_distance", hausdorff_distance),
    ("hausdorff_distance_95", hausdorff_distance_95),
    ("assd", assd),
)


def _binarize(mask):
    # int64 (signed), not uint8: Biomarker_based_metrics.thickness_difference
    # sums columns and subtracts, and numpy keeps unsigned accumulators
    # unsigned, so an unsigned dtype here makes that subtraction wrap around
    # to a huge value instead of going negative.
    mask = np.asarray(mask)
    return (mask > 0).astype(np.int64)


def compute_all_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute every metric in Metrics/ for one ground-truth/prediction mask pair.

    Args:
        y_true: ground-truth mask (H, W). Any nonzero value is treated as foreground.
        y_pred: predicted mask (H, W), same convention as y_true.
        y_pred_proba: optional probability map (H, W) used for AUC only;
            defaults to y_pred if not given.
    Returns:
        dict mapping metric name -> float score. Contour-based metrics are
        NaN if no contour exists in either mask (e.g. empty prediction).
    """
    proba = y_pred_proba if y_pred_proba is not None else y_pred
    y_true = _binarize(y_true)
    y_pred = _binarize(y_pred)

    results = {}

    # Region-based
    results["dice"] = dice_coefficient(y_true, y_pred)
    results["iou"] = iou_score(y_true, y_pred)
    results["precision"] = precision(y_true, y_pred)
    results["recall"] = recall(y_true, y_pred)

    # Confusion-matrix-based
    results["accuracy"] = accuracy(y_true, y_pred)
    results["sensitivity"] = sensitivity(y_true, y_pred)
    results["specificity"] = specificity(y_true, y_pred)
    results["auc"] = auc_score(y_true, proba)

    # Contour-based
    for name, fn in _CONTOUR_METRICS:
        try:
            results[name] = fn(y_true, y_pred)
        except (IndexError, ValueError):
            results[name] = float("nan")
    results["mad"] = mad(y_true, y_pred)

    # Pixel-error-based
    results["mse"] = mean_squared_error(y_true, y_pred)
    results["rmse"] = root_mean_squared_error(y_true, y_pred)

    # Biomarker-based
    results["thickness_difference"] = thickness_difference(y_true, y_pred)
    results["vascularity_index"] = vascularity_index(y_true, y_pred)

    return results


def evaluate_method(y_true_list, y_pred_list, y_pred_proba_list=None):
    """
    Run compute_all_metrics over every sample of one method's test set.

    Args:
        y_true_list: list of ground-truth masks.
        y_pred_list: list of predicted masks, same length/order as y_true_list.
        y_pred_proba_list: optional list of probability maps for AUC.
    Returns:
        (per_sample, summary): per_sample is a list of per-sample metric
        dicts; summary maps metric name -> {"mean": ..., "std": ...} computed
        with NaNs ignored.
    """
    if y_pred_proba_list is None:
        y_pred_proba_list = [None] * len(y_pred_list)

    per_sample = [
        compute_all_metrics(yt, yp, proba)
        for yt, yp, proba in zip(y_true_list, y_pred_list, y_pred_proba_list)
    ]

    summary = {}
    for name in per_sample[0]:
        values = np.array([sample[name] for sample in per_sample], dtype=float)
        summary[name] = {"mean": np.nanmean(values), "std": np.nanstd(values)}

    return per_sample, summary


def compare_methods(methods, output_csv=None):
    """
    Summarize and optionally save a metric comparison table across methods.

    Args:
        methods: dict mapping method_name -> (y_true_list, y_pred_list) or
            (y_true_list, y_pred_list, y_pred_proba_list).
        output_csv: optional path to write the comparison table as CSV.
    Returns:
        dict mapping method_name -> summary dict from evaluate_method.
    """
    summaries = {}
    for method_name, data in methods.items():
        y_true_list, y_pred_list = data[0], data[1]
        y_pred_proba_list = data[2] if len(data) > 2 else None
        _, summary = evaluate_method(y_true_list, y_pred_list, y_pred_proba_list)
        summaries[method_name] = summary

    if output_csv is not None:
        metric_names = list(next(iter(summaries.values())).keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["method"]
                + [f"{m}_mean" for m in metric_names]
                + [f"{m}_std" for m in metric_names]
            )
            for method_name, summary in summaries.items():
                row = [method_name]
                row += [summary[m]["mean"] for m in metric_names]
                row += [summary[m]["std"] for m in metric_names]
                writer.writerow(row)

    return summaries


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = (rng.random((64, 64)) > 0.5).astype(np.uint8)
    y_pred = (rng.random((64, 64)) > 0.5).astype(np.uint8)

    for name, value in compute_all_metrics(y_true, y_pred).items():
        print(f"{name}: {value:.4f}")
