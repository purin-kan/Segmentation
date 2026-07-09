"""
Eval harness for the metrics chosen in docs/implementation_plan.md
(Setup > Metrics): Dice/IoU per layer (region_metrics.py) and MAD/RMSE per
boundary (boundary_metrics.py). Aggregates per-sample scores into
per-method summaries, optionally as CSV.
"""
import csv

import numpy as np

from src.s5_eval.region_metrics import region_metrics
from src.s5_eval.boundary_metrics import boundary_metrics


def compute_metrics(y_true_layers, y_pred_layers, y_true_boundaries, y_pred_boundaries):
    """
    Compute the four chosen metrics for one sample.

    Args:
        y_true_layers, y_pred_layers: per-layer binary masks — see
            region_metrics.region_metrics.
        y_true_boundaries, y_pred_boundaries: per-boundary row positions —
            see boundary_metrics.boundary_metrics.
    Returns:
        dict with "dice", "iou", "mad", "rmse".
    """
    results = {}
    results.update(region_metrics(y_true_layers, y_pred_layers))
    results.update(boundary_metrics(y_true_boundaries, y_pred_boundaries))
    return results


def evaluate_method(samples):
    """
    Run compute_metrics over every sample of one method's test set.

    Args:
        samples: iterable of (y_true_layers, y_pred_layers,
            y_true_boundaries, y_pred_boundaries) tuples, one per B-scan.
    Returns:
        (per_sample, summary): per_sample is a list of per-sample metric
        dicts; summary maps metric name -> {"mean": ..., "std": ...} computed
        with NaNs ignored.
    """
    per_sample = [compute_metrics(*sample) for sample in samples]

    summary = {}
    for name in per_sample[0]:
        values = np.array([sample[name] for sample in per_sample], dtype=float)
        summary[name] = {"mean": np.nanmean(values), "std": np.nanstd(values)}

    return per_sample, summary


def compare_methods(methods, output_csv=None):
    """
    Summarize and optionally save a metric comparison table across methods.

    Args:
        methods: dict mapping method_name -> samples (see evaluate_method).
        output_csv: optional path to write the comparison table as CSV.
    Returns:
        dict mapping method_name -> summary dict from evaluate_method.
    """
    summaries = {}
    for method_name, samples in methods.items():
        _, summary = evaluate_method(samples)
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
