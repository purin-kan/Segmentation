"""
Eval harness for the metrics chosen in docs/implementation_plan.md
(Setup > Metrics): Dice/IoU per layer (region_metrics.py) and MAD/RMSE per
boundary (boundary_metrics.py). Aggregates per-sample scores into
per-method summaries, optionally as CSV.
"""
import csv
from collections.abc import Sequence, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from src.s1_data.labels import annotated_columns_per_layer
from src.s5_eval.region_metrics import region_metrics
from src.s5_eval.boundary_metrics import boundary_metrics


def compute_metrics(y_true_layers: Sequence[np.ndarray], y_pred_layers: Sequence[np.ndarray], y_true_boundaries: Sequence[np.ndarray], y_pred_boundaries: Sequence[np.ndarray]) -> dict[str, float]:
    """
    Compute the four chosen metrics for one sample, scored only where the
    ground truth is annotated (DUKE-DME leaves ~31% of columns unlabeled).

    Args:
        y_true_layers, y_pred_layers: per-layer binary masks — see
            region_metrics.region_metrics.
        y_true_boundaries, y_pred_boundaries: per-boundary row positions —
            see boundary_metrics.boundary_metrics.
    Returns:
        dict with "dice", "iou", "mad", "rmse" (means), plus a per-layer or
        per-boundary breakdown of each.
    """
    valid_columns = annotated_columns_per_layer(np.asarray(y_true_boundaries, dtype=float))

    results = {}
    results.update(region_metrics(y_true_layers, y_pred_layers, valid_columns=valid_columns))
    results.update(boundary_metrics(y_true_boundaries, y_pred_boundaries))
    return results


def evaluate_method(samples: Iterable[tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]]) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
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
    if not per_sample:
        raise ValueError("No samples to evaluate: the dataset/split yielded zero B-scans.")

    summary = {}
    for name in per_sample[0]:
        values = np.array([sample[name] for sample in per_sample], dtype=float)
        # A boundary every method declines is all-NaN here, which is expected
        # for 1a/2/4 rather than exceptional, so report NaN instead of letting
        # nanmean/nanstd warn on every run.
        if np.isnan(values).all():
            summary[name] = {"mean": float("nan"), "std": float("nan")}
        else:
            summary[name] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}

    return per_sample, summary


def aggregate_per_patient(subject_ids: Sequence[str], per_sample: Sequence[dict[str, float]]) -> list[dict[str, Any]]:
    """Average per-B-scan metrics within each patient.

    The comparison unit is the patient, not the B-scan (implementation_plan.md,
    Evaluation > Comparison): B-scans within a patient are correlated slices of
    the same eye, and each patient is held out in exactly one fold, so one row
    per patient per method double-counts nothing.

    Patients contribute equally regardless of B-scan count (Duke 11, HC-MS 49),
    which is deliberate: it keeps HC-MS volume from dominating the mean.

    Args:
        subject_ids: patient id per entry of per_sample, same order.
        per_sample: per-B-scan metric dicts, as returned by evaluate_method.
    Returns:
        list of {"subject_id", "n_bscans", <metric>: mean, ...}, one row per
        patient, ordered by subject_id. A metric that is NaN on every B-scan of
        a patient stays NaN rather than raising.
    """
    if len(subject_ids) != len(per_sample):
        raise ValueError(
            f"subject_ids and per_sample must line up: got {len(subject_ids)} ids "
            f"for {len(per_sample)} scored B-scans."
        )
    if not per_sample:
        return []

    by_patient: dict[str, list[dict[str, float]]] = {}
    for subject_id, sample in zip(subject_ids, per_sample):
        by_patient.setdefault(subject_id, []).append(sample)

    metric_names = list(per_sample[0])
    rows = []
    for subject_id in sorted(by_patient):
        patient_samples = by_patient[subject_id]
        row: dict[str, Any] = {"subject_id": subject_id, "n_bscans": len(patient_samples)}
        for name in metric_names:
            values = np.array([sample[name] for sample in patient_samples], dtype=float)
            row[name] = float("nan") if np.isnan(values).all() else float(np.nanmean(values))
        rows.append(row)
    return rows


def compare_methods(methods: dict[str, Iterable[tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]]], output_csv: str | Path | None = None) -> dict[str, dict[str, dict[str, float]]]:
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
