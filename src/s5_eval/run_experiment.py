"""
Shared eval harness, reused by every method in src/s3_methods/
(implementation_plan.md, Setup).

Imported into src/notebooks/02_run_methods.ipynb, not run as a
standalone CLI script.
"""

import csv
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from src.s5_eval.metrics import aggregate_per_patient, evaluate_method


def run_experiment(method_name: str, segment_fn: Callable[..., tuple[np.ndarray, np.ndarray]], dataset: Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray]], fold: int | None = None, seed: int | None = None, output_csv: str | Path | None = None) -> tuple[list[dict[str, float]], list[dict[str, object]], dict[str, dict[str, float]]]:
    """
    Run one method's segment_fn over every sample in dataset, then score it
    with the metrics in implementation_plan.md (Setup > Metrics): Dice/IoU per
    layer and MAD/RMSE/coverage per boundary.

    Args:
        method_name: label used in results tables/plots.
        segment_fn: callable(bscan, **kwargs) -> predicted per-layer masks
            and per-boundary positions, matching the segment()/forward()
            signature of the src/s3_methods/* stubs.
        dataset: iterable of (subject_id, bscan, y_true_layers,
            y_true_boundaries) samples, i.e.
            s1_data.dataset.iter_samples(load_fold_datasets(...)) over one
            fold's held-out patients. The subject_id is what makes the
            per-patient rows below possible.
        fold: which CV fold produced these predictions, recorded in the CSV.
        seed: which training seed, recorded in the CSV. None for the
            deterministic methods (1a, 2, 3c, 4).
        output_csv: optional path to save the per-patient metrics.
    Returns:
        (per_sample, per_patient, summary). per_sample is one metric dict per
        B-scan (failure-case inspection); per_patient is one row per held-out
        patient, the unit paired comparisons consume; summary is the
        per-method mean/std over B-scans.
    """
    subject_ids = []
    samples = []
    for subject_id, bscan, y_true_layers, y_true_boundaries in dataset:
        y_pred_layers, y_pred_boundaries = segment_fn(bscan)
        subject_ids.append(subject_id)
        samples.append((y_true_layers, y_pred_layers, y_true_boundaries, y_pred_boundaries))

    per_sample, summary = evaluate_method(samples)
    per_patient = aggregate_per_patient(subject_ids, per_sample)

    if output_csv is not None:
        _write_per_patient_csv(output_csv, method_name, fold, seed, per_patient)

    return per_sample, per_patient, summary


def _write_per_patient_csv(path: str | Path, method_name: str, fold: int | None, seed: int | None, per_patient: list[dict[str, object]]) -> None:
    """Write one row per (method, fold, seed, patient).

    The row granularity implementation_plan.md (Evaluation > Run mechanics)
    calls for: paired per-patient differences need every patient's score kept
    separately, so collapsing to one row per method here would discard the
    comparison unit.
    """
    if not per_patient:
        raise ValueError("No per-patient rows to write: the fold yielded zero scored B-scans.")

    metric_names = [name for name in per_patient[0] if name not in ("subject_id", "n_bscans")]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "fold", "seed", "subject_id", "n_bscans"] + metric_names)
        for row in per_patient:
            writer.writerow(
                [method_name, fold, seed, row["subject_id"], row["n_bscans"]]
                + [row[name] for name in metric_names]
            )
