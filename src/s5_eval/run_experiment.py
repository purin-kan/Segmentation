"""
Shared eval harness: one data loader + inference + metrics call, reused by
every method in src/s3_methods/ (implementation_plan.md, Setup).

Meant to be imported into a Colab notebook cell, not run as a standalone
CLI script — see src/notebooks/02_run_methods.ipynb.
"""

from src.s5_eval.metrics import evaluate_method


def run_experiment(method_name, segment_fn, dataset, output_csv=None):
    """
    Run one method's segment_fn over every sample in dataset, then score it
    with the four metrics in implementation_plan.md (Setup > Metrics):
    Dice/IoU per layer and MAD/RMSE per boundary.

    Args:
        method_name: label used in results tables/plots.
        segment_fn: callable(bscan, **kwargs) -> predicted per-layer masks
            and per-boundary positions, matching the segment()/forward()
            signature of the src/s3_methods/* stubs.
        dataset: iterable of (bscan, y_true_layers, y_true_boundaries)
            samples, typically a src.s1_data.dataset.OCTDataset test split.
        output_csv: optional path to save the per-method metrics summary.
    Returns:
        (per_sample, summary) as returned by eval.metrics.evaluate_method.
    """
    raise NotImplementedError
