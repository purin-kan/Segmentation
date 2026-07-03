"""
Shared eval harness: one data loader + inference + metrics call, reused by
every method in src/methods/ (implementation_plan.md, Setup).

Meant to be imported into a Colab notebook cell, not run as a standalone
CLI script — see notebooks/02_run_methods.ipynb.
"""

from src.eval.run_all_metrics import evaluate_method


def run_experiment(method_name, segment_fn, dataset, output_csv=None):
    """
    Run one method's segment_fn over every sample in dataset, then score it
    with every metric in Metrics/.

    Args:
        method_name: label used in results tables/plots.
        segment_fn: callable(bscan, **kwargs) -> predicted mask, matching
            the segment()/forward() signature of the src/methods/* stubs.
        dataset: iterable of (bscan, ground_truth_mask) pairs, typically a
            src.data.dataset.OCTDataset test split.
        output_csv: optional path to save the per-method metrics summary.
    Returns:
        (per_sample, summary) as returned by eval.run_all_metrics.evaluate_method.
    """
    raise NotImplementedError
