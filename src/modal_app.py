"""Modal app for the GPU-bound steps: DL training and DL scoring.

Wraps the existing `train()` and `run_experiment()` rather than reimplementing
them. Classical methods (1a, 2, 3c, 4) are CPU-only and deterministic, so they
keep running locally in `02_run_methods.ipynb`.

Notebooks call these functions inside a `with app.run():` block; `modal run
src/modal_app.py` runs the same work as a batch sweep from the CLI.

Volume layout:

    /vol/data/processed/{duke_dme_denoised,hc_ms_denoised}/   inputs
    /vol/data/processed/folds.json                            fold definitions
    /vol/output/checkpoints/<method>_fold<f>_seed<s>.pt       checkpoints
    /vol/output/<method>_fold<f>.csv                          per-patient metrics
"""

from pathlib import Path

import modal

APP_NAME = "retinal-oct-segmentation"
VOLUME_NAME = "segmentation-data"
VOLUME_PATH = "/vol"

# Container-side roots. paths.resolve_roots() returns these when MODAL_TASK_ID
# is set; keep the two in sync.
DATA_ROOT = Path(VOLUME_PATH) / "data"
OUTPUT_ROOT = Path(VOLUME_PATH) / "output"

PROCESSED_SUBDIRS = ("processed/duke_dme_denoised", "processed/hc_ms_denoised")
FOLDS_PATH = "processed/folds.json"

# Override per call with .with_options(gpu=...). Set this from a measurement,
# not an estimate: see benchmark_gpus().
DEFAULT_GPU = "A10"

# implementation_plan.md estimates 1.5-2 h per training run; leave headroom for
# a slower GPU than the one that estimate assumed.
TRAIN_TIMEOUT = 6 * 60 * 60
SCORE_TIMEOUT = 60 * 60

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    # opencv-python and SimpleITK link against these. Colab's base image
    # shipped them; debian_slim does not.
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .add_local_python_source("src")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


def _build_model(method: str):
    """Instantiate a method's model class.

    Imported lazily so the stubs' NotImplementedError is raised when a method
    is actually requested, not when this module is imported.
    """
    if method == "unet":
        from src.s3_methods.m5_deep_learning.d_unet import UNet

        return UNet()
    if method == "cnn_graph":
        from src.s3_methods.m5_deep_learning.b_cnn_graph import CNNGraph

        return CNNGraph()
    raise ValueError(f"Unknown DL method {method!r}. Expected 'unet' or 'cnn_graph'.")


def _fold_datasets(fold: int):
    """Return (train_ids, val_ids) and the processed directories for a fold."""
    from src.s1_data.splits import load_folds

    processed = [DATA_ROOT / sub for sub in PROCESSED_SUBDIRS]
    train_ids, val_ids = load_folds(DATA_ROOT / FOLDS_PATH)[fold]
    return processed, train_ids, val_ids


def checkpoint_path(method: str, fold: int, seed: int) -> Path:
    """Checkpoint location for one (method, fold, seed) run."""
    return OUTPUT_ROOT / "checkpoints" / f"{method}_fold{fold}_seed{seed}.pt"


@app.function(gpu=DEFAULT_GPU, volumes={VOLUME_PATH: volume}, timeout=TRAIN_TIMEOUT)
def train_fold(method: str, fold: int, seed: int = 42, max_epochs: int = 100, batch_size: int = 4, lr: float = 1e-3, patience: int = 10) -> dict:
    """Train one method on one fold.

    Calls src.s3_methods.m5_deep_learning.training.train() unchanged. Use
    .map()/.starmap() to run all 5 folds concurrently.

    Args:
        method: 'unet' or 'cnn_graph'.
        fold: index into folds.json.
        seed: training seed, recorded in the checkpoint name.
        max_epochs, batch_size, lr, patience: forwarded to train().
    Returns:
        {'method', 'fold', 'seed', 'checkpoint', 'history'}.
    """
    from src.s3_methods.m5_deep_learning.training import CanvasDataset, train
    from src.s1_data.dataset import load_fold_datasets

    volume.reload()
    processed, train_ids, val_ids = _fold_datasets(fold)

    train_set = CanvasDataset(load_fold_datasets(processed, train_ids), augment_samples=True, seed=seed)
    val_set = CanvasDataset(load_fold_datasets(processed, val_ids))

    path = checkpoint_path(method, fold, seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    history = train(
        _build_model(method),
        train_set,
        val_set,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        seed=seed,
        checkpoint_path=path,
    )
    volume.commit()

    return {"method": method, "fold": fold, "seed": seed, "checkpoint": str(path), "history": history}


@app.function(gpu=DEFAULT_GPU, volumes={VOLUME_PATH: volume}, timeout=SCORE_TIMEOUT)
def score_fold(method: str, fold: int, seed: int = 42) -> dict:
    """Score a trained checkpoint on its fold's held-out patients.

    Calls src.s5_eval.run_experiment.run_experiment() unchanged, so the CSV
    keeps its one-row-per-(method, fold, seed, patient) shape.

    Returns:
        {'method', 'fold', 'seed', 'csv', 'summary'}.
    """
    import torch

    from src.s1_data.dataset import iter_samples, load_fold_datasets
    from src.s3_methods.m5_deep_learning.inference import make_segment_fn
    from src.s5_eval.run_experiment import run_experiment

    volume.reload()
    processed, _, val_ids = _fold_datasets(fold)

    path = checkpoint_path(method, fold, seed)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}. Run train_fold for this (method, fold, seed) first.")

    model = _build_model(method)
    model.load_state_dict(torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu"))

    output_csv = OUTPUT_ROOT / f"{method}_fold{fold}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    _, _, summary = run_experiment(
        method_name=method,
        segment_fn=make_segment_fn(model),
        dataset=iter_samples(load_fold_datasets(processed, val_ids)),
        fold=fold,
        seed=seed,
        output_csv=output_csv,
    )
    volume.commit()

    return {"method": method, "fold": fold, "seed": seed, "csv": str(output_csv), "summary": summary}


def benchmark_gpus(method: str = "unet", fold: int = 0, gpu_types: tuple[str, ...] = ("T4", "A10", "A100"), max_epochs: int = 1) -> list[dict]:
    """Time a short training run on each GPU type, to pick DEFAULT_GPU.

    Replaces implementation_plan.md's estimated per-run cost with a measured
    one, which its Stage 0 already calls for.

    Requires an implemented model class: every entry in _build_model() is
    still a NotImplementedError stub, so this cannot run until at least
    d_unet.py is written.

    Call inside `with app.run():`. Returns one dict per GPU type with the
    wall-clock seconds for max_epochs epochs.
    """
    import time

    results = []
    for gpu in gpu_types:
        started = time.perf_counter()
        run = train_fold.with_options(gpu=gpu).remote(method=method, fold=fold, max_epochs=max_epochs, patience=max_epochs)
        results.append({"gpu": gpu, "seconds": time.perf_counter() - started, "epochs": len(run["history"])})
    return results


@app.local_entrypoint()
def main(method: str = "unet", folds: str = "0,1,2,3,4", seed: int = 42, score: bool = True) -> None:
    """Train (and optionally score) a method across folds, all folds in parallel.

    modal run src/modal_app.py --method unet
    modal run src/modal_app.py --method unet --folds 0 --no-score
    """
    fold_ids = [int(f) for f in folds.split(",")]

    for run in train_fold.starmap([(method, fold, seed) for fold in fold_ids]):
        print(f"trained fold {run['fold']}: {len(run['history'])} epochs -> {run['checkpoint']}")

    if score:
        for run in score_fold.starmap([(method, fold, seed) for fold in fold_ids]):
            print(f"scored fold {run['fold']} -> {run['csv']}")
