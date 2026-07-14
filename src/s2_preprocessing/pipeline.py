"""Denoise + normalize composition and parallel batch driver
(implementation_plan.md, Setup > Preprocessing)."""

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from src.s2_preprocessing.denoise import denoise, estimate_sigma
from src.s2_preprocessing.normalize import normalize


def preprocess(bscan: np.ndarray) -> np.ndarray:
    """Denoise + normalize to [0, 1], then re-quantize to uint8 for storage —
    the signal has no more than 8 bits of real precision after BM3D anyway
    (denoise() already rounds to uint8 internally); OCTDataset expands back
    to float32 [0, 1] at load time."""
    normalized = normalize(denoise(bscan, sigma=estimate_sigma(bscan)))
    return np.clip(normalized * 255, 0, 255).round().astype(np.uint8)


def _preprocess_one(image_path: Path, label_path: Path, image_dir: Path, label_dir: Path) -> None:
    bscan = np.asarray(Image.open(image_path))
    np.save(image_dir / f"{image_path.stem}.npy", preprocess(bscan))
    shutil.copy(label_path, label_dir / label_path.name)


def preprocess_and_save(raw_dir: str | Path, output_dir: str | Path, max_workers: int | None = None) -> None:
    """Read raw image/*.png + label/*.txt under raw_dir (MATLAB output),
    denoise+normalize each B-scan in parallel across max_workers processes,
    and write image/*.npy + a copy of label/*.txt (boundaries are
    unaffected by denoising) to output_dir — the layout OCTDataset expects.

    Reads raw_dir directly rather than through OCTDataset: OCTDataset only
    ever sees the denoised+normalized *.npy output, never the raw *.png.

    Args:
        raw_dir: MATLAB output dir containing {image,label}/ subdirs.
        output_dir: dir to write the denoised {image,label}/ subdirs to.
        max_workers: worker process count; None uses os.cpu_count().
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    image_dir = output_dir / "image"
    label_dir = output_dir / "label"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for image_path in sorted((raw_dir / "image").glob("*.png")):
        label_path = raw_dir / "label" / (image_path.stem + ".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_preprocess_one, image_path, label_path, image_dir, label_dir)
            for image_path, label_path in pairs
        ]
        done = 0
        for future in as_completed(futures):
            future.result()  # re-raise any worker exception in the main process
            done += 1
            if done % 200 == 0 or done == len(pairs):
                print(f"  {done}/{len(pairs)}")

    print(f"OK   {len(pairs)} B-scans -> {output_dir}")
