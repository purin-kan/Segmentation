"""Overlay boundary annotations onto denoised B-scan images, for visual QA of
src/s2_preprocessing/ output.

Runs locally against data/processed/{duke_dme,hc_ms}_denoised (or the DATA_ROOT
equivalent) — the output of src/notebooks/01_preprocessing.ipynb. Not a notebook
step and not a Modal step; see src/scripts/README.md.

Edit the config block below, then run:
    python src/scripts/visualize_boundaries.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import resolve_roots
from src.s1_data.labels import harmonize_boundaries, load_boundaries_json

# --- config ------------------------------------------------------------
DATASET = "duke_dme"  # "duke_dme" or "hc_ms"
SUBJECT = None  # e.g. "Subject_01" to restrict to one subject, or None for all
N = 5  # number of B-scans to visualize, or None for all
HARMONIZE = True  # reduce to the shared 6 boundaries (labels.HARMONIZED_INDICES),
# i.e. what OCTDataset feeds the methods. False shows the native 8 (DUKE-DME) or
# 9 (HC-MS) as stored in label/*.txt.
OUT_DIR = None  # output dir, or None for OUTPUT_ROOT/boundary_overlays/<dataset>
# -------------------------------------------------------------------------


def overlay(image: np.ndarray, boundaries: np.ndarray, title: str, out_path: Path) -> None:
    height, width = image.shape
    cols = np.arange(width)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[: boundaries.shape[0]]

    fig, ax = plt.subplots(figsize=(10, 10 * height / width), dpi=150)
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    for k in range(boundaries.shape[0]):
        ax.plot(cols, boundaries[k], color=colors[k], linewidth=0.8)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    DATA_ROOT, OUTPUT_ROOT = resolve_roots()
    result_dir = DATA_ROOT / "processed" / f"{DATASET}_denoised"
    image_dir, label_dir = result_dir / "image", result_dir / "label"
    if not image_dir.exists():
        sys.exit(f"{image_dir} not found — run 01_preprocessing.ipynb first.")

    stems = sorted(p.stem for p in image_dir.glob("*.npy"))
    if SUBJECT:
        stems = [s for s in stems if s.startswith(f"{SUBJECT}_")]
    if N is not None:
        stems = stems[:N]
    if not stems:
        sys.exit("No matching B-scans found.")

    out_dir = Path(OUT_DIR) if OUT_DIR else OUTPUT_ROOT / "boundary_overlays" / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        image = np.load(image_dir / f"{stem}.npy").astype(np.float32) / 255.0
        boundaries = load_boundaries_json(label_dir / f"{stem}.txt")
        if HARMONIZE:
            boundaries = harmonize_boundaries(boundaries)
        overlay(image, boundaries, title=f"{stem}  ({boundaries.shape[0]} boundaries)", out_path=out_dir / f"{stem}.png")

    print(f"{DATASET}: {len(stems)} B-scan(s) -> {out_dir}")


if __name__ == "__main__":
    main()
