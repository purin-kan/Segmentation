"""Rasterize per-layer masks from boundary labels and save them as viewable
color-coded label maps, for visual QA of src/s2_preprocessing/ output.

Masks are not persisted by the pipeline — OCTDataset derives them on the fly
via boundaries_to_layer_masks. This script saves them to disk for inspection.

Runs locally against data/processed/{duke_dme,hc_ms}_denoised (or the DATA_ROOT
equivalent). Not a notebook step and not a Modal step; see src/scripts/README.md.

Edit the config block below, then run:
    python src/scripts/visualize_masks.py
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
from src.s1_data.labels import boundaries_to_layer_masks, harmonize_boundaries, load_boundaries_json

# --- config ------------------------------------------------------------
DATASET = "duke_dme"  # "duke_dme" or "hc_ms"
SUBJECT = None  # e.g. "Subject_01" to restrict to one subject, or None for all
N = 5  # number of B-scans to visualize, or None for all
HARMONIZE = True  # reduce to the shared 6 boundaries / 5 layers
# (labels.HARMONIZED_INDICES), i.e. what OCTDataset feeds the methods. False
# shows the native 7 (DUKE-DME) or 8 (HC-MS) layers.
SAVE_NPY = False  # also save raw (n_layers, H, W) uint8 masks as <stem>.npy
OUT_DIR = None  # output dir, or None for OUTPUT_ROOT/layer_masks/<dataset>
# -------------------------------------------------------------------------


def save_mask_map(masks: np.ndarray, title: str, out_path: Path) -> None:
    n_layers, height, width = masks.shape
    label_map = np.zeros((height, width), dtype=np.int64)
    for k in range(n_layers):
        label_map[masks[k].astype(bool)] = k + 1

    cmap = plt.cm.tab20.copy()
    cmap.set_under("black")

    fig, ax = plt.subplots(figsize=(10, 10 * height / width), dpi=150)
    ax.imshow(label_map, cmap=cmap, vmin=0.5, vmax=n_layers + 0.5, interpolation="nearest")
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

    out_dir = Path(OUT_DIR) if OUT_DIR else OUTPUT_ROOT / "layer_masks" / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        image = np.load(image_dir / f"{stem}.npy")
        boundaries = load_boundaries_json(label_dir / f"{stem}.txt")
        if HARMONIZE:
            boundaries = harmonize_boundaries(boundaries)
        masks = boundaries_to_layer_masks(boundaries, height=image.shape[0])
        save_mask_map(masks, title=f"{stem}  ({masks.shape[0]} layers)", out_path=out_dir / f"{stem}.png")
        if SAVE_NPY:
            np.save(out_dir / f"{stem}.npy", masks)

    print(f"{DATASET}: {len(stems)} B-scan(s) -> {out_dir}")


if __name__ == "__main__":
    main()
