"""Shared OCT dataset loader (implementation_plan.md, Setup > Shared eval harness).

Reads the denoised+normalized output of 01_preprocessing.ipynb
(data/processed/{duke_dme,hc_ms}_denoised):
  - image/*.npy: B-scan images (flattened+cropped, denoised, normalized to
    [0, 1], stored as uint8 — the signal has no more than 8 bits of real
    precision after BM3D, so OCTDataset expands back to float32 [0, 1] at
    load time rather than storing the float result directly)
  - label/*.txt: JSON boundary annotations

01_preprocessing.ipynb reads the raw MATLAB output (generate_dme_train.m,
generate_hc_train.m: data/processed/{duke_dme,hc_ms}, image/*.png) directly,
not through this class — by design, every OCTDataset-visible sample has
already gone through denoise+normalize.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from src.s1_data.labels import boundaries_to_layer_masks, load_boundaries_json

class OCTDataset:
    """One sample = one annotated B-scan: (image, layer_masks, boundaries).

    Shared loader for both DUKE-DME and HC-MS datasets.

    Expects directory structure with image/*.npy and label/*.txt (JSON).
    Label filenames must match image filenames (except extension).
    """

    def __init__(self, result_dir: str | Path, patient_ids: Iterable[str] | None = None) -> None:
        """
        Args:
            result_dir: path to directory containing {image,label}/ subdirs.
            patient_ids: if provided, only load samples matching these IDs.
        """
        self.result_dir = Path(result_dir)
        self.samples = self._load_samples(patient_ids)

    def _load_samples(self, patient_ids: Iterable[str] | None) -> list[dict[str, Any]]:
        samples = []
        image_dir = self.result_dir / "image"
        label_dir = self.result_dir / "label"

        for image_path in sorted(image_dir.glob("*.npy")):
            label_path = label_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                continue

            parts = image_path.stem.split("_")
            subject_id = "_".join(parts[:-1])
            if patient_ids is not None and subject_id not in patient_ids:
                continue

            boundaries = load_boundaries_json(label_path)
            samples.append({
                "subject_id": subject_id,
                "image_path": image_path,
                "boundaries": boundaries,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample = self.samples[idx]
        image = np.load(sample["image_path"]).astype(np.float32) / 255.0
        boundaries = sample["boundaries"]
        layer_masks = boundaries_to_layer_masks(boundaries, height=image.shape[0])
        return image, layer_masks, boundaries

    def patient_ids(self) -> list[str]:
        return sorted({s["subject_id"] for s in self.samples})
