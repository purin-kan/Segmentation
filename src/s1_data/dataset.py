"""Shared OCT dataset loader (implementation_plan.md, Setup > Shared eval harness).

Reads MATLAB-generated datasets (generate_dme_train.m, generate_hc_train.m):
  - image/*.png: B-scan images (flattened+cropped)
  - label/*.txt: JSON boundary annotations
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.s1_data.labels import load_boundaries_json


class OCTDataset:
    """One sample = one annotated B-scan: (image, boundaries, subject_id).

    Expects directory structure with image/*.png and label/*.txt (JSON).
    Label filenames must match image filenames (except extension).
    """

    def __init__(self, result_dir, patient_ids=None):
        """
        Args:
            result_dir: path to directory containing {image,label}/ subdirs.
            patient_ids: if provided, only load samples matching these IDs.
        """
        self.result_dir = Path(result_dir)
        self.samples = self._load_samples(patient_ids)

    def _load_samples(self, patient_ids):
        samples = []
        image_dir = self.result_dir / "image"
        label_dir = self.result_dir / "label"

        for image_path in sorted(image_dir.glob("*.png")):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.asarray(Image.open(sample["image_path"]))
        return image, sample["boundaries"], sample["subject_id"]

    def patient_ids(self):
        return sorted({s["subject_id"] for s in self.samples})
