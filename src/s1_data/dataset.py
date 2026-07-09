"""Shared OCT dataset loader (implementation_plan.md, Setup > Shared eval harness).

Reads a BOE.py-style metadata.csv plus its sibling layers/<name>.npy
boundary files — metadata.csv has no layers_path column, so the file is
located from image_path's name (see temp/BOE_usage.md).
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.s1_data.labels import is_annotated


class OCTDataset:
    """One sample = one annotated B-scan: (image, boundaries, subject_id).

    Only slices with at least one non-NaN boundary column are kept — most
    BOE.py slices have no layer annotation (Chiu DME: 11 of 61 B-scans per
    subject, implementation_plan.md's "110 annotated B-scans total").
    """

    def __init__(self, metadata_csv, patient_ids=None):
        self.samples = self._load_metadata(Path(metadata_csv), patient_ids)

    @staticmethod
    def _load_metadata(metadata_csv, patient_ids):
        samples = []
        with open(metadata_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            # BOE.py pads header names with spaces to align columns
            # (e.g. "image_path                    "), so raw fieldnames
            # don't match the literal column names used below.
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            for row in reader:
                subject_id = row["subject_id"].strip()
                if patient_ids is not None and subject_id not in patient_ids:
                    continue
                image_path = Path(row["image_path"].strip())
                layers_path = image_path.parent.parent / "layers" / (image_path.stem + ".npy")
                boundaries = np.load(layers_path)
                if not is_annotated(boundaries):
                    continue
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
