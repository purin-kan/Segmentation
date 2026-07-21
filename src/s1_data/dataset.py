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

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from src.s1_data.labels import boundaries_to_layer_masks, harmonize_boundaries, load_boundaries_json


def parse_subject_id(image_path: str | Path) -> str:
    """Extract the patient id from a sample filename.

    Args:
        image_path: path to an image/*.npy, named '<subject_id>_<index>'.
    Returns:
        The subject id, e.g. 'Subject_01' or 'hc01'.
    Raises:
        ValueError: if the trailing token is not an index. An unexpected
            token would split one patient across two ids, and so across
            train/val folds.
    """
    stem = Path(image_path).stem
    parts = stem.split("_")
    if len(parts) < 2 or not parts[-1].isdigit():
        raise ValueError(
            f"Cannot parse subject id from {Path(image_path).name!r}: expected "
            f"'<subject_id>_<bscan_index>'. An unexpected trailing token would "
            f"split one patient across two ids, and so across train/val folds."
        )
    return "_".join(parts[:-1])


def available_patient_ids(result_dir: str | Path) -> set[str]:
    """Patient ids present in a processed directory.

    Reads filenames only, so it stays cheap enough to call for validation
    before committing to loading any labels.

    Args:
        result_dir: directory containing {image,label}/ subdirs.
    Returns:
        set of subject ids with at least one image/label pair on disk.
    """
    result_dir = Path(result_dir)
    label_dir = result_dir / "label"
    return {
        parse_subject_id(image_path)
        for image_path in (result_dir / "image").glob("*.npy")
        if (label_dir / (image_path.stem + ".txt")).exists()
    }


def load_fold_datasets(result_dirs: Iterable[str | Path], patient_ids: Iterable[str], harmonize: bool = True) -> list["OCTDataset"]:
    """Build one OCTDataset per directory for a fold's patient ids.

    A fold spans both datasets, but each OCTDataset sees only its own
    directory, so neither can tell "this id belongs to the other dataset"
    from "this id is stale". Completeness is therefore checked here, against
    the union, and each OCTDataset is then given only the ids it owns.

    Args:
        result_dirs: processed directories to draw from, e.g.
            (DATA_ROOT/'processed/duke_dme_denoised',
             DATA_ROOT/'processed/hc_ms_denoised').
        patient_ids: one fold's train or val ids, spanning every directory.
        harmonize: passed through to OCTDataset.
    Returns:
        list of OCTDatasets, one per directory in the given order. Chain them
        for a single pass: itertools.chain(*load_fold_datasets(...)).
    Raises:
        ValueError: if an id is on disk in no directory (a stale fold) or in
            more than one (the same patient scored twice, and placeable in
            both train and val).
    """
    result_dirs = [Path(result_dir) for result_dir in result_dirs]
    requested = set(patient_ids)
    available = {result_dir: available_patient_ids(result_dir) for result_dir in result_dirs}

    if missing := requested - set().union(*available.values()):
        raise ValueError(
            f"{len(missing)} requested patient id(s) are on disk in none of "
            f"{[str(d) for d in result_dirs]}: {sorted(missing)}. The fold split is "
            f"stale with respect to what is on disk."
        )

    duplicated = {
        patient_id: [str(d) for d in result_dirs if patient_id in available[d]]
        for patient_id in requested
        if sum(patient_id in available[d] for d in result_dirs) > 1
    }
    if duplicated:
        raise ValueError(
            f"Patient id(s) present in more than one directory: {duplicated}. The same "
            f"patient would be scored twice, and could land in both train and val."
        )

    return [
        OCTDataset(result_dir, patient_ids=requested & available[result_dir], harmonize=harmonize)
        for result_dir in result_dirs
    ]


def iter_samples(datasets: Iterable["OCTDataset"]) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Iterate several OCTDatasets as one stream, carrying patient identity.

    The eval-side replacement for itertools.chain(*load_fold_datasets(...)).
    Chaining drops each dataset's .samples, so a metric row computed
    downstream can no longer be traced back to a patient, and the per-patient
    pairing in implementation_plan.md (Evaluation > Comparison) needs it.
    Training does not, and keeps using __getitem__ directly.

    Args:
        datasets: OCTDatasets to read in order, e.g. load_fold_datasets(...).
    Yields:
        (subject_id, image, layer_masks, boundaries) per B-scan.
    """
    for dataset in datasets:
        for idx in range(len(dataset)):
            yield (dataset.samples[idx]["subject_id"], *dataset[idx])


class OCTDataset:
    """One sample = one annotated B-scan: (image, layer_masks, boundaries).

    Shared loader for both DUKE-DME and HC-MS datasets.

    Expects directory structure with image/*.npy and label/*.txt (JSON).
    Label filenames must match image filenames (except extension).

    Boundaries are harmonized to the shared 6 at load time, so both datasets
    yield 6 boundaries / 5 layer masks. The label files keep their native 8
    (DUKE-DME) or 9 (HC-MS); nothing on disk is rewritten.
    """

    def __init__(self, result_dir: str | Path, patient_ids: Iterable[str] | None = None, harmonize: bool = True) -> None:
        """
        Args:
            result_dir: path to directory containing {image,label}/ subdirs.
            patient_ids: if provided, only load samples matching these IDs.
            harmonize: reduce boundaries to the shared 6 (labels.py,
                HARMONIZED_INDICES). Set False to get the dataset's native
                boundaries, e.g. for inspection.
        """
        self.result_dir = Path(result_dir)
        self.harmonize = harmonize
        self.samples = self._load_samples(patient_ids)

    def _load_samples(self, patient_ids: Iterable[str] | None) -> list[dict[str, Any]]:
        samples = []
        label_dir = self.result_dir / "label"

        requested = None if patient_ids is None else set(patient_ids)

        # Nested ids are checked against every id in the directory, not just
        # the requested ones: the two halves of a nested pair are exactly what
        # a fold split puts on opposite sides, so a filtered view sees one and
        # misses the collision.
        present = available_patient_ids(self.result_dir)
        nested = [(a, b) for a in present for b in present if b.startswith(a + "_")]
        if nested:
            raise ValueError(
                f"Patient id(s) nested inside another in {self.result_dir}: {sorted(nested)}. "
                f"One patient has parsed into two ids, which would place the same eye in both "
                f"the train and val fold."
            )

        for image_path in sorted((self.result_dir / "image").glob("*.npy")):
            label_path = label_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                continue

            subject_id = parse_subject_id(image_path)
            if requested is not None and subject_id not in requested:
                continue

            boundaries = load_boundaries_json(label_path)
            if self.harmonize:
                boundaries = harmonize_boundaries(boundaries)
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
