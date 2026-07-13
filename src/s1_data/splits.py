"""Patient-level 5-fold CV split, stratified by group
(implementation_plan.md, Setup > Data Split)."""

import json
from collections.abc import Iterable
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
SEED = 42


def build_patient_groups(duke_dme_patient_ids: Iterable[str], hc_ms_patient_ids: Iterable[str]) -> dict[str, str]:
    """
    Build the {patient_id: group} dict patient_folds() expects, combining
    DUKE-DME and HC-MS patient IDs.

    Args:
        duke_dme_patient_ids: OCTDataset(duke_dme_dir).patient_ids() — all 'DME'.
        hc_ms_patient_ids: OCTDataset(hc_ms_dir).patient_ids() — group is
            read off each id's 'hc'/'ms' prefix (generate_hc_train.m names
            subjects e.g. 'hc01_...', 'ms06_...').
    Returns:
        dict {patient_id: 'DME'/'HC'/'MS'}.
    """
    groups = {patient_id: "DME" for patient_id in duke_dme_patient_ids}
    for patient_id in hc_ms_patient_ids:
        prefix = patient_id[:2].lower()
        if prefix == "hc":
            groups[patient_id] = "HC"
        elif prefix == "ms":
            groups[patient_id] = "MS"
        else:
            raise ValueError(f"Unrecognized HC-MS patient id prefix: {patient_id!r}")
    return groups


def patient_folds(patients: dict[str, str]) -> list[tuple[set[str], set[str]]]:
    """
    Args:
        patients: dict of {patient_id: group}, group one of 'DME'/'HC'/'MS'.
    Returns:
        list of N_FOLDS (train_patient_ids, val_patient_ids) sets.

    Uses plain StratifiedKFold: each row is already one patient, so
    there's no group of correlated rows to keep together.
    """
    patient_ids = list(patients.keys())
    groups = [patients[p] for p in patient_ids]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in skf.split(patient_ids, groups):
        train_ids = {patient_ids[i] for i in train_idx}
        val_ids = {patient_ids[i] for i in val_idx}
        folds.append((train_ids, val_ids))
    return folds


def save_folds(folds: list[tuple[set[str], set[str]]], path: str | Path) -> None:
    """Persist patient_folds() output to JSON, so downstream notebooks don't
    need to recompute (rerun 01_preprocessing.ipynb) to get the same split.

    Args:
        folds: patient_folds()'s return value.
        path: file to write, e.g. DATA_ROOT / 'processed/folds.json'.
    """
    payload = [
        {"train": sorted(train_ids), "val": sorted(val_ids)}
        for train_ids, val_ids in folds
    ]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_folds(path: str | Path) -> list[tuple[set[str], set[str]]]:
    """Load folds written by save_folds(), in the same shape patient_folds() returns."""
    with open(path) as f:
        payload = json.load(f)
    return [(set(fold["train"]), set(fold["val"])) for fold in payload]
