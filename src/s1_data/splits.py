"""Patient-level 5-fold CV split, stratified by group
(implementation_plan.md, Setup > Data Split)."""

from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
SEED = 42


def patient_folds(patients):
    """
    Args:
        patients: dict of {patient_id: group}, group one of 'DME'/'HC'/'MS'.
    Returns:
        list of N_FOLDS (train_patient_ids, val_patient_ids) sets.

    Uses plain StratifiedKFold, not the group-aware StratifiedGroupKFold,
    because each row here is already one whole patient (not one slice) —
    the "patient-level split unit" from implementation_plan.md is
    satisfied by construction, so there's no group of correlated rows to
    keep together.
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
