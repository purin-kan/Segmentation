"""HC-MS dataset loader (implementation_plan.md, Setup > Shared eval harness).

Reads the image/*.png + label/*.txt output of
external/oct_preprocess/Scripts/generate_hc_train.m (see temp/HCMS_usage.md)
into the same (image, boundaries, subject_id) sample shape as
src.s1_data.dataset.OCTDataset, so both datasets can share one eval harness.

Blocked on actually running generate_hc_train.m — that output doesn't exist
locally yet. Notes for whoever implements this:
  - label/*.txt is JSON {"bds": [...]}, not a .npy — parse and reshape to
    the (n_boundaries, width) convention src.s1_data.labels expects.
  - MATLAB is 1-indexed: subtract 1 from every row position in "bds".
"""


class HCMSDataset:
    def __init__(self, result_dir, patient_ids=None):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def patient_ids(self):
        raise NotImplementedError
