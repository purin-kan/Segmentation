# Implementation Plan: Retinal Layer Segmentation in OCT Images

## Setup

- [x] **Dataset:** DUKE-DME (Chiu et al. 2015) + HC-MS (He et al. 2019)
  - **Layer-level ground truth**: Both datasets provide expert-annotated boundary/layer segmentation
  - **Complementary pathology coverage**: DUKE-DME provides pathological cases (fluid-driven structural distortion from diabetic macular edema); HC-MS adds healthy controls plus a second disease population (multiple sclerosis, causing diffuse thinning), together covering both structural and non-structural disease patterns.
  - **Includes healthy subjects**: HC-MS contributes 14 healthy controls, addressing the gap left by DUKE-DME (DME-only, no healthy cases).
  - **Established pairing in the literature**: Multiple published segmentation studies use this exact DUKE-DME + HC-MS combination.
  - **Direct comparability**: Using the same dataset combination as this literature cluster allows direct comparison of Dice score / MAD results against multiple published baselines.
  - **Practical scale for early-stage work**: Both datasets are modest in size, suitable for rapid iteration during the conventional-methods and U-Net baseline phases on limited local compute.
  - **Data Split**
    - **Split unit**: *patient-level*. B-scans within a patient are correlated slices of the same eye.
    - **Strategy**: 5-fold cross-validation, stratified by group (DME / HC / MS).
    - Per-fold composition (~9 patients held out): 2 DME, ~3 HC, ~4 MS.
    - **Rationale**: DME has only 10 patients and just 110 annotated B-scans total. A single fixed test split would strand only 1–2 DME patients in test, too few for a stable Dice/MAD estimate
- [ ] **Shared eval harness:** one data loader + inference + metrics call, reused by every
      method below, so results are comparable on identical splits/preprocessing.
- [x] **Preprocessing:**
  - **Raw extraction scope (DUKE-DME, via `generate_dme_train.m`)**: primary source, used for
    training/evaluation
    - **script**: `external/oct_preprocess/Scripts/generate_dme_train.m`, from He et al.'s vendored `oct_preprocess` repo (see `docs/citation.md`).
    - **toolbox gap**: `Preprocess.m`'s DME branch calls `nansum` (Statistics and Machine
      Learning Toolbox, not installed here). Worked around with a minimal drop-in
      `nansum(x,dim) = sum(x,dim,'omitnan')` shim at `data_helper/duke_dme/nansum.m`.
    - **inputs**: `data/raw/2015_BOE_Chiu/Subject_01.mat` – `Subject_10.mat`.
    - **output**: `data/processed/duke_dme/{image,label}/`
      - 110 `image/*.png` (flattened+cropped to 224x768 grayscale; annotated B-scans
        only, unannotated ones are filtered out inside `Preprocess.m` before saving, unlike
        `BOE.py` which keeps all 610)
      - 110 `label/*.txt` (JSON `{"bds": [...], "lesion": [...]}`: 8 boundaries x 768 columns,
        1-indexed; `lesion` is a full 224x768 binary fluid mask)
      - 10 subjects x 11 annotated B-scans each (centered on fovea)
  - **Raw extraction scope (DUKE-DME, via `BOE.py`)**: reserve source, kept for potential future
    self/semi-supervised learning on unannotated B-scans, not used by any method currently planned
    - **script**: `external/Public-available-retinal-OCT-datasets/BOE.py`, from the survey's
      companion repo (see `docs/citation.md`).
    - **inputs**: `data/raw/2015_BOE_Chiu/Subject_01.mat` – `Subject_10.mat`.
    - **regenerate**:
      ```bash
      python external/Public-available-retinal-OCT-datasets/BOE.py \
        --input_root data/raw/2015_BOE_Chiu \
        --output_root data/processed/duke_dme_unannotated \
        --save_overlay
      rm -rf data/processed/duke_dme_unannotated/{labels_fluid1,labels_fluid2,npz,overlays_fluid1,overlays_fluid2,splits}
      ```
    - **output**: `data/processed/duke_dme_unannotated/{images,layers}/` + `metadata.csv`
      - 610 `images/*.png` (768x496 grayscale, percentile-stretched to uint8, **raw**:
        unflattened/uncropped, unlike the MATLAB output above)
      - 610 `layers/*.npy` (raw `(8, width)` boundary y-positions per B-scan, from
        `manualLayers1`; all-NaN where unannotated)
      - 110 of the 610 overlap with the annotated set above (11 per subject, centered on fovea);
        the other 500 have no layer annotation and are the reason this extraction is kept
      - `labels_fluid1/2`, `overlays_fluid1/2`, `npz/`, and `splits/` that `BOE.py` also writes
        are deleted immediately after the run (see `rm` above); only `images/`, `layers/`, and
        `metadata.csv` are kept from dataset conversion.
  - **Raw extraction scope (HC-MS, via `generate_hc_train.m`)**
    - **script**: `external/oct_preprocess/Scripts/generate_hc_train.m`, from He et al.'s vendored
      `oct_preprocess` repo (see `docs/citation.md`). Run locally in MATLAB R2026a.
    - **inputs**: `data_helper/OCT_Manual_Delineations-2018_June_29/filelist.txt` /
      `segname.txt`: 35 line-matched paths into
      `data/raw/OCT_Manual_Delineations-2018_June_29/{vol,delineation}/`.
    - **output**: `data/processed/hc-ms/{image,label}/`
      - 1,715 `image/*.png` (flattened+cropped,
      1024x128 grayscale)
      - 1,715 `label/*.txt` (JSON `{"bds": [...]}` boundary y-positions,
      1-indexed)
      - 35 subjects x 49 B-scans each
      - boundaries per B-scan: 9 for all 35 patients.
  - **Flattening** (retinal curvature correction)
    - Kim at al. retinal curvature causes layer misalignment across images, and flattening corrects for it.
  - **Cropping** (remove above and below that is not the retina)
    - Kim et al. (2026) the top and bottom of the image are vitreous and choroid, which contain no retinal layers, cropping to the 1/8–5/8 band focuses processing on the region that matters.
  - **Noise reduction** (BM3D)
    - BM3D used by https://opg.optica.org/boe/fulltext.cfm?uri=boe-5-10-3568
  - **Normalization**
    - percentile-clip + min-max
      - clip 1st and 99th percentile to reduce effects of outliers (speckle noise)
      - Normalize to [0,1]
  - ~~Contrast enhancement (CLAHE)~~ [experimental, scrapped]
    - Kim et al. state CLAHE's purpose is to enhance layer visibility through local rather than global contrast adjustment.

- [ ] **Metrics:** reuse `external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/Metrics/`.
  - Region: Dice, IoU [computed per layer as a separate binary mask, then averaged]
  - Boundary: MAD, RMSE [computed per boundary, then averaged]

### To Be Decided

- [ ] **Label definition**: number of layer boundaries/classes to segment
- [ ] **Postprocessing**: how DL outputs get ordered, non-crossing boundaries, to compare
      fairly against Graph Search/DP/Graph-Cut (which enforce this inherently)

## Methods

| # | Method | Family | Source | Status |
| - | - | - | - | - |
| 1a | Intensity Thresholding | Traditional | self-implement | [ ] |
| 1b | Canny Edge Detection | Traditional | self-implement | [ ] |
| 2 | Active Contours | Model-based | self-implement | [ ] |
| 3a | Graph Search (Shortest Path) | Graph-based | self-implement | [ ] |
| 3b | Graph-Cut | Graph-based | self-implement | [ ] |
| 3c | Dynamic Programming | Graph-based | self-implement | [ ] |
| 4 | Region-based (e.g. region growing) | Region-based | self-implement | [ ] |
| 5a | CNN | Deep Learning | self-implement | [ ] |
| 5b | CNN with Graph-based | Deep Learning | self-implement | [ ] |
| 5c | Fully CNN | Deep Learning | self-implement | [ ] |
| 5d | U-Net | Deep Learning | self-implement | [ ] |
| 5e | Boundary-Aware U-Net | Deep Learning | self-implement | [ ] |
| 5f | Transformer-based (TransUNet) | Deep Learning | self-implement | [ ] |
| 5g | Swin Transformer (Swin-UNet) | Deep Learning | self-implement | [ ] |
| 5h | 2.5D | Deep Learning | self-implement | [ ] |

## Evaluation

For each method, record:

- Primary/secondary metric scores on the held-out test split
- Failure cases: where does it break down? (low contrast, speckle noise, boundaries near lesions, thin/merging layers, etc.)
- Qualitative overlay of predicted vs. ground-truth boundaries on a few representative
  B-scans
