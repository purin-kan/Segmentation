# Implementation Plan: Retinal Layer Segmentation in OCT Images

## Setup

- [x] **Dataset:** DUKE-DME (Chiu et al. 2015) + HC-MS
  - **Layer-level ground truth**: Both datasets provide expert-annotated boundary/layer segmentation
  - **Complementary pathology coverage**: DUKE-DME provides pathological cases (fluid-driven structural distortion from diabetic macular edema); HC-MS adds healthy controls plus a second disease population (multiple sclerosis, causing diffuse thinning) — together covering both structural and non-structural disease patterns.
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
- [ ] **Preprocessing:**
  - Flattening (retinal curvature correction)
    - Kim at al. retinal curvature causes layer misalignment across images, and flattening corrects for it.
  - Cropping (remove above and below that is not the retina)
    - Kim et al. (2026) the top and bottom of the image are vitreous and choroid, which contain no retinal layers, cropping to the 1/8–5/8 band focuses processing on the region that matters.
  - (Gaussian?) Noise reduction
    - Li et al. (2020, DeepRetina) effect graph based methods
    - BM3D used by https://opg.optica.org/boe/fulltext.cfm?uri=boe-5-10-3568
  - ~~Contrast enhancement (CLAHE)~~ [experimental, scrapped]
    - Kim et al. state CLAHE's purpose is to enhance layer visibility through local rather than global contrast adjustment.

- [ ] **Metrics:** reuse `external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/Metrics/`.
  - Region: Dice, IoU [computed per layer as a separate binary mask, then averaged]
  - Boundary: MAD, RMSE [computed per boundary, then averaged]

### To Be Decided

- [ ] **Label definition** — number of layer boundaries/classes to segment
- [ ] **Postprocessing** — how DL outputs get ordered, non-crossing boundaries, to compare
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
- Failure cases — where does it break down? (low contrast, speckle noise, boundaries near lesions, thin/merging layers, etc.)
- Qualitative overlay of predicted vs. ground-truth boundaries on a few representative
  B-scans
