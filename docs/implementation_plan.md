# Implementation Plan: Retinal Layer Segmentation in OCT Images

1. decide preprocessing pipeline (Items in setup)
2. decide on data split strategy, look at relevant literature
3. decide on metrics to use (all?)
4. decide on post processing
5. compute reports after running

## Setup

- [x] **Dataset:** DUKE-DME
- [ ] **Shared eval harness:** one data loader + inference + metrics call, reused by every
      method below, so results are comparable on identical splits/preprocessing.
- [ ] **Preprocessing:**
  - Flattening (retinal curvature correction)
  - Cropping (ROI around retina)
  - Noise reduction
  - Contrast enhancement
- [ ] **Metrics:** reuse `external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/Metrics/`.
  - Primary: boundary-based — MAD / HD95 (layers are thin; pixel-overlap metrics hide
    boundary shifts)
  - Secondary: Dice / IoU, per-layer thickness error

### To Be Decided

- [ ] **Split strategy** — patient/volume-level train/val/test split (avoid B-scans from the
      same eye leaking across splits)
- [ ] **Label definition** — number of layer boundaries/classes to segment
- [ ] **Preprocessing methods** — concrete algorithm choice for noise reduction (e.g. BM3D
      vs. median filter) and contrast enhancement (e.g. CLAHE vs. histogram equalization)
- [ ] **2.5D backbone** — which architecture (5h) the 2.5D input stacking is applied to
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
