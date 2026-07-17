# Implementation Plan: Retinal Layer Segmentation in OCT Images

## Setup

- [x] **Dataset:** DUKE-DME (Chiu et al. 2015) + HC-MS (He et al. 2019; boundary definitions
      come from Lang et al. 2013, which delineated the same 35 scans. He 2019 is the data
      release and defines no boundaries.)
  - **Layer-level ground truth**: Both datasets provide expert-annotated boundary/layer segmentation
    - HC-MS has 9 boundaries while DUKE-DME has 8, but the identities differ by 3, not 1.
      Union = 10, intersection = 6.

      | idx | interface | Lang 2013 (HC-MS) | Chiu 2015 (Duke) | kept |
      |----|-----------|-------------------|------------------|:----:|
      | 1 | vitreous / RNFL | ILM | top of NFL | ✓ |
      | 2 | RNFL / GCL | RNFL-GCL | NFL / GCL-IPL | ✓ |
      | 3 | IPL / INL | GCL+IPL-INL | GCL-IPL / INL | ✓ |
      | 4 | INL / OPL | INL-OPL | INL / OPL | ✓ |
      | 5 | OPL / ONL | OPL-ONL | OPL / ONL-ISM | ✓ |
      | 6 | ONL / IS | ELM | not drawn | |
      | 7 | inner edge of band 2 | `IS-OS` under reading B | ONL-ISM / ISE | |
      | 8 | outer edge of band 2 | `IS-OS` under reading A | ISE / OS-RPE | |
      | 9 | OS / RPE | OS-RPE | not drawn | |
      | 10 | RPE / choroid | BrM | bottom of OS-RPE | ✓ |

      HC-MS only: ELM, OS/RPE. Duke only: the myoid/ellipsoid split. HC-MS's `IS-OS` lands at
      idx 7 or idx 8 depending on which reading holds; the two are not established as the same
      curve and neither paper disambiguates. See `docs/label_normalization.md`.

      ``` text
        HC-MS (9 boundaries → 8 layers)
        1. RNFL — Retinal Nerve Fiber Layer
        2. GCL-IPL — Ganglion Cell Layer + Inner Plexiform Layer (combined)
        3. INL — Inner Nuclear Layer
        4. OPL — Outer Plexiform Layer
        5. ONL — Outer Nuclear Layer
        6. IS — Inner Segment (photoreceptors)
        7. OS — Outer Segment (photoreceptors)
        8. RPE — Retinal Pigment Epithelium

        Boundaries: ILM, RNFL/GCL, IPL/INL, INL/OPL, OPL/ONL, ELM, IS/OS, OS/RPE, BM

        DUKE-DME (8 boundaries → 7 layers, + a separate Fluid class [unused])
        1. NFL — Nerve Fiber Layer
        2. GCL-IPL — Ganglion Cell Layer + Inner Plexiform Layer (combined)
        3. INL — Inner Nuclear Layer
        4. OPL — Outer Plexiform Layer
        5. ONL-ISM — Outer Nuclear Layer + Inner Segment Myoid zone (combined)
        6. ISE — Inner Segment Ellipsoid zone
        7. OS-RPE — Outer Segment + Retinal Pigment Epithelium (combined)

        Boundaries: ILM, NFL/GCL-IPL, GCL-IPL/INL, INL/OPL, OPL/ONL-ISM, ONL-ISM/ISE, ISE/OS-RPE, (bottom of OS-RPE)
      ```

  - **Complementary pathology coverage**: DUKE-DME provides pathological cases (fluid-driven structural distortion from diabetic macular edema); HC-MS adds healthy controls plus a second disease population (multiple sclerosis, causing diffuse thinning), together covering both structural and non-structural disease patterns.
  - **Includes healthy subjects**: HC-MS contributes 14 healthy controls, addressing the gap left by DUKE-DME (DME-only, no healthy cases).
  - **Comparability: per layer and per boundary, not aggregate**: the 6-boundary/5-layer scheme (see label harmonization below) means aggregate scores are not comparable to published baselines, which average over more layers/boundaries. Per-layer and per-boundary scores stay directly comparable: RNFL, GCL+IPL, INL and OPL are the same layers those baselines report. Only layer 5 (ONL→BM) is a composite with no published counterpart.
  - **Practical scale for early-stage work**: Both datasets are modest in size, suitable for rapid iteration during the conventional-methods and U-Net baseline phases on limited local compute.
  - **Data Split**
    - **Split unit**: *patient-level*. B-scans within a patient are correlated slices of the same eye.
    - **Strategy**: 5-fold cross-validation, stratified by group (DME / HC / MS).
    - Per-fold composition (~9 patients held out): 2 DME, ~3 HC, ~4 MS.
    - **Rationale**: DME has only 10 patients and just 110 annotated B-scans total. A single fixed test split would strand only 1–2 DME patients in test, too few for a stable Dice/MAD estimate
    - **Dataset imbalance**: 1,715 HC-MS B-scans vs 110 Duke, roughly 15:1. Training is HC-MS-dominated unless sampling corrects for it. Sampling policy undecided, record it once chosen.
- [x] **Shared eval harness:** one data loader + inference + metrics call, reused by every
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
      - 110 `image/*.png` (flattened+cropped to **768 wide x 224 tall** grayscale; annotated
        B-scans only, unannotated ones are filtered out inside `Preprocess.m` before saving,
        unlike `BOE.py` which keeps all 610)
      - 110 `label/*.txt` (JSON `{"bds": [...], "lesion": [...]}`: 8 boundaries x 768 columns,
        1-indexed; `lesion` is a full 224x768 binary fluid mask)
        - **~31% of `bds` is NaN by design**: graders traced only a window around the fovea,
          about 529 of 768 columns, positioned per-subject rather than at a fixed offset. So:
          mask rather than fill, mask per boundary (coverage differs slightly between them),
          and use the non-NaN count in metric denominators. Duke is scored on its central ~69%
          while HC-MS is scored on its full width, which flatters Duke slightly.
        - Duke also ships a second grader (`manualLayers2`) and algorithm output
          (`automaticLayersDME`/`Normal`); only `manualLayers1` is extracted. See Metrics.
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
    - **output**: `data/processed/hc_ms/{image,label}/`
      - 1,715 `image/*.png` (flattened+cropped, **1024 wide x 128 tall** grayscale)
      - 1,715 `label/*.txt` (JSON `{"bds": [...]}` boundary y-positions,
      1-indexed)
      - 35 subjects x 49 B-scans each
      - boundaries per B-scan: 9 for all 35 patients.
  - **Flattening** (retinal curvature correction)
    - Kim et al. retinal curvature causes layer misalignment across images, and flattening corrects for it.
  - **Cropping** (remove above and below that is not the retina)
    - Kim et al. (2026) the top and bottom of the image are vitreous and choroid, which contain no retinal layers, cropping to the 1/8–5/8 band focuses processing on the region that matters.
  - **Noise reduction** (BM3D)
    - BM3D used by https://opg.optica.org/boe/fulltext.cfm?uri=boe-5-10-3568
  - **Normalization**
    - percentile-clip + min-max
      - clip 1st and 99th percentile to reduce effects of outliers (speckle noise)
      - Normalize to [0,1]
    - **label harmonization** (full analysis: `docs/label_normalization.md`)
      - Target: **6 boundaries / 5 layers**, keeping only the boundaries where both protocols
        name the same interface.
        - HC-MS drops ELM, IS/OS, OS/RPE. 9 to 6.
        - Duke drops ONL-ISM/ISE, ISE/OS-RPE. 8 to 6.
        - Kept boundaries: ILM, RNFL/GCL, IPL/INL, INL/OPL, OPL/ONL, BM.
        - Layers: RNFL, GCL+IPL, INL, OPL, ONL→BM (composite).
      - Rationale: the outer-retina correspondence between the two protocols is unresolved and
        cannot be settled from the papers. This scheme sidesteps it rather than resolving it.
        Harmonization is not the focus of the experiment.
      - All 5 layers are confirmed as definitionally intersecting.
  - ~~Contrast enhancement (CLAHE)~~ [experimental, scrapped]
    - Kim et al. state CLAHE's purpose is to enhance layer visibility through local rather than global contrast adjustment.

- [x] **Metrics:** reuse `external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/Metrics/`.
  - Region: Dice, IoU [computed per layer as a separate binary mask, then averaged]
  - Boundary: MAD, RMSE [computed per boundary, then averaged]
    - **MAD is the headline metric; Dice is supporting.** Dice scales with layer thickness, so the
    thick composite layer 5 scores high regardless of method. MAD measures boundary displacement directly and is unaffected.
    - **Coverage** fraction of annotated columns a method actually predicts (non-NaN),
    per boundary then averaged. A method may not be able to resolve a prediction.

### To Be Decided

- [ ] **Input size policy**: Duke is 768 wide x 224 tall, HC-MS is 1024 wide x 128 tall.
      Different in *both* dimensions, so a single model needs a policy. Blocks the DL methods
      (5a-5h). Undecided.
  - **Pad, do not resize.** The height difference is a crop window, not a resolution difference;
    both datasets share ~3.87 µm/px axially, which is what makes MAD mean the same thing in
    each. Resizing 224 and 128 to a common height scales them by different factors and destroys
    that. Pad HC-MS 128 -> 224 instead.
  - Padding shifts the labels: pad *n* rows on top and every boundary y must shift by *n*.
  - Candidate policy covering 5a-5h: pad HC-MS to 224 tall, tile both into 224x224 windows.
    No resampling, and it suits the fixed-size transformers (5f/5g) as well as the
    fully-convolutional nets (which could otherwise take native sizes with per-dataset batches).
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
| 5h | 2.5D [beyond scope] | Deep Learning | self-implement | [ ] |

## Evaluation

For each method, record:

- Primary/secondary metric scores on the held-out test split
- Failure cases: where does it break down? (low contrast, speckle noise, boundaries near lesions, thin/merging layers, etc.)
- Qualitative overlay of predicted vs. ground-truth boundaries on a few representative
  B-scans
