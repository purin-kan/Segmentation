# Implementation Plan: Retinal Layer Segmentation in OCT Images

## Setup

- [x] **Dataset:** DUKE-DME (Chiu et al. 2015) + HC-MS (He et al. 2019; boundary definitions
      come from Lang et al. 2013, which delineated the same 35 scans. He 2019 is the data
      release and defines no boundaries.)
  - **Layer-level ground truth**: Both datasets provide expert-annotated boundary/layer segmentation
    - HC-MS has 9 boundaries while DUKE-DME has 8, but these are not 9 vs 8 of the same curves.
      HC-MS draws 2 that Duke does not (ELM, OS/RPE); Duke brackets the ellipsoid band with 2
      lines where HC-MS crosses it with 1, and which of Duke's 2 that single line matches is
      unresolved. Union = 10, intersection = 6.

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
    - **Dataset imbalance**: 1,715 HC-MS B-scans vs 110 Duke. Oversample Duke.
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
- [x] **Input size:** fixed 224x1024 canvas (`src/s3_methods/m5_deep_learning/canvas.py`).
  - **Pad, not resize.** Both datasets share ~3.87 µm/px axially, which is what makes MAD mean
    the same thing in each; resizing 224 and 128 to a common height scales them differently.
  - **Bottom and right only**, so every (row, column) coordinate is unchanged and boundary
    y-positions need no offset. Duke 224x768 pads 256 columns right; HC-MS 128x1024 pads 96 rows
    bottom. Predictions are cropped to native size before the eval harness.
  - **No tiling**: it would cut the lateral context 3c uses across the full scan width.

- [x] **Training protocol:** shared across methods, so differences are attributable to the method.
  - A parameter is **per-method**. Everything else is **shared**: folds, canvas, augmentation,
    optimizer/LR/early-stopping, seeds, preprocessing.
  - **Augmentation**: horizontal flip, small horizontal translation, mild intensity/contrast
    jitter. No vertical flip or rotation, both destroy boundary ordering.
  - **Early stopping** on the validation fold with fixed patience: a shared *rule*, not a shared
    epoch count, so each method reaches its own convergence. Select the best validation
    checkpoint per fold.

- [x] **Postprocessing**: enforce non-crossing boundaries on DL are compared
      fairly against Graph Search/DP/Graph-Cut (which guarantee this structurally).
  - **`enforce_non_crossing`** (`s4_postprocessing/ordering.py`): per column, running max down
    the boundary axis (`y[k] >= y[k-1]`), skipping NaNs so declined predictions stay lost
    coverage instead of poisoning boundaries beneath them. Cummax, not sort, so one misplaced
    boundary is pinned locally rather than shifting the others in its column.
  - **Ordering only**, no smoothing or gap interpolation, that decision is joint with the
    coverage metric. Wired into `m5_deep_learning/inference.py`.

## Methods

Scoped to 6 methods, one or more per family.

| # | Method | Family | Source | Status |
| - | - | - | - | - |
| 1a | Intensity Thresholding | Traditional | self-implement | [ ] |
| 2 | Active Contours | Model-based | self-implement | [ ] |
| 3c | Dynamic Programming | Graph-based | self-implement | [ ] |
| 4 | Region-based (region growing) | Region-based | self-implement | [ ] |
| 5d | U-Net | Deep Learning | self-implement | [ ] |
| 5b | CNN with Graph-based | Deep Learning | 5d + 3c | [ ] |

**5b's backbone is 5d**, fixed, so the 5b-vs-5d contrast is a clean ablation of the graph pass.
That contrast measures 3c's DP optimization *on top of* the non-crossing already enforced on every
DL method by s4, not the presence versus absence of topological structure.

**Identity by anatomical order, not labeling.** 1a, 2 and 4 detect edges/regions generically, with
no notion of boundary identity, each assigns detections to the 6 boundaries by fixed top-to-
bottom order instead, leaving a boundary NaN where no transition is defensible. Expect low
coverage and high MAD on the four low-contrast inner boundaries.

**Scope cuts.** Cut from 8 methods to 6 under a 5-day budget. The binding constraint is GPU hours,
not implementation time: 4 DL methods x 5 folds x 3 seeds is ~60 training runs, and at roughly
1.5-2 h per run (~1,460 training B-scans, 224x1024 canvas, ~50 epochs to early stopping) that is
100+ GPU-hours. 2 DL methods x 5 folds x 1 seed is ~18.

Running on Modal changes what that number costs but not what it is. Folds run concurrently, so
the budget binds on total GPU-hours billed rather than on wall-clock, and the cut stands
on the same arithmetic.

5c and 5e are the cuts, both recorded in Future Work as time rather than merit. Every family keeps
a representative and the 5b-vs-5d ablation survives, so the comparison still answers its question
with a smaller DL arm. Restore 5c first.

These per-run figures are estimates. Stage 0 replaces them with a measurement, and the scope should
be revisited once it does.

## Evaluation

For each method, record:

- Primary/secondary metric scores on each of the 5 patient-level CV folds, reported as mean +/-
  CI across folds. Each fold's held-out patients are the scoring set; there is no separate fixed
  test split (see Setup > Split). This is the per-method descriptive headline; cross-method
  comparison is paired per patient instead (see Comparison)
- Failure cases: where does it break down? (low contrast, speckle noise, boundaries near lesions, thin/merging layers, etc.)
- Qualitative overlay of predicted vs. ground-truth boundaries on a few representative
  B-scans
- Metrics stratified by dataset (Duke vs HC-MS), not only pooled: pooled scores are ~94% HC-MS.

**Seeds.** Averaged *within* a fold, not treated as independent samples (method x fold x seed as
independent observations is pseudo-replication). **1 seed per fold**, forced by the compute budget
(see Scope cuts). 1a, 2, 3c and 4 are deterministic and would run once regardless.

The exception: 5d runs 3 seeds on **one** fold, about 2 extra runs. That is enough to say whether
seed spread is small relative to fold spread, which is the question the seed protocol existed to
answer. It is not enough to average seeds within a fold, so single-seed results carry that as a
stated limitation: a per-method score includes one draw of training noise.

**Comparison.** Paired per-patient differences, not a significance test. Every patient is held out
exactly once across the 5 folds, so each method yields 45 per-patient scores. A comparison is the
mean paired difference with a bootstrap CI over patients, reported in px. Two constraints: the
difference must be computed on a common support (see Setup > Metrics), or the pair compares
different quantities; and it must be paired per patient, not read off two methods' overlapping
mean +/- CI, which is a weaker inference.

Stratified comparisons run the same way. By patient the split is 10 Duke vs 35 HC-MS, less
lopsided than the 94% B-scan imbalance. DME is reported as a per-patient distribution (n=10)
rather than a fold-level estimate: at ~2 held-out DME patients per fold, one atypical patient
swings the fold.

**No p-values.** 5 folds cannot support a test at all (2^5 sign arrangements put the minimum
two-sided p at 0.0625, before any multiple-comparison correction). A difference CI in px also
carries the effect size and its uncertainty in the units the result is argued in, which a p-value
does not. The per-patient scores are retained, so a paired test can be added later for a specific
pre-specified comparison (5b vs 5d is the likeliest) without re-running anything.

**Ranking.** Resolving a boundary is part of the task, not a precondition for being scored on it,
so coverage enters the verdict rather than sitting beside it as a caveat. Methods are tiered by
how many of the 6 boundaries they resolve, and a 6/6 method outranks a 2/6 method whatever its
MAD. Within a tier the support is equal, so aggregate MAD ranks directly and paired per-patient
differences on it are valid. Across tiers aggregate MAD is not a common quantity, is not
compared, and the per-boundary table carries that comparison instead.

A boundary counts as **resolved** only above a coverage floor and below a MAD bound. Both
conditions are needed: coverage alone makes the primary axis gameable by predicting everywhere
regardless of quality, which every DL method does for free. Set both thresholds from the Stage 1
distributions rather than in advance, since what counts as "not a real segmentation" in px is not
knowable until the ordinal methods have run.

Expect this to put every DL method in the 6/6 tier and 1a/2/4 well below it, making the headline a
capability finding rather than a precision one. The precision comparisons carrying the scientific
claim (5b vs 5d above all) sit inside the 6/6 tier, where ranking reduces to plain aggregate MAD.

**Order.** Stage 0: one fold, one seed, all 6 methods, to shake out adapters/padding/CSV plumbing
(report nothing). Run it before anything else: it converts the compute estimate below into a real
per-epoch number and catches adapter bugs before they waste an overnight training run. It also
sets the GPU type: `modal_app.benchmark_gpus()` times a short run on each candidate, and
`DEFAULT_GPU` is set from that measurement rather than an estimate. Stage 1:
classical (1a, 2, 3c, 4) at full 5-fold, near-free without training and sets the baseline floor.
These methods are deterministic and run no training loop, so Stage 1 de-risks the data/metrics/CSV
path only. Stage 2: DL (5d, then 5b). 5d first: it is 5b's pinned backbone, and it is where the
shared training loop is de-risked, validated against self-written code so model bugs stay
separable from harness bugs. 5b additionally needs 3c.

**3c is the critical path.** 5b depends on it, so slipping 3c costs two methods rather than one.
It is the one item worth reordering the schedule around.

Stage 1 must land before the Ranking thresholds can be set: the coverage floor and MAD bound come
from the classical methods' observed distributions.

**Run mechanics.** Training and DL scoring run on Modal (`src/modal_app.py`); the classical
methods run locally. Folds fan out to concurrent containers via `.starmap()`, so a 5-fold sweep
costs roughly one run of wall-clock rather than five. Checkpoint per fold to the
`segmentation-data` Volume, which is also where the CSVs land and what
`modal volume get` pulls down. Write one CSV row
per (method, fold, seed, patient): `run_experiment.py` currently collapses to one row per method,
and `OCTDataset.__getitem__` does not return `subject_id` at all, so patient identity has to be
threaded through before the paired differences can be computed. If economizing, cut seeds before
folds.

## Future Work

Methods cut from the Methods table above. Every family is still represented, so these are
additive: implementing any of them extends the comparison without changing the existing results.

| # | Method | Family | Cost of dropping | Reason |
| - | - | - | - | - |
| 5c | Fully CNN (ReLayNet) | Deep Learning | **High** | Cut for **time, not merit**: a vendored `ReLayNet_2017.py` starting point exists and it was the no-topology DL baseline alongside 5d. Restore first, ahead of 3b, if the schedule allows |
| 5e | Boundary-Aware U-Net | Deep Learning | **High** | Cut for **time, not merit**: vendored `SD_Layer_Net/`, so it carries the highest adaptation risk per hour of any remaining method under a 5-day budget. Loses the boundary-aware DL variant |
| 3b | Graph-Cut | Graph-based | **High** | Genuinely distinct from 3c: solves all boundaries simultaneously as a min s-t cut with global optimality and structural non-crossing guarantees, rather than per-boundary. The first method to restore once the DL arm is whole |
| 5f | Transformer-based (TransUNet) | Deep Learning | Moderate | Loses the transformer family, the only family left with no representative in the table. Cut on scope: the DL arm was already at its ceiling before the time cuts above |
| 1b | Canny Edge Detection | Traditional | Low | Redundant with 1a as a traditional floor: same ordinal identity assignment, different edge detector. In the literature it is a cost-map generator feeding graph search/DP, not a standalone segmenter |
| 3a | Graph Search (Shortest Path) | Graph-based | Low | Same weighted-graph formulation as 3c, different solver (Dijkstra vs DP recurrence). Near-duplicate result |
| 5a | CNN | Deep Learning | Low | Subsumed by 5b, which is 5a plus a graph pass. 5d covers the no-topology DL baseline |
| 5g | Swin Transformer (Swin-UNet) | Deep Learning | Low | No vendored starting point anywhere in `external/`, so the highest implementation cost on the list for a result confounded by the same data scarcity as 5f |
| 5h | 2.5D | Deep Learning | Low | Already marked beyond scope. No vendored file stacks multi-slice input, and the only retinal-layer-specific reference is a 2026 preprint |

## Presentation scope

Three states, kept distinct in the write-up:

| State | Methods | Treatment |
| - | - | - |
| Implemented and measured | the 6 in the Methods table | the only numbers this study produces |
| Cut but covered | every Deep Learning entry above (5a, 5c, 5e, 5f, 5g, 5h) | architecture explained, published performance given as context for what was foregone |
| Listed only | the non-DL cuts (1b, 3a, 3b) | named as future work, not presented |

**Published numbers never share a table with measured ones.** Four protocol differences stack, and
any one alone breaks comparability:

- label scheme: 6 boundaries / 5 layers here, against the native 7-9 layer schemes those papers report
- split: 5-fold CV over 45 patients here, against their fixed train/test subject splits
- data pool: DUKE-DME + HC-MS pooled here, single-dataset there
- preprocessing: the flatten/crop/BM3D chain here is not theirs

Put them in a separate block, state each one's protocol (dataset, split, layer count), and label
them published rather than measured.

**Grounding differs by method.** What is in `external/` bounds what can be claimed:

| # | Method | Vendored code | Paper citation | Published numbers |
| - | - | - | - | - |
| 5c | ReLayNet | `Lesions_Segment/ReLayNet_2017.py` | full BibTeX (Roy et al. 2017, BOE) | usable, see below |
| 5e | SD_Layer_Net | `Layers_Segment/SD_Layer_Net/` | none in repo | not without identifying the source paper first |
| 5a | CNN | none | n/a | generic architecture, no single result to quote |
| 5f | TransUNet | none | n/a | literature only |
| 5g | Swin-UNet | none | n/a | literature only |
| 5h | 2.5D | none | n/a | literature only; the only retinal-layer reference is a 2026 preprint |

No vendored TransUNet, Swin, ViT or 2.5D exists anywhere in `external/`. The name list in
`src/s3_methods/m5_deep_learning/` describes intended scope, not available code.

**The one defensible cross-reference.** ReLayNet was published on DUKE-DME (Chiu 2015), the same
data as half this pool, and reports per-layer Dice. Its per-layer Dice on RNFL, GCL+IPL, INL and
OPL against the Duke-only subset of this study's results compares the same quantity on the same
data, which is the comparability carve-out in Setup > Comparability. Still confounded by split and
preprocessing, so label it indicative, not controlled, and verify their split from the paper before
quoting. No other cut method supports even this much.

**Describe 5e accurately if presented.** "Boundary-Aware U-Net" undersells the vendored code: an
SDNet-style disentangled representation (separate anatomy and modality encoders, FiLM conditioning)
over an attention U-Net, plus a `LayerEngine` doing column-wise soft position regression with
Sobel/Laplacian kernels along the depth axis under a curvature-max constraint. That engine is a
differentiable form of the same structural prior `s4_postprocessing/ordering.py` applies post hoc,
which is the comparison worth drawing. `LayerEngine.__init__` also hardcodes `.cuda()` on its
kernels, a concrete instance of the adaptation cost that drove the cut.
