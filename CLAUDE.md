# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research reference repository for **Retinal OCT Image Segmentation via Deep Learning**, accompanying the survey paper:

> Zhang et al., "Retinal OCT image segmentation with deep learning: A review of advances, datasets, and evaluation metrics," *Computerized Medical Imaging and Graphics*, 2025. DOI: 10.1016/j.compmedimag.2025.102539

The repo collects SOTA model implementations, evaluation metric code, and dataset preprocessing scripts. There is no unified training pipeline — each model file is a self-contained reference implementation.

The root `README.md` is a general project overview. The in-progress implementation plan for a specific experiment (comparing traditional, graph-based, and deep-learning methods for retinal layer segmentation on DUKE-BOE) lives in `docs/implementation_plan.md` — check it for current task status/decisions before starting related work.

## Nested repos — do not modify

`external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/` and `external/Public-available-retinal-OCT-datasets/` are separate, independent git repos (own `.git`, own GitHub remotes under `ZhangHH233/`) checked in as gitlinks inside this repo. Do not edit, add, or commit files inside these two directories — treat them as read-only vendored upstream code. If a change is genuinely needed there, flag it to the user instead of editing directly.

## Repository Structure

```
src/                              # All self-implemented Python (experiment code + eval harness), not vendored
  data/                           # Shared dataset loader + patient/volume-level split logic
  preprocessing/                  # Flattening, ROI cropping, denoise, contrast enhancement
  methods/                        # One subpackage per Family column of docs/implementation_plan.md's Methods table
    traditional/                  #   1a Intensity Thresholding, 1b Canny
    model_based/                  #   2  Active Contours
    graph_based/                  #   3a Graph Search, 3b Graph-Cut, 3c Dynamic Programming
    region_based/                 #   4  Region Growing
    deep_learning/                #   5a-5h CNN, FCN, U-Net, Boundary-Aware U-Net, TransUNet, Swin-UNet, 2.5D
  postprocessing/                 # Boundary ordering / non-crossing enforcement for DL outputs
  eval/
    run_all_metrics.py            # Wrapper calling every metric in external/Retinal_OCT_.../Metrics/
    run_experiment.py             # Shared eval harness: dataset -> method -> metrics -> output/*.csv
  configs/                        # Per-method experiment configs (paths, hyperparameters, split seeds)
  notebooks/                      # Colab notebooks — the actual run entry points (see below)
  scripts/                        # Local-machine helpers (NOT Colab): Google Drive relay — see src/scripts/README.md

output/                           # Gitignored — metric CSVs, overlays, checkpoints (Drive-backed OUTPUT_ROOT on Colab)
data/                             # Gitignored — raw + preprocessed DUKE-DME data
  raw/                            # Untouched downloaded datasets (e.g. Publication_Dataset/, .zip archives)
  processed/                      # Output of src/preprocessing/ + src/notebooks/01_preprocessing.ipynb

docs/                             # implementation_plan.md (task status/decisions — read first), citation.md

external/                         # Vendored reference repos (read-only, see above)
  Retinal_OCT_Image_Segmentation_via_Deep_Learning/
    SOTAS/
      Layers_Segment/    # Layer segmentation models (BioNet, FourierNet, MGUNet, etc.)
      Lesions_Segment/   # Lesion/fluid segmentation models (AnoGAN, ReLayNet, YNet, etc.)
    Metrics/             # Evaluation metric implementations by category
    Datasets.md          # Dataset catalog with download links

  Public-available-retinal-OCT-datasets/
    BOE.py               # Preprocessing for DUKE-BOE/Chiu DME .mat datasets
    RETOUCH.py           # Preprocessing for RETOUCH .mhd volumes → PNG B-scans
    Readme.md            # Extended dataset catalog
```

`src/methods/*` are currently stubs (`raise NotImplementedError`) — implementation status per
method is tracked in `docs/implementation_plan.md`'s Methods table, not here.

### Notebooks are the run surface

This project's experiment code executes exclusively on a Google Colab server (connected via the
VS Code Colab extension, working directly against local workspace files — no `git clone` step).
The experiment has no local run surface — `notebooks/*.ipynb` are the actual entry points, run in
order:

1. `00_setup.ipynb` — the sole setup notebook: install deps, verify imports, pull `external/*`
   submodules. Its (optional) Drive cell mounts Google Drive and defines `DATA_ROOT` / `OUTPUT_ROOT`
   at `MyDrive/Segmentation/` so data + results persist across the ephemeral runtime. Downstream
   notebooks read those two variables, so run this first.
2. `01_preprocessing.ipynb` — raw `.mat` → PNG via `external/Public-available-retinal-OCT-datasets/BOE.py`, then `src/preprocessing/`
3. `02_run_methods.ipynb` — import one `src/methods/*` implementation, score it via `src/eval/run_experiment.py`; writes per-method CSVs under `OUTPUT_ROOT`
4. `03_results_analysis.ipynb` — load the `OUTPUT_ROOT/*.csv` summaries, build the cross-method comparison table

Everything under `src/` must stay plain importable Python (functions/classes), not
argparse-only CLI scripts, so it can be called directly from notebook cells.

The one exception to "runs on Colab": `src/scripts/` holds **local-machine** helpers that run on your
own computer, not the Colab VM — a Google Drive relay to pull `OUTPUT_ROOT` results down and clear
them off Drive. See `src/scripts/README.md`.

## Key Architectural Patterns

**Model files** (`external/Retinal_OCT_.../SOTAS/**/*.py`): Each file implements a single SOTA network as a PyTorch `nn.Module`. The original paper citation is embedded as a docstring at the top. Multi-file models (e.g., `SD_Layer_Net/`, `FourierNet/`) contain modular components (encoder, decoder, etc.).

**Metrics** (`external/Retinal_OCT_.../Metrics/*.py`): Pure NumPy/SciPy functions, no ML dependencies. Organized into five files by metric family:
- `Region_based_metrics.py` — Dice, IoU, Precision, Recall
- `ConfusionMatrix_based_metrics.py` — Accuracy, Sensitivity, Specificity, AUC
- `Contour_based_metrics.py` — Hausdorff Distance (HD, HD95), ASSD, MAD
- `PixelError_based_metrics.py` — MSE, RMSE
- `Biomarker_based_metrics.py` — Thickness Difference, Vascularity Index

**Dataset preprocessors** (`external/Public-available-retinal-OCT-datasets/*.py`): CLI scripts that convert raw dataset formats (`.mat` via scipy/h5py, `.mhd` via SimpleITK) into PNG B-scans with matching label PNGs and a `metadata.csv`.

## Running Dataset Preprocessors

```bash
# BOE/Chiu DME dataset — inspect .mat keys first
python external/Public-available-retinal-OCT-datasets/BOE.py --input_root "<path>" --inspect

# BOE/Chiu DME dataset — full preprocessing
python external/Public-available-retinal-OCT-datasets/BOE.py \
  --input_root "<path_to_mat_files>" \
  --output_root "<output_path>" \
  --save_overlay

# RETOUCH dataset — edit root_dir/save_dir inside RETOUCH.py, then run
python external/Public-available-retinal-OCT-datasets/RETOUCH.py
```

## Dependencies

Pinned in `requirements.txt`: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `Pillow`, `h5py`,
`imageio`, `opencv-python`, `SimpleITK`, `matplotlib`, `torch`, `torchvision`, `einops`, `timm`, `thop`.

`00_setup.ipynb` installs them on the Colab runtime with `pip install -r requirements.txt`. There is
no `setup.py` — `src/` is imported directly from the notebooks (kept plain-importable), not installed
as a package.

## Working Directory

Make changes directly in the main working checkout. Do not create or work in a git
worktree for changes in this repo (background sessions have `worktree.bgIsolation` set
to `none` in `.claude/settings.local.json` so direct edits are allowed).

## Git Commit Conventions

Do not add a `Co-Authored-By` / co-author tag to commit messages.

## Domain Abbreviations

When reading or writing about datasets and pathologies:

| Abbrev | Meaning |
|--------|---------|
| AMD | Age-Related Macular Degeneration |
| CNV | Choroidal Neovascularization |
| DME | Diabetic Macular Edema |
| DR | Diabetic Retinopathy |
| IRF/SRF/PED | Intraretinal/Subretinal Fluid, Pigment Epithelial Detachment |
| MH | Macular Hole |
| H. | Healthy |
| nAMD | Neovascular AMD |
| B-scan | 2D cross-sectional OCT slice; Volume = 3D stack of B-scans |
