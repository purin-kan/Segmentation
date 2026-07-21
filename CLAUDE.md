# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research reference repository for **Retinal OCT Image Segmentation via Deep Learning**, accompanying the survey paper:

> Zhang et al., "Retinal OCT image segmentation with deep learning: A review of advances, datasets, and evaluation metrics," *Computerized Medical Imaging and Graphics*, 2025. DOI: 10.1016/j.compmedimag.2025.102539

The repo collects SOTA model implementations, evaluation metric code, and dataset preprocessing scripts. There is no unified training pipeline — each model file is a self-contained reference implementation.

The root `README.md` is a general project overview. The in-progress implementation plan for a specific experiment (comparing traditional, graph-based, and deep-learning methods for retinal layer segmentation on DUKE-BOE) lives in `docs/implementation_plan.md` — check it for current task status/decisions before starting related work.

## Working Style

For research and data tasks, base answers on direct file inspection (read the actual files, folder
keys, scripts) rather than assumptions before explaining structure or making recommendations.

## Nested repos — do not modify

`external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/` and `external/Public-available-retinal-OCT-datasets/` are separate, independent git repos (own `.git`, own GitHub remotes under `ZhangHH233/`) checked in as gitlinks inside this repo. Do not edit, add, or commit files inside these two directories — treat them as read-only vendored upstream code. If a change is genuinely needed there, flag it to the user instead of editing directly.

## Repository Structure

```
src/                              # All self-implemented Python (experiment code + eval harness), not vendored
                                   # s1-s5 are numbered pipeline steps, in execution order; configs/notebooks/scripts
                                   # are cross-cutting support, not steps, so they stay unprefixed
  configs/                        # Per-method experiment configs (paths, hyperparameters, split seeds)
  s1_data/                        # Shared dataset loader + patient/volume-level split logic
  s2_preprocessing/               # Denoise (flattening/cropping happen upstream in the MATLAB extraction, so no
                                   # separate flatten/crop step here)
  s3_methods/                     # One subpackage per Family column of docs/implementation_plan.md's Methods table
    m1_traditional/                 #   1a Intensity Thresholding, 1b Canny
    m2_model_based/                 #   2  Active Contours
    m3_graph_based/                 #   3a Graph Search, 3b Graph-Cut, 3c Dynamic Programming
    m4_region_based/                #   4  Region Growing
    m5_deep_learning/               #   5a-5h CNN, FCN, U-Net, Boundary-Aware U-Net, TransUNet, Swin-UNet, 2.5D
  s4_postprocessing/               # Boundary ordering / non-crossing enforcement for DL outputs
  s5_eval/
    metrics.py                    # Aggregates region_metrics.py + boundary_metrics.py into per-method summaries/CSV
    region_metrics.py             # Dice, IoU — per layer, then averaged
    boundary_metrics.py           # MAD, RMSE — per boundary, then averaged
    run_experiment.py             # Shared eval harness: dataset -> method -> metrics -> output/*.csv
  modal_app.py                    # Modal app: GPU containers for DL training + DL scoring (see below)
  notebooks/                      # The run entry points, run locally (see below)
  scripts/                        # Local-machine helpers: visual QA + Modal volume transfer — see src/scripts/README.md

output/                           # Gitignored — metric CSVs, overlays, checkpoints (Volume-backed OUTPUT_ROOT on Modal)
data/                             # Gitignored — raw + preprocessed DUKE-DME data
  raw/                            # Untouched downloaded datasets (e.g. Publication_Dataset/, .zip archives)
  processed/                      # Output of src/s2_preprocessing/ + src/notebooks/01_preprocessing.ipynb

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

`src/s3_methods/*` are currently stubs (`raise NotImplementedError`) — implementation status per
method is tracked in `docs/implementation_plan.md`'s Methods table, not here.

### Notebooks are the run surface; GPU work runs on Modal

`notebooks/*.ipynb` are the entry points and run locally. The GPU-bound steps (DL training, DL
scoring) do not run in the notebook kernel: they run in Modal containers defined by
`src/modal_app.py`, which the notebooks invoke with `.remote()` / `.starmap()`. Everything else
runs in the kernel.

Run the notebooks in order:

1. `00_setup.ipynb` — the sole setup notebook: install deps locally, verify imports, pull
   `external/*` submodules, check `modal` is authenticated, and define `DATA_ROOT` /
   `OUTPUT_ROOT` via `src/paths.py`. Downstream notebooks read those two variables, so run
   this first.
2. `01_preprocessing.ipynb` — raw `.mat` → PNG via `external/Public-available-retinal-OCT-datasets/BOE.py`, then `src/s2_preprocessing/`. Local.
3. `02a_train_dl.ipynb` — train one DL method across folds on Modal, all folds concurrently; checkpoints land on the Volume. Classical methods skip this.
4. `02_run_methods.ipynb` — score a method via `src/s5_eval/run_experiment.py`. Classical methods (1a, 2, 3c, 4) score locally; DL methods (5d, 5b) score on Modal.
5. `03_results_analysis.ipynb` — load the `OUTPUT_ROOT/*.csv` summaries, build the cross-method comparison table. Local.

Everything under `src/` must stay plain importable Python (functions/classes), not
argparse-only CLI scripts, so it can be called directly from notebook cells. `modal_app.py` is
the one module that also exposes a CLI (`modal run src/modal_app.py`), as a batch-sweep
convenience; its functions stay importable and are what the notebooks call.

Data does not travel with the code. Input lives on the `segmentation-data` Modal Volume and
must be seeded once with `modal volume put`; results are pulled back with `modal volume get`.
`src/scripts/` holds local-machine helpers (visual QA scripts, plus those transfer commands).
See `src/scripts/README.md`.

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
`imageio`, `opencv-python`, `SimpleITK`, `matplotlib`, `bm3d`, `modal`, `torch`, `torchvision`,
`einops`, `timm`, `thop`.

The same `requirements.txt` is installed twice, from one source of truth: `00_setup.ipynb`
installs it into your local environment, and `src/modal_app.py` builds it into the Modal image
with `pip_install_from_requirements`. The Modal image additionally `apt_install`s `libgl1` and
`libglib2.0-0`, which `opencv-python` and `SimpleITK` link against and which `debian_slim` does
not ship.

There is no `setup.py` — `src/` is imported directly from the notebooks (kept plain-importable),
not installed as a package. Inside a Modal container it is made importable by
`add_local_python_source("src")`.

## Working Directory

Make changes directly in the main working checkout. Do not create or work in a git
worktree for changes in this repo (background sessions have `worktree.bgIsolation` set
to `none` in `.claude/settings.local.json` so direct edits are allowed).

## Git & Version Control

Do not add a `Co-Authored-By` / co-author tag to commit messages.

When exploring open-ended problems (e.g., duplicate git commits), first summarize a plan and pause
for confirmation instead of running many Bash commands in sequence.

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
