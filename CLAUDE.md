# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research reference repository for **Retinal OCT Image Segmentation via Deep Learning**, accompanying the survey paper:

> Zhang et al., "Retinal OCT image segmentation with deep learning: A review of advances, datasets, and evaluation metrics," *Computerized Medical Imaging and Graphics*, 2025. DOI: 10.1016/j.compmedimag.2025.102539

The repo collects SOTA model implementations, evaluation metric code, and dataset preprocessing scripts. There is no unified training pipeline — each model file is a self-contained reference implementation.

## Repository Structure

```
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

## Key Architectural Patterns

**Model files** (`SOTAS/**/*.py`): Each file implements a single SOTA network as a PyTorch `nn.Module`. The original paper citation is embedded as a docstring at the top. Multi-file models (e.g., `SD_Layer_Net/`, `FourierNet/`) contain modular components (encoder, decoder, etc.).

**Metrics** (`Metrics/*.py`): Pure NumPy/SciPy functions, no ML dependencies. Organized into five files by metric family:
- `Region_based_metrics.py` — Dice, IoU, Precision, Recall
- `ConfusionMatrix_based_metrics.py` — Accuracy, Sensitivity, Specificity, AUC
- `Contour_based_metrics.py` — Hausdorff Distance (HD, HD95), ASSD, MAD
- `PixelError_based_metrics.py` — MSE, RMSE
- `Biomarker_based_metrics.py` — Thickness Difference, Vascularity Index

**Dataset preprocessors** (`Public-available-retinal-OCT-datasets/*.py`): CLI scripts that convert raw dataset formats (`.mat` via scipy/h5py, `.mhd` via SimpleITK) into PNG B-scans with matching label PNGs and a `metadata.csv`.

## Running Dataset Preprocessors

```bash
# BOE/Chiu DME dataset — inspect .mat keys first
python Public-available-retinal-OCT-datasets/BOE.py --input_root "<path>" --inspect

# BOE/Chiu DME dataset — full preprocessing
python Public-available-retinal-OCT-datasets/BOE.py \
  --input_root "<path_to_mat_files>" \
  --output_root "<output_path>" \
  --save_overlay

# RETOUCH dataset — edit root_dir/save_dir inside RETOUCH.py, then run
python Public-available-retinal-OCT-datasets/RETOUCH.py
```

## Dependencies

Core: `torch`, `torchvision`, `numpy`, `scipy`, `scikit-image`, `SimpleITK`, `Pillow`, `scikit-learn`, `h5py`, `imageio`

There is no `requirements.txt` or `setup.py`. Install dependencies manually per model/script as needed.

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
