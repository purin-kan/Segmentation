# Retinal OCT Image Segmentation

This repository is a working sandbox for comparing traditional, graph-based, and deep-learning
methods for retinal layer/lesion segmentation in OCT images, built on top of the reference
implementations collected in the accompanying survey:

> Zhang et al., "Retinal OCT image segmentation with deep learning: A review of advances,
> datasets, and evaluation metrics," *Computerized Medical Imaging and Graphics*, 2025.
> DOI: [10.1016/j.compmedimag.2025.102539](https://doi.org/10.1016/j.compmedimag.2025.102539)

## Setup (Google Colab via VS Code)

This project runs against a Google Colab kernel connected through the VS Code Colab extension,
executing cells remotely while keeping your local workspace files — there's no `git clone` step;
notebooks assume they're running from inside this repo root.

```python
!pip install -r requirements.txt
```

Colab preinstalls several of these packages (`torch`, `torchvision`, `numpy`, `scipy`,
`scikit-learn`, `Pillow`, `scikit-image`) already, so the install step mainly picks up
`SimpleITK`, `h5py`, `opencv-python`, `einops`, `timm`, and `thop`.

Notes:

- Each Colab runtime is ephemeral — re-run the install cell after every disconnect/restart.
- `data/` is gitignored — optionally mount Google Drive
  (`from google.colab import drive; drive.mount('/content/drive')`) and point `data/raw/` /
  `data/processed/` at a Drive-backed path so files persist across sessions.
- Dataset preprocessing scripts are invoked with `!python`, e.g.
  `!python external/Public-available-retinal-OCT-datasets/BOE.py --input_root ... --output_root ...`.
- Model files under `external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/SOTAS/` are meant to
  be `import`ed into notebook cells rather than run standalone.

## Gitignored Paths

`.gitignore` excludes the following — they won't exist after a fresh `git clone` and must be
recreated manually before running the notebooks:

```
data/
├── raw/         # untouched downloads (e.g. Publication_Dataset/, *.zip archives)
└── processed/   # output of src/preprocessing/ + src/notebooks/01_preprocessing.ipynb

output/         # metric CSVs / overlays / checkpoints written by notebooks
                 # (referenced by src/notebooks/02_run_methods.ipynb, 03_results_analysis.ipynb)
```

## Running the experiment (Colab notebooks)

Since this project runs on a Colab server with no local terminal, `notebooks/` are the entry
points — run them in order:

1. `notebooks/00_setup.ipynb` — install deps, verify imports, optionally mount Drive
2. `notebooks/01_preprocessing.ipynb` — raw data → preprocessed PNGs + splits
3. `notebooks/02_run_methods.ipynb` — run one method from `src/methods/` through the shared
   eval harness (`src/eval/run_experiment.py`)
4. `notebooks/03_results_analysis.ipynb` — compare metrics across methods

`src/` holds all self-implemented Python for this experiment — methods from
`implementation_plan.md`'s Methods table (currently stubs, organized by family:
`traditional/`, `model_based/`, `graph_based/`, `region_based/`, `deep_learning/`), plus
`data/`, `preprocessing/`, `postprocessing/`, and the shared `eval/` harness.

### Nested repos (manual step on a new machine)

`external/Retinal_OCT_Image_Segmentation_via_Deep_Learning/` and
`external/Public-available-retinal-OCT-datasets/` are independent git repos (nested `.git` dirs,
own remotes) that are **not** registered as submodules in this repo — they're just checked in as
gitlink entries with no `.gitmodules`. A plain `git clone` of this repo will leave both folders
**empty** on a new machine. After cloning, re-clone them yourself:

```bash
git clone https://github.com/ZhangHH233/Retinal_OCT_Image_Segmentation_via_Deep_Learning.git external/Retinal_OCT_Image_Segmentation_via_Deep_Learning
git clone https://github.com/ZhangHH233/Public-available-retinal-OCT-datasets.git external/Public-available-retinal-OCT-datasets
```

(Pin to the commits currently referenced if you need the exact same state: `ac6d4c5` and `f50b6a3`
respectively — check with `git ls-tree HEAD` in this repo.)

## Citation

```bibtex
@article{ZHANG2025102539,
  title = {Retinal OCT image segmentation with deep learning: A review of advances, datasets, and evaluation metrics},
  journal = {Computerized Medical Imaging and Graphics},
  volume = {123},
  pages = {102539},
  year = {2025},
  issn = {0895-6111},
  doi = {https://doi.org/10.1016/j.compmedimag.2025.102539},
  url = {https://www.sciencedirect.com/science/article/pii/S0895611125000485},
  author = {Huihong Zhang and Bing Yang and Sanqian Li and Xiaoqing Zhang and Xiaoling Li and Tianhang Liu and Risa Higashita and Jiang Liu},
}
```
