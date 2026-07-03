# Retinal OCT Image Segmentation

This repository is a working sandbox for comparing traditional, graph-based, and deep-learning
methods for retinal layer/lesion segmentation in OCT images, built on top of the reference
implementations collected in the accompanying survey:

> Zhang et al., "Retinal OCT image segmentation with deep learning: A review of advances,
> datasets, and evaluation metrics," *Computerized Medical Imaging and Graphics*, 2025.
> DOI: [10.1016/j.compmedimag.2025.102539](https://doi.org/10.1016/j.compmedimag.2025.102539)

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

`venv/` and `data/` are gitignored — recreate the environment and re-download/place datasets on each machine.

### Nested repos (manual step on a new machine)

`Retinal_OCT_Image_Segmentation_via_Deep_Learning/` and `Public-available-retinal-OCT-datasets/` are
independent git repos (nested `.git` dirs, own remotes) that are **not** registered as submodules in
this repo — they're just checked in as gitlink entries with no `.gitmodules`. A plain `git clone` of
this repo will leave both folders **empty** on a new machine. After cloning, re-clone them yourself:

```bash
git clone https://github.com/ZhangHH233/Retinal_OCT_Image_Segmentation_via_Deep_Learning.git
git clone https://github.com/ZhangHH233/Public-available-retinal-OCT-datasets.git
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
