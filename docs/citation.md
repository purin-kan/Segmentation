# Citations

## References

[Zhang et al. 2025 Survey](https://doi.org/10.1016/j.compmedimag.2025.102539) (accompanying survey paper)

`external/Public-available-retinal-OCT-datasets/BOE.py` is from this survey's companion repo.

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

[Quintana-Quintana et al. 2025 Review](https://doi.org/10.3390/computers14080298) (accompanying review paper)

```bibtex
@article{quintana2025deep,
  title = {Deep Learning Techniques for Retinal Layer Segmentation to Aid Ocular Disease Diagnosis: A Review},
  journal = {Computers},
  volume = {14},
  number = {8},
  pages = {298},
  year = {2025},
  doi = {https://doi.org/10.3390/computers14080298},
  url = {https://www.mdpi.com/2073-431X/14/8/298},
  author = {Oliver Jonathan Quintana-Quintana and Marco Antonio Aceves-Fern{\'a}ndez and Jes{\'u}s Carlos Pedraza-Ortega and Gendry Alfonso-Francia and Saul Tovar-Arriaga},
}
```

---

## Datasets

[HC-MS](https://iacl.ece.jhu.edu/index.php/Resources) (Healthy and Multiple Sclerosis, used for control)

`external/oct_preprocess/Scripts/generate_hc_train.m` (flatten + crop preprocessing, used to
extract this dataset's images/labels) is from He et al.'s companion repo
([heyufan1995/oct_preprocess](https://github.com/heyufan1995/oct_preprocess)).

Y. He, A. Carass, S.D. Solomon, S. Saidha, P.A. Calabresi, and J.L.
Prince, “Retinal layer parcellation of optical coherence tomography
images: Data resource for Multiple Sclerosis and Healthy Controls”,
Data in Brief, 22:601-604, 2019. (DOI: 10.1016/j.dib.2018.12.073)
(PubMed: 30671506)

```bibtex
@article{he2019retinal,
  title={Retinal layer parcellation of optical coherence tomography images: Data resource for multiple sclerosis and healthy controls},
  author={He, Yufan and Carass, Aaron and Solomon, Sharon D and Saidha, Shiv and Calabresi, Peter A and Prince, Jerry L},
  journal={Data in brief},
  volume={22},
  pages={601--604},
  year={2019},
  publisher={Elsevier}
}

@inproceedings{he2019fully,
  title={Fully Convolutional Boundary Regression for Retina OCT Segmentation},
  author={He, Yufan and Carass, Aaron and Liu, Yihao and Jedynak, Bruno M and Solomon, Sharon D and Saidha, Shiv and Calabresi, Peter A and Prince, Jerry L},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={120--128},
  year={2019},
  organization={Springer}
}

@article{he2019deep,
  title={Deep learning based topology guaranteed surface and MME segmentation of multiple sclerosis subjects from retinal OCT},
  author={He, Yufan and Carass, Aaron and Liu, Yihao and Jedynak, Bruno M and Solomon, Sharon D and Saidha, Shiv and Calabresi, Peter A and Prince, Jerry L},
  journal={Biomedical Optics Express},
  volume={10},
  number={10},
  pages={5042--5058},
  year={2019},
  publisher={Optica Publishing Group}
}

@article{he2020structured,
  title={Structured layer surface segmentation for retina OCT using fully convolutional regression networks},
  author={He, Yufan and Carass, Aaron and Liu, Yihao and Jedynak, Bruno M and Solomon, Sharon D and Saidha, Shiv and Calabresi, Peter A and Prince, Jerry L},
  journal={Medical Image Analysis},
  pages={101856},
  year={2020},
  publisher={Elsevier}
}
```

---

[DUKE-DME, Chiu et al. 2015](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm) (DME, used for irregular layers segmentation)

 S. J. Chiu, M. J. Allingham, P. S. Mettu, S. W. Cousins, J. A. Izatt, S. Farsiu, "Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema", ( BIOMEDICAL OPTICS EXPRESS), 6(4), pp. 1172-1194, April, 2015

---

## De-noising

[BM3D](https://webpages.tuni.fi/foi/GCF-BM3D/index.html#ref_problems)
