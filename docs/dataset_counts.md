# Dataset Counts

## DUKE-DME (Chiu et al. 2015)

Location: `data/raw/2015_BOE_Chiu/Subject_01.mat` – `Subject_10.mat`

| | |
| - | - |
| Patients | 10 |
| B-scans per patient | 61 |
| Total B-scans | 610 |
| Annotated B-scans per patient | 11 (centered on fovea) |
| Total annotated B-scans | 110 |
| Boundaries per annotated B-scan | 8 (7 layers) |
| Raters | 2 (`manualLayers1`, `manualLayers2`) |
| Fluid annotation | yes, per-instance (`manualFluid1`, `manualFluid2`) |

Volume shape: `(496, 768, 61)` = (H, W, Z). One volume per patient.

## HC-MS (He et al. 2019)

Location: `data/raw/OCT_Manual_Delineations-2018_June_29/{vol,delineation}/`

| | |
| - | - |
| Patients | 35 (14 HC + 21 MS) |
| B-scans per patient | 49 |
| Total B-scans | 1,715 |
| Annotated B-scans | all 49/patient |
| Boundaries per B-scan | 11 (32 patients); 9 (3 MS patients: `ms14`, `ms16`, `ms17`) |

## Combined

| | Healthy | DME | MS | Total |
| - | - | - | - | - |
| Patients | 14 (31.1%) | 10 (22.2%) | 21 (46.7%) | 45 (100%) |
| B-scans per patient | 49 | 61 | 49 | — |
| Total B-scans | 686 (29.5%) | 610 (26.2%) | 1,029 (44.3%) | 2,325 (100%) |
| Annotated B-scans | 686 (37.6%) | 110 (6.0%) | 1,029 (56.4%) | 1,825 (100%) |
