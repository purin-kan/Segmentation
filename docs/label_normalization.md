# Label harmonization: HC-MS + DUKE-DME

The decision and the scheme live in `implementation_plan.md` (Setup > label harmonization); the
union boundary table and its idx 1-10 numbering are there too. This doc holds only the
justification for dropping the outer retina, the one known risk, and the data-quality findings.

## Why the outer retina is dropped

Duke brackets the outer-retinal bright band (band 2, the ellipsoid) with **two** lines:
`ONL-ISM/ISE` at its inner edge (idx 7), `ISE/OS-RPE` at its outer edge (idx 8). HC-MS crosses
it with **one**, `IS-OS`. Which of Duke's two it matches is not established, and Lang 2013
supports both readings equally:

- **A, anatomical.** Lang's `IS` layer runs ELM to `IS-OS`. The inner segment is myoid plus
  ellipsoid, so `IS-OS` sits below the ellipsoid, matching Duke's idx 8.
- **B, visible bands.** Lang's four outer boundaries (ELM, `IS-OS`, `OS-RPE`, BrM) map 1:1 onto
  the four visible hyper-reflective bands. A rater clicking what is visible puts `IS-OS` on
  band 2 itself, matching Duke's idx 7. Off by one.

Neither paper states which band a click lands on. Naming history favours B: the 2014 IN·OCT
consensus renamed band 2 to the ellipsoid zone precisely because the legacy name "IS/OS
junction" was wrong, and Lang 2013 predates the consensus and uses the legacy name. Layer naming
favours A. That Duke needs two lines where HC-MS uses one is itself evidence the protocols
differ here, whichever reading holds.

The candidates sit roughly 5 px (20 µm) apart, so pairing the wrong two trains the model on
contradictory targets. The scheme drops every boundary in this region from both datasets, so the
question needs no answer.

## The one soft spot: OPL/ONL (boundary 5)

Not the same species as boundary 8. There the protocols name different structures; here both
name the identical interface (`OPL-ONL` and `OPL / ONL-ISM`), so the risk is execution, not
definition. The Henle fiber layer sits at it and is anatomically part of OPL but reads as ONL on
OCT. Neither paper says which the rater traced. Both used on-axis Spectralis where HFL is not
separable, so both likely traced the visible transition and agree. If it does bite, layer 4
(OPL) carries a systematic cross-dataset thickness offset.

To detect it: train on one dataset, test on the other, and look at **signed** bias at boundary 5
with boundaries 1-4 as controls. A protocol offset shows as systematic signed error where the
controls show none. Cross-dataset layer thickness cannot be used for this: Duke is DME
(thickened) and HC-MS is healthy plus MS, so thickness is confounded by pathology.

## Alternative considered: masked union (rejected)

Predict all 10 union boundaries; compute loss only where the source dataset annotated, NaN
elsewhere. Established practice (partially-supervised segmentation), not novel:

- Shi et al., "Marginal loss and exclusion loss for partially supervised multi-organ
  segmentation," *Medical Image Analysis* 2021, arXiv:2007.03868. Marginal/exclusion loss
  addresses semantic segmentation; for boundary regression, masking is just skipping NaN targets.
- "General retinal layer segmentation in OCT via reinforcement constraint," *MedIA* 2024.
  Applies this to OCT layers at differing granularity, HC-MS included.

Rejected because harmonization is not the experiment's focus, and it does not dissolve boundary
8: that pair still has to be resolved or split into two dataset-specific indices.

## Data quality

- **HC-MS raw indices 3 and 11 are never delineated**: empty in 1568/1568 B-scans across all 32
  control-point files. `Preprocess.m`'s `bd_pts(:,:,[3,11])=[]` drops exactly those, leaving
  Lang's 9. The GCL/IPL split is absent by protocol, so the GCL+IPL merge with Duke is exact
  rather than a coarsening.
- **Boundary row order is inner to outer** in both datasets: monotonic across the full data
  (31 decreasing steps in 14,049,280 for HC-MS, 0 in 399,147 for Duke).
- **`ms14`, `ms16`, `ms17`** ship dense `bd_pts` (1024, 49, 9) instead of sparse `control_pts`,
  and `ms14`/`ms16` hold all 31 of the ordering violations. Treat as slightly lower quality.
- **`dmeheader.mat` latent bug, not fixed**: a canned Spectralis header (`SizeX=1024`,
  `NumBScans=49`) reused for Duke, whose real geometry is 768x496x61. `Preprocess.m` overwrites
  `SizeX`/`SizeZ`/`NumBScans` from the data but not `ScaleX`/`ScaleZ`. `ScaleZ=3.87` is correct;
  `ScaleX=5.64 µm` is the 1024-A-scan value and understates Duke's true lateral spacing, so the
  retina detector's lateral smoothing sigmas run about 35% wide, mildly affecting Duke's
  flattening. Labels stay in register, so ground truth is unaffected.
