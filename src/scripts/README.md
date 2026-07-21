# src/scripts/ — local-machine helpers

Scripts that run on your own machine, not in a Modal container and not as a notebook step.

| Script | Purpose |
|--------|---------|
| `visualize_boundaries.py` | Overlay boundary annotations onto denoised B-scans, for visual QA |
| `visualize_masks.py`      | Rasterize per-layer masks from boundary labels and save them as viewable label maps |

Both read `data/processed/{duke_dme,hc_ms}_denoised`. Edit the config block at the top of each,
then run it:

```bash
python src/scripts/visualize_boundaries.py
python src/scripts/visualize_masks.py
```

## Moving data to and from Modal

DL training and scoring run on Modal against the `segmentation-data` Volume
(`src/modal_app.py`), so inputs have to be uploaded once and results pulled back down. The
`modal` CLI does both.

```
local data/processed/ ──put──▶ Volume /data/processed ──▶ containers ──▶ Volume /output ──get──▶ local output/
```

Create the Volume once. `modal_app.py` passes `create_if_missing=True`, but that only takes
effect when the app runs; the `modal volume` CLI needs it to exist already:

```bash
modal volume create segmentation-data
```

Seed it with preprocessed data (once, and again whenever preprocessing changes). Only the
denoised directories and `folds.json` are read by `modal_app.py`; the pre-denoise
`duke_dme/` and `hc_ms/` stay local:

```bash
modal volume put segmentation-data data/processed/duke_dme_denoised /data/processed/duke_dme_denoised
modal volume put segmentation-data data/processed/hc_ms_denoised   /data/processed/hc_ms_denoised
modal volume put segmentation-data data/processed/folds.json       /data/processed/folds.json
```

Pull results down:

```bash
modal volume get segmentation-data /output ./output
```

Inspect without downloading:

```bash
modal volume ls segmentation-data /output
```

`modal volume get` copies rather than moves, so the Volume keeps its copy. Remove it explicitly
when you no longer need it:

```bash
modal volume rm segmentation-data /output/<file>
```

## Path invariant

The Volume paths are defined in two places, kept in sync manually:

- `src/modal_app.py`: `DATA_ROOT` / `OUTPUT_ROOT` (`/vol/data`, `/vol/output`)
- `src/paths.py`: `resolve_roots()`, which returns the same pair when `MODAL_TASK_ID` is set

`paths.py` does not import them from `modal_app.py`: that module requires the `modal` package,
which a purely local run does not need.
