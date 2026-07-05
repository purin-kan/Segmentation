# scripts/ — local-machine helpers

These run on **your own computer** (macOS / Windows / Linux), **not** on the Colab VM. The
experiment itself runs on Colab via `src/notebooks/*.ipynb`; these scripts are the local side of
the **Google Drive relay** used to get results off the ephemeral Colab runtime and onto your
machine.

## Why a relay?

A Colab runtime is temporary and can't write to your local disk, so results hop through Drive:

```
Colab VM (temporary) ──writes──▶ Google Drive (durable) ──pull + delete──▶ your machine
  notebooks: OUTPUT_ROOT         MyDrive/Segmentation/output   scripts/pull_from_drive.py
```

`00_setup.ipynb`'s Drive cell sets `OUTPUT_ROOT = /content/drive/MyDrive/Segmentation/output`, and
`02_run_methods.ipynb` writes metric CSVs there. These scripts bring them home.

## Files

| Script | Runs on | Purpose |
|--------|---------|---------|
| `setup_rclone.sh`   | macOS / Linux | one-time: install rclone + create the `gdrive` remote |
| `setup_rclone.ps1`  | Windows       | one-time: install rclone + create the `gdrive` remote |
| `pull_from_drive.py`| any (local)   | download `Segmentation/output` → local `output/`, then delete it from Drive |

## Quick start

1. **One-time, per machine** — install rclone and authorize Google Drive:

   ```bash
   scripts/setup_rclone.sh                                          # macOS / Linux
   ```
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_rclone.ps1  # Windows
   ```
   This installs rclone (via Homebrew / apt / winget / scoop / choco) and opens a browser once to
   create an rclone remote named `gdrive` with full `drive` scope — needed because Colab, not
   rclone, created the files (the narrower `drive.file` scope wouldn't see them).

2. **Each time you want results locally:**

   ```bash
   python scripts/pull_from_drive.py            # Segmentation/output -> repo output/, then clears Drive
   python scripts/pull_from_drive.py processed  # a different subfolder under Segmentation/
   python scripts/pull_from_drive.py --yes      # skip the confirmation prompt
   ```

## Safety

`pull_from_drive.py` uses `rclone move`: each file is **copied and checksum-verified before** its
Drive copy is removed, so an interrupted run never loses data. It always shows a **dry run and asks
for confirmation** first (unless `--yes`), and deletions go to **Drive trash** (recoverable ~30
days).

## Path invariant

The path `Segmentation/output` must stay in sync in three places — change one, change the others:

- `00_setup.ipynb` → `OUTPUT_ROOT = /content/drive/MyDrive/Segmentation/output`
- `pull_from_drive.py` → `--drive-base Segmentation` + subpath `output` (the defaults)
- the Drive folder itself
