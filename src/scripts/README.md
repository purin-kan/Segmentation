# src/scripts/ — local-machine helpers

Local side of the Google Drive relay. The Colab VM (`src/notebooks/*.ipynb`) is ephemeral and
can't write to local disk, so results pass through Drive; these scripts pull them onto the local
machine and clear the Drive copy.

```
Colab VM ──writes──▶ Google Drive ──pull + delete──▶ local machine
OUTPUT_ROOT           MyDrive/Segmentation/output    src/scripts/pull_from_drive.py
```

## Files

| Script | Platform | Purpose |
|--------|----------|---------|
| `setup_rclone.sh`    | macOS/Linux | One-time rclone install + `gdrive` remote |
| `setup_rclone.ps1`   | Windows     | Same, via PowerShell |
| `pull_from_drive.py` | any         | `Segmentation/output` → local `output/`, deletes the Drive copy |

## Setup

```bash
src/scripts/setup_rclone.sh                                           # macOS/Linux
powershell -ExecutionPolicy Bypass -File src\scripts\setup_rclone.ps1  # Windows
```

Creates an rclone remote named `gdrive` with full `drive` scope — `drive.file` scope can't see
files Colab created rather than rclone itself.

## Usage

```bash
python src/scripts/pull_from_drive.py            # Segmentation/output -> output/
python src/scripts/pull_from_drive.py processed  # other subfolder
python src/scripts/pull_from_drive.py --yes      # skip confirmation
```

`rclone move`: checksum-verifies each file before deleting the Drive copy, dry-runs first, prompts
for confirmation unless `--yes`. Deletions land in Drive trash (~30 days).

## Path invariant

`Segmentation/output` is defined in three places, kept in sync manually:
- `00_setup.ipynb`: `OUTPUT_ROOT`
- `pull_from_drive.py`: `--drive-base` / subpath defaults
- the Drive folder itself
