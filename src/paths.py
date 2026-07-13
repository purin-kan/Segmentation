"""DATA_ROOT / OUTPUT_ROOT resolution shared by all notebooks.

Call after the repo root is on sys.path (each notebook's first cell does this
itself — that step can't live here, since it's what makes `import src` work).
"""

from pathlib import Path


def resolve_roots() -> tuple[Path, Path]:
    """Mount Drive and return (DATA_ROOT, OUTPUT_ROOT) under it on Colab,
    or local data/ + output/ folders otherwise. Creates raw/processed/output
    subfolders if missing.

    Keep the 'Segmentation/output' Drive path in sync with
    scripts/pull_from_drive.py's --drive-base / subpath.
    """
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        drive_root = Path("/content/drive/MyDrive/Segmentation")
        data_root = drive_root / "data"
        output_root = drive_root / "output"
    except ImportError:
        data_root = Path("data")
        output_root = Path("output")

    for d in [data_root / "raw", data_root / "processed", output_root]:
        d.mkdir(parents=True, exist_ok=True)

    return data_root, output_root
