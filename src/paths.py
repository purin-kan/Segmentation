"""DATA_ROOT / OUTPUT_ROOT resolution shared by all notebooks.

Call after the repo root is on sys.path (each notebook's first cell does this
itself — that step can't live here, since it's what makes `import src` work).
"""

import os
from pathlib import Path


def resolve_roots() -> tuple[Path, Path]:
    """Return (DATA_ROOT, OUTPUT_ROOT): the mounted Modal Volume inside a
    Modal container, local data/ + output/ folders anywhere else. Creates
    raw/processed/output subfolders if missing.

    Keep the volume paths in sync with modal_app.py's DATA_ROOT / OUTPUT_ROOT.
    Not imported from there: that module needs the modal package, which a
    local run does not require.
    """
    if "MODAL_TASK_ID" in os.environ:
        volume_root = Path("/vol")
        data_root = volume_root / "data"
        output_root = volume_root / "output"
    else:
        data_root = Path("data")
        output_root = Path("output")

    for d in [data_root / "raw", data_root / "processed", output_root]:
        d.mkdir(parents=True, exist_ok=True)

    return data_root, output_root
