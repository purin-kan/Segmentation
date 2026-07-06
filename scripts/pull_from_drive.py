"""Pull method outputs from Google Drive down to this local repo, then delete
them from Drive (into Drive trash, recoverable 30 days).

Runs locally, not on the Colab VM.

Drives `rclone`: each file is copied and checksummed before its Drive copy is removed.

Run one-time setup before using:
    macOS/Linux:  scripts/setup_rclone.sh
    Windows:      powershell -ExecutionPolicy Bypass -File scripts\\setup_rclone.ps1

Usage:
    python scripts/pull_from_drive.py                 # pull default "output"
    python scripts/pull_from_drive.py processed       # pull another subfolder
    python scripts/pull_from_drive.py output --dest /some/local/dir
    python scripts/pull_from_drive.py --yes           # skip the confirm prompt
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("subpath", nargs="?", default="output", help="folder under --drive-base to pull (default: output)")
    p.add_argument("--dest", type=Path, default=None, help="local destination (default: <repo>/<subpath>)")
    p.add_argument("--remote", default="gdrive", help="rclone remote name")
    p.add_argument("--drive-base", default="Segmentation", help="base folder on your MyDrive")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()

    if shutil.which("rclone") is None:
        sys.exit("rclone not installed. Run: brew install rclone")

    remotes = subprocess.run(
        ["rclone", "listremotes"], check=True, capture_output=True, text=True
    ).stdout.split()
    if f"{args.remote}:" not in remotes:
        sys.exit(
            f"No rclone remote named '{args.remote}:'. Run: rclone config")

    src = f"{args.remote}:{args.drive_base}/{args.subpath}"
    dest = args.dest or (REPO_ROOT / args.subpath)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Source (Drive): {src}")
    print(f"Dest  (local) : {dest}\n")

    move = ["rclone", "move", src, str(dest), "--delete-empty-src-dirs"]

    print("== DRY RUN (nothing is downloaded or deleted) ==")
    subprocess.run([*move, "--dry-run", "-v"], check=True)

    if not args.yes:
        ans = input(
            "\nProceed: download the above and DELETE from Drive? [y/N] ")
        if ans.strip().lower() != "y":
            print("Aborted — nothing changed.")
            return 0

    print("== MOVING ==")
    # move = copy + verify, then delete source. Deletions go to Drive trash.
    subprocess.run([*move, "--progress"], check=True)
    print(
        f"Done. Files in {dest} and removed from Drive (check Drive Trash if needed).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        sys.exit(f"rclone failed (exit {e.returncode}).")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
