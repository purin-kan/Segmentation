#!/usr/bin/env python3
"""Pull method outputs from Google Drive down to this local repo, then delete
them from Drive (into Drive trash, recoverable ~30 days).

Runs LOCALLY on your Mac, not on the Colab VM. The VM writes results to Drive
(the setup notebook's OUTPUT_ROOT); this is the "pull back + clean up" half of
that relay.

It drives `rclone` under the hood, so it keeps rclone's verified-move safety:
each file is copied and checksummed BEFORE its Drive copy is removed, so an
interrupted run never loses data.

One-time setup (opens a browser to authorize; run once per machine):
    macOS/Linux:  scripts/setup_rclone.sh
    Windows:      powershell -ExecutionPolicy Bypass -File scripts\setup_rclone.ps1

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


def rclone(*args: str) -> None:
    """Run an rclone command, streaming its output; raise on non-zero exit."""
    subprocess.run(["rclone", *args], check=True)


def remote_exists(remote: str) -> bool:
    out = subprocess.run(
        ["rclone", "listremotes"], check=True, capture_output=True, text=True
    ).stdout
    return f"{remote}:" in out.split()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("subpath", nargs="?", default="output",
                   help="folder under --drive-base to pull (default: output)")
    p.add_argument("--dest", type=Path, default=None,
                   help="local destination (default: <repo>/<subpath>)")
    p.add_argument("--remote", default="gdrive", help="rclone remote name")
    p.add_argument("--drive-base", default="Segmentation",
                   help="base folder on your MyDrive")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()

    if shutil.which("rclone") is None:
        sys.exit("rclone not installed. Run: brew install rclone")
    if not remote_exists(args.remote):
        sys.exit(f"No rclone remote named '{args.remote}:'. Run: rclone config")

    src = f"{args.remote}:{args.drive_base}/{args.subpath}"
    dest = args.dest or (REPO_ROOT / args.subpath)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Source (Drive): {src}")
    print(f"Dest  (local) : {dest}\n")

    print("== DRY RUN (nothing is downloaded or deleted) ==")
    rclone("move", src, str(dest), "--dry-run", "-v", "--delete-empty-src-dirs")

    if not args.yes:
        ans = input("\nProceed: download the above and DELETE from Drive? [y/N] ")
        if ans.strip().lower() != "y":
            print("Aborted — nothing changed.")
            return 0

    print("== MOVING ==")
    # move = copy + verify, THEN delete source. Deletions go to Drive trash.
    rclone("move", src, str(dest), "--progress", "--delete-empty-src-dirs")
    print(f"Done. Files in {dest} and removed from Drive (Drive trash ~30 days).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        sys.exit(f"rclone failed (exit {e.returncode}).")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
