#!/usr/bin/env bash
# One-time setup (macOS / Linux): install rclone and create a Google Drive
# remote named "gdrive" so `scripts/pull_from_drive.py` can reach your Drive.
# Safe to re-run — it skips any step that's already done.
set -euo pipefail

REMOTE="${RCLONE_REMOTE:-gdrive}"

# 1. Install rclone if it's missing.
if command -v rclone >/dev/null 2>&1; then
  echo "rclone already installed: $(rclone version | head -1)"
elif command -v brew >/dev/null 2>&1; then
  echo "Installing rclone via Homebrew..."
  brew install rclone
elif command -v apt-get >/dev/null 2>&1; then
  echo "Installing rclone via apt..."
  sudo apt-get update && sudo apt-get install -y rclone
else
  echo "Couldn't find Homebrew or apt. Install rclone manually: https://rclone.org/install/"
  echo "(macOS: install Homebrew from https://brew.sh first, then re-run this script.)"
  exit 1
fi

# 2. Create the Drive remote if it doesn't already exist.
#    This opens a browser to authorize — sign in and click Allow.
#    Scope "drive" (full) is required: the files were created by Colab's Drive
#    mount, not by rclone, so the narrower "drive.file" scope wouldn't see them.
if rclone listremotes | grep -qx "${REMOTE}:"; then
  echo "rclone remote '${REMOTE}:' already exists — nothing to do."
else
  echo "Creating rclone remote '${REMOTE}' for Google Drive."
  echo "A browser window will open — sign in and click Allow."
  rclone config create "${REMOTE}" drive scope drive
fi

echo
echo "Done. Verify with:   rclone lsd ${REMOTE}:"
echo "Pull results with:   python scripts/pull_from_drive.py"
