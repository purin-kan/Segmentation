#!/usr/bin/env bash
# One-time setup (macOS/Linux): install rclone and create a Google Drive
set -euo pipefail

REMOTE="${RCLONE_REMOTE:-gdrive}"

# 1. Install rclone if missing.
if command -v rclone >/dev/null 2>&1; then
  echo "rclone already installed: $(rclone version | head -1)"
elif command -v brew >/dev/null 2>&1; then
  echo "Installing rclone via Homebrew..."
  brew install rclone
elif command -v apt-get >/dev/null 2>&1; then
  echo "Installing rclone via apt..."
  sudo apt-get update && sudo apt-get install -y rclone
else
  echo "Couldn't find Homebrew"
  exit 1
fi

# 2. Create the Drive remote if it doesn't already exist.
if rclone listremotes | grep -qx "${REMOTE}:"; then
  echo "rclone remote '${REMOTE}:' already exists"
else
  echo "Creating rclone remote '${REMOTE}' for Google Drive."
  rclone config create "${REMOTE}" drive scope drive
fi

echo
echo "Done. Verify with:   rclone lsd ${REMOTE}:"
echo "Pull results with:   python src/scripts/pull_from_drive.py"
