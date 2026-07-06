<#
One-time setup (Windows / PowerShell): install rclone and create a Google Drive
remote named "gdrive" so `scripts\pull_from_drive.py` can reach Drive.
Safe to re-run.

Run from the repo root:
    powershell -ExecutionPolicy Bypass -File scripts\setup_rclone.ps1
#>
$ErrorActionPreference = "Stop"
$Remote = if ($env:RCLONE_REMOTE) { $env:RCLONE_REMOTE } else { "gdrive" }

# 1. Install rclone (winget)
if (Get-Command rclone -ErrorAction SilentlyContinue) {
    Write-Host "rclone already installed: $((rclone version)[0])"
} else {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing rclone via winget..."
        winget install --id Rclone.Rclone -e --accept-source-agreements --accept-package-agreements
    } else {
        Write-Host "No winget/scoop/choco found. Install rclone manually: https://rclone.org/install/"
        exit 1
    }
    # A package manager may not refresh PATH for this session.
    if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
        Write-Host "rclone installed, but it's not on PATH yet."
        Write-Host "Open a NEW terminal and re-run this script to finish setup."
        exit 1
    }
}

# 2. Create the Drive remote if it doesn't already exist.
#    This opens a browser to authorize.
if ((rclone listremotes) -contains "${Remote}:") {
    Write-Host "rclone remote '${Remote}:' already exists — nothing to do."
} else {
    Write-Host "Creating rclone remote '${Remote}' for Google Drive."
    rclone config create $Remote drive scope drive
}

Write-Host ""
Write-Host "Done. Verify with:   rclone lsd ${Remote}:"
Write-Host "Pull results with:   python scripts\pull_from_drive.py"
