#!/bin/bash
# Download experiment results from Google Drive to local machine.
# Requires rclone configured with Google Drive, OR manual download.
#
# Usage:
#   ./scripts/download_results.sh
#
# If rclone is not set up, download manually from:
#   https://drive.google.com/drive/folders/depth_or_length/results/

set -e

LOCAL_DIR="results"
DRIVE_DIR="depth_or_length/results"

mkdir -p "$LOCAL_DIR"

if command -v rclone &> /dev/null; then
    echo "Syncing from Google Drive via rclone..."
    rclone sync "gdrive:$DRIVE_DIR" "$LOCAL_DIR" --progress
    echo "Done. Results in $LOCAL_DIR/"
else
    echo "rclone not found. Install it or download manually."
    echo ""
    echo "To install rclone:"
    echo "  brew install rclone  # macOS"
    echo "  rclone config        # set up Google Drive remote named 'gdrive'"
    echo ""
    echo "Or download manually from Google Drive:"
    echo "  1. Go to drive.google.com"
    echo "  2. Navigate to depth_or_length/results/"
    echo "  3. Download all .jsonl files to results/"
    echo ""
    echo "Then run analysis:"
    echo "  python src/analysis/pilot_analysis.py --results_dir results/"
fi

# List what we have
echo ""
echo "Local results:"
ls -lh "$LOCAL_DIR"/*.jsonl 2>/dev/null || echo "  No JSONL files found yet."
