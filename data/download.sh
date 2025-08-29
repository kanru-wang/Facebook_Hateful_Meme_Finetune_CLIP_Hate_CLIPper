#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="./data"

# Parses command-line arguments:
# --target_dir <path>: Allow overriding the default TARGET_DIR value.
# Any other argument triggers an error and exits.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target_dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

echo "[Local] Downloading dataset to: $TARGET_DIR"

# Use Kaggle CLI to download data
kaggle datasets download -d parthplc/facebook-hateful-meme-dataset -p "$TARGET_DIR"
# Unzip it into the local ./data directory for use
unzip -o "$TARGET_DIR"/*.zip -d "$TARGET_DIR"
