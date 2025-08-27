#!/usr/bin/env bash
set -euo pipefail

# On Kaggle, attach the dataset:
#   parthplc/facebook-hateful-meme-dataset
# It will mount under: /kaggle/input/facebook-hateful-meme-dataset

TARGET_DIR="/kaggle/input/facebook-hateful-meme-dataset"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target_dir) TARGET_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
  echo "[Kaggle] Using attached dataset at: $TARGET_DIR"
  echo "Expected files:"
  echo "  $TARGET_DIR/data/train.jsonl"
  echo "  $TARGET_DIR/data/dev.jsonl"
  echo "  $TARGET_DIR/data/img/<images>"
  exit 0
fi

echo "[Local] Download with Kaggle CLI:"
echo "  kaggle datasets download -d parthplc/facebook-hateful-meme-dataset -p ./data"
echo "  unzip -o ./data/*.zip -d ./data"