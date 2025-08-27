#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
# Kaggle already has torch/torchvision; we (re)install open_clip_torch explicitly.
pip install -r requirements.txt