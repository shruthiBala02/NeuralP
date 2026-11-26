#!/usr/bin/env bash
# One-time setup on the cluster: a Python virtual environment with PyTorch built for
# CUDA 12.8 (needed by the Blackwell RTX PRO 6000).
#   bash scripts/setup_environment.sh
# If your cluster uses modules, load a Python >= 3.9 first, e.g.:  module load python/3.11
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
echo
echo "Environment check (run this on a GPU node to see the GPU):"
python -m neuralp.check_environment
