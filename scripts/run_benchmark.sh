#!/usr/bin/env bash
# Run the timing benchmark directly on a GPU node (interactive session).
#   bash scripts/run_benchmark.sh            # full benchmark (about 30-60 min, estimate)
#   bash scripts/run_benchmark.sh --quick    # 1-minute code check
# Results: outputs/benchmark/<date-time>/summary.md   Log: outputs/logs/
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p outputs/logs
python -m neuralp.benchmark "$@" 2>&1 | tee "outputs/logs/benchmark_$(date +%Y%m%d-%H%M%S).log"
