#!/bin/bash
# Load sweep: one GPU job per batch load, all on the same GPU model, run in parallel.
#   ./sweep.sh <scc_project> [gpu_type]        e.g. ./sweep.sh mlresearch A40
# Results: results/sweep/rps<load>/*.json, then: python plot_sweep.py
set -euo pipefail
cd "$(dirname "$0")"
PROJ=${1:?usage: ./sweep.sh <scc_project> [gpu_type]}
GPU=${2:-A40}
LOADS=${LOADS:-"0.25 0.5 1.0 1.5 2.0"}
mkdir -p logs results/sweep

for r in $LOADS; do
  qsub -P "$PROJ" -N "sweep-$r" -l gpu_type="$GPU" -l h_rt=04:00:00 \
       -v OUT="results/sweep/rps$r",BENCH_ARGS="--batch-rps $r --ttft-timeout-s 30" \
       run_all.sh
done
