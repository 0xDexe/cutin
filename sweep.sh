#!/bin/bash
# Load sweep: one GPU job per batch load, all on the same GPU model, run in parallel.
#   ./sweep.sh <scc_project> [gpu_type]                 e.g. ./sweep.sh mlresearch A40
# Optional env:
#   LOADS="0.25 0.5 1.0 1.5 2.0"   batch loads (requests/s)
#   ARMS="static"                  run only some arms (default: all five)
#   MIX=200,800                    two caller classes with different TTFT SLOs
#   TAG=mix                        results go to results/sweep/<TAG>/rps<load>/
# Plot: python plot_sweep.py --dir results/sweep[/<TAG>]
set -euo pipefail
cd "$(dirname "$0")"
PROJ=${1:?usage: ./sweep.sh <scc_project> [gpu_type]}
GPU=${2:-A40}
LOADS=${LOADS:-"0.25 0.5 1.0 1.5 2.0"}
BASE=results/sweep${TAG:+/$TAG}
EXTRA="--ttft-timeout-s 30${MIX:+ --slo-mix ${MIX//,//}}"   # qsub -v splits on commas
mkdir -p logs "$BASE"

for r in $LOADS; do
  VARS="OUT=$BASE/rps$r,BENCH_ARGS=--batch-rps $r $EXTRA"
  [[ -n ${ARMS:-} ]] && VARS="$VARS,ARMS=$ARMS"
  qsub -P "$PROJ" -N "sweep${TAG:+-$TAG}-$r" -l gpu_type="$GPU" -l h_rt=04:00:00 -v "$VARS" run_all.sh
done
