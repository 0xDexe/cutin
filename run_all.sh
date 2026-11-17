#!/bin/bash -l
# BU SCC batch job: runs all benchmark arms on one GPU, one after another.
#   qsub -P <your_project> run_all.sh
#$ -N voice-sched
#$ -l h_rt=03:00:00
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

set -uo pipefail
cd "${SGE_O_WORKDIR:-$(dirname "$0")}"
source ./scc_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

export PORT=$(shuf -i 20000-40000 -n 1)   # avoid clashing with other jobs on the node
mkdir -p logs results

run_arm() {  # run_arm <label> <serve mode> [extra env] -- [bench args...]
  local label=$1 mode=$2; shift 2
  local envs=(); while [[ $# -gt 0 && $1 != -- ]]; do envs+=("$1"); shift; done; shift
  echo "=== $label ==="
  env "${envs[@]}" ./serve.sh "$mode" > "logs/server_$label.log" 2>&1 &
  local pid=$!
  for _ in $(seq 600); do
    curl -sf "localhost:$PORT/health" >/dev/null && break
    kill -0 $pid 2>/dev/null || { echo "server died; see logs/server_$label.log"; tail -30 "logs/server_$label.log"; return 1; }
    sleep 1
  done
  python bench.py --port "$PORT" --model "$MODEL" --label "$label" "$@" ${BENCH_ARGS:-}
  kill $pid; wait $pid 2>/dev/null
  grep "voice SLO attainment" "logs/server_$label.log" | tail -1
}

run_arm fcfs          baseline                   --
run_arm voice         voice                      -- --deadlines
run_arm voice+prefill voice                      -- --deadlines --prefill
run_arm voice-nopin   voice VOICE_PIN_TTL_S=0    -- --deadlines   # ablation: no KV pinning

python bench.py --report results/fcfs.json results/voice.json results/voice+prefill.json results/voice-nopin.json \
  | tee results/summary.txt