#!/usr/bin/env bash
# usage: ./serve.sh baseline | priority | voice
#   baseline = stock FCFS, priority = stock vLLM priority policy, voice = this scheduler
# Env: MODEL, KV_BYTES (KV cache size; keep it small enough that the batch load
# actually evicts cached prefixes, otherwise pinning has nothing to protect),
# VOICE_PIN_TTL_S (0 = scheduler without KV pinning), VOICE_PIN_MAX_FRAC.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
COMMON=(--port "${PORT:-8000}" --max-model-len 16384 --enable-prompt-tokens-details
        --kv-cache-memory-bytes "${KV_BYTES:-4000000000}")

case "${1:-}" in
  baseline) exec vllm serve "$MODEL" "${COMMON[@]}" ;;
  priority) exec vllm serve "$MODEL" "${COMMON[@]}" --scheduling-policy priority ;;
  voice)    PYTHONPATH=. exec vllm serve "$MODEL" "${COMMON[@]}" \
              --scheduling-policy priority --scheduler-cls voice_sched.VoiceScheduler ;;
  *) echo "usage: $0 baseline|priority|voice" >&2; exit 1 ;;
esac
