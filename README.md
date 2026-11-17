# Voice-aware scheduler for vLLM

Real-time voice agents need the first token before a deadline, and callers interrupt
(barge in) constantly. Stock vLLM schedules first-come-first-served and throws away a
request's KV cache once it ends, so under mixed voice and batch load, voice turns queue
behind long batch prompts and every new turn re-prefills the whole conversation.

This project is a drop-in scheduler for vLLM (no fork, loaded with `--scheduler-cls`)
that adds:

- **TTFT-deadline scheduling (EDF).** Each voice request carries its TTFT SLO; the
  scheduler orders requests by absolute deadline and preempts batch work first.
- **KV cache kept across barge-in.** When a voice turn ends, whether finished or
  interrupted, its KV blocks are pinned for a short TTL, so the next turn gets a prefix-cache
  hit instead of being evicted by batch load.
- **Speculative prefill on partial ASR transcripts.** While the user is still speaking,
  the client sends the partial transcript; its KV is computed and pinned, so at end of
  speech only the last few words need prefilling.
- **SLO attainment tracking**, server side (logged) and client side (benchmark).

## Results

Qwen2.5-7B-Instruct, one NVIDIA A40, vLLM 0.32.0, 4 GB KV cache. 16 voice sessions x 6
turns (96 turns, 46 barge-ins), ~2k-token per-session system prompt, plus an open-loop
background load of ~4k-token batch prompts. TTFT is measured from end of speech to the
first streamed token. SLO = 300 ms.

**Arms:** `fcfs` = stock vLLM; `voice-nopin` = deadline scheduling only;
`voice` = deadlines + KV pinning; `voice+prefill` = all three.

### Moderate load (0.5 batch req/s)

| | fcfs | voice-nopin | voice | voice+prefill |
|---|---|---|---|---|
| SLO attainment (300 ms) | 34% | 35% | **68%** | **68%** |
| within 500 ms | 72% | 66% | **94%** | 91% |
| first turns | 62% | 75% | 69% | **81%** |
| turns after a barge-in | 29% | 27% | 63% | **73%** |
| TTFT p50 | 402 ms | 384 ms | 186 ms | **163 ms** |
| TTFT p95 | 1,073 ms | 1,012 ms | **516 ms** | 625 ms |
| TTFT p99 | 1,353 ms | 1,108 ms | **717 ms** | 956 ms |
| prompt tokens from prefix cache | 39% | 36% | 84% | **99%** |
| batch output tok/s | 129 | 130 | 137 | 137 |

### Overload (2 batch req/s)

| | fcfs | voice-nopin | voice | voice+prefill |
|---|---|---|---|---|
| SLO attainment (300 ms) | 0% | 1% | 32% | **48%** |
| within 1,000 ms | 1% | 50% | 70% | **89%** |
| TTFT p50 | 89.6 s | 1.01 s | 650 ms | **365 ms** |
| TTFT p95 | 397 s | 4.4 s | 2.7 s | **1.25 s** |
| prompt tokens from prefix cache | 0% | 0% | 73% | **99%** |
| batch output tok/s | 294 | 194 | 201 | 192 |

### Takeaways

- **KV pinning is the main lever.** At moderate load, deadline ordering alone matches
  FCFS (35% vs 34%); adding pinning doubles SLO attainment and halves p50/p95 TTFT, at
  no cost in batch throughput. Without it, batch traffic evicts most voice prefixes
  between turns.
- **Deadline ordering matters under overload,** where FCFS queueing grows without bound
  (90 s median TTFT) and EDF keeps voice turns near 1 s.
- **Speculative prefill is a trade-off.** It lifts first-turn and post-barge-in
  attainment and gives the lowest median, but the extra requests raise the tail at
  moderate load. Under overload it is a clear win (32% to 48%).

### Caveats

- One run per arm; differences of a few points (especially first-turn, n=16) are noise.
- In the overload run the FCFS queue diverges, so its numbers measure the backlog, not
  a steady state. The voice arms starve batch work (about 0.8 of 2 req/s complete).
- All voice turns share one SLO, so EDF reduces to "voice before batch". Mixed SLOs
  would be needed to isolate deadline-awareness itself.
- Synthetic workload: random-word prompts, fixed-length replies (`ignore_eos`),
  simulated ASR at one word per 0.3 s.

Raw per-turn data: `results/rps0.5/` and `results/rps2.0/`.

## Files

| File | What it is |
|---|---|
| `voice_sched.py` | The scheduler (~130 lines), a subclass of vLLM's scheduler. |
| `bench.py` | Voice sessions with barge-in and partial transcripts, plus batch load. Standard library only. |
| `serve.sh` | Starts vLLM as `baseline` (stock) or `voice`. |
| `run_all.sh` | Runs all four arms in one GPU job (SGE / BU SCC). |
| `scc_env.sh` | Environment for BU SCC (venv, caches, CUDA module). |

## How clients use it

Requests are classified by the OpenAI `priority` field:

| `priority` | Meaning |
|---|---|
| `0` or absent | batch, no deadline |
| `N` (0 < N < 1,000,000) | voice turn, TTFT SLO of N ms |
| `1,000,000 + N` | speculative prefill on a partial transcript (`max_tokens=1`), soft deadline N ms |

Tunables (environment): `VOICE_PIN_TTL_S` (default 15, 0 disables pinning),
`VOICE_PIN_MAX_FRAC` (default 0.3 of the KV pool).

## Reproduce

Any machine with one GPU:

```bash
pip install vllm==0.32.0
./serve.sh baseline   # terminal 1, then in terminal 2:
python bench.py --label fcfs
./serve.sh voice
python bench.py --label voice --deadlines
./serve.sh voice
python bench.py --label voice+prefill --deadlines --prefill
python bench.py --report results/*.json
```

On BU SCC (SGE): edit `WORK` in `scc_env.sh`, set up the venv there, then

```bash
qsub -P <project> run_all.sh                                   # 2 req/s batch load
qsub -P <project> -v BENCH_ARGS="--batch-rps 0.5" run_all.sh   # moderate load
```

FlashInfer needs a CUDA compiler for its JIT kernels: `scc_env.sh` loads a CUDA module
and sets `CUDA_HOME`, and `pip install ninja` is required.

## Limits

Written against vLLM 0.32.0 internals (`_free_request`, block `ref_cnt`,
`block_pool.free_blocks`); other versions may need small changes. Pins expire by TTL
only (no session id). Single KV-cache-group models only. Not production code.
