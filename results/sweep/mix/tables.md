saved figures/sweep_mix.png

**SLO attainment (200 / 800 ms classes)**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 85% | 85% | 86% | 86% | 91% |
| 0.5 | 51% | 51% | 52% | 72% | 77% |
| 1 | 7% | 14% | 15% | 48% | 70% |
| 1.5 | 1% | 7% | 10% | 48% | 62% |
| 2 | 0% | 14% | 14% | 53% | 56% |

**TTFT p50 (ms)**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 117 | 117 | 116 | 113 | 101 |
| 0.5 | 407 | 394 | 405 | 201 | 155 |
| 1 | 12,127 | 1,116 | 1,077 | 606 | 261 |
| 1.5 | 30,000 | 1,103 | 1,056 | 547 | 335 |
| 2 | 30,000 | 1,078 | 1,081 | 534 | 503 |

**TTFT p95 (ms)**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 502 | 468 | 485 | 414 | 329 |
| 0.5 | 1,131 | 1,053 | 958 | 559 | 624 |
| 1 | 16,773 | 3,243 | 3,066 | 2,464 | 1,048 |
| 1.5 | 30,000 | 5,281 | 4,273 | 2,794 | 1,375 |
| 2 | 30,000 | 4,920 | 4,103 | 2,606 | 1,688 |

**batch output tok/s**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 82 | 82 | 82 | 82 | 80 |
| 0.5 | 128 | 127 | 128 | 137 | 137 |
| 1 | 222 | 189 | 192 | 198 | 195 |
| 1.5 | 270 | 197 | 196 | 199 | 192 |
| 2 | 283 | 195 | 195 | 203 | 192 |

**SLO attainment, 200 ms class**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 71% | 71% | 73% | 73% | 81% |
| 0.5 | 21% | 19% | 21% | 44% | 54% |
| 1 | 0% | 0% | 0% | 29% | 48% |
| 1.5 | 0% | 0% | 0% | 33% | 42% |
| 2 | 0% | 0% | 0% | 40% | 48% |

**SLO attainment, 800 ms class**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 100% | 100% | 100% | 100% | 100% |
| 0.5 | 81% | 83% | 83% | 100% | 100% |
| 1 | 15% | 27% | 29% | 67% | 92% |
| 1.5 | 2% | 15% | 21% | 62% | 83% |
| 2 | 0% | 27% | 27% | 67% | 65% |
