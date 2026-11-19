saved figures/sweep.png

**SLO attainment (300 ms)**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 86% | 89% | 86% | 90% | 93% |
| 0.5 | 34% | 34% | 35% | 70% | 67% |
| 1 | 5% | 5% | 4% | 30% | 53% |
| 1.5 | 0% | 2% | 2% | 25% | 48% |
| 2 | 0% | 0% | 0% | 24% | 43% |

**TTFT p50 (ms)**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 110 | 114 | 113 | 111 | 97 |
| 0.5 | 404 | 400 | 356 | 214 | 162 |
| 1 | 11,556 | 993 | 1,061 | 687 | 274 |
| 1.5 | 30,000 | 1,100 | 984 | 649 | 355 |
| 2 | 30,000 | 1,024 | 1,058 | 683 | 400 |

**TTFT p95 (ms)**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 448 | 477 | 421 | 434 | 358 |
| 0.5 | 1,185 | 909 | 989 | 527 | 614 |
| 1 | 16,470 | 2,613 | 2,684 | 3,387 | 1,041 |
| 1.5 | 30,000 | 4,690 | 4,741 | 3,737 | 1,455 |
| 2 | 30,000 | 4,437 | 4,435 | 2,696 | 1,614 |

**batch output tok/s**

| batch req/s | stock vLLM, FCFS | stock vLLM, static priority | deadlines (EDF) | + KV pinning | + partial-ASR prefill |
|---|---|---|---|---|---|
| 0.25 | 81 | 82 | 82 | 82 | 80 |
| 0.5 | 128 | 130 | 128 | 137 | 136 |
| 1 | 217 | 195 | 195 | 193 | 194 |
| 1.5 | 269 | 194 | 194 | 197 | 190 |
| 2 | 283 | 194 | 192 | 201 | 192 |
