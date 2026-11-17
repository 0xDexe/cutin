#!/usr/bin/env python3
"""Voice + batch workload for a vLLM server; reports TTFT and SLO attainment.

    python bench.py --label fcfs                        # stock vLLM
    python bench.py --label voice --deadlines           # voice scheduler
    python bench.py --label voice+prefill --deadlines --prefill
    python bench.py --report results/*.json

Standard library only. TTFT is measured from end of speech (the final transcript)
to the first streamed content token.
"""

import argparse
import http.client
import json
import os
import random
import threading
import time

PREFILL_FLAG = 1_000_000  # must match voice_sched.py
WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his "
    "from at which but have an had they you were their one all we can her has there "
    "been if more when will would who so no time people water city number sound"
).split()


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def post(args, body, on_chunk=None):
    """POST a chat completion. If streaming, on_chunk(chunk) may return False to
    hang up mid-stream, which makes vLLM abort the request (barge-in)."""
    conn = http.client.HTTPConnection(args.host, args.port, timeout=600)
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body),
            {"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
        if not body.get("stream"):
            return json.loads(resp.read())
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if on_chunk(json.loads(line[6:])) is False:
                    return None
    finally:
        conn.close()


def prefill(args, msgs):
    body = {"model": args.model, "messages": msgs, "max_tokens": 1}
    if args.deadlines:
        body["priority"] = PREFILL_FLAG + args.prefill_slack_ms
    try:
        post(args, body)
    except Exception:
        pass


def voice_session(args, sid, records):
    rng = random.Random(sid)
    time.sleep(rng.uniform(0, args.stagger_s))
    msgs = [
        {
            "role": "system",
            "content": f"[session {sid}] You are a concise voice assistant. Notes: "
            + text(rng, args.sys_words),
        }
    ]
    barged = False
    for turn in range(args.turns):
        # User speaks; ASR emits a growing partial transcript word by word.
        words = [rng.choice(WORDS) for _ in range(rng.randint(8, 20))]
        for k in range(1, len(words) + 1):
            time.sleep(args.word_s)
            if args.prefill and k % args.prefill_every == 0 and k < len(words):
                partial = msgs + [{"role": "user", "content": " ".join(words[:k])}]
                threading.Thread(target=prefill, args=(args, partial), daemon=True).start()

        # End of speech: the real request. The clock starts here.
        user = {"role": "user", "content": " ".join(words)}
        body = {
            "model": args.model,
            "messages": msgs + [user],
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        if args.deadlines:
            body["priority"] = args.slo_ms
        rec = {"sid": sid, "turn": turn, "after_barge": barged, "ttft": None,
               "prompt": None, "cached": None, "barged": False, "err": None}
        barge_after = rng.uniform(0.2, 1.0) if rng.random() < args.barge_p else None
        reply, t0, t_first = [], time.perf_counter(), [None]

        def on_chunk(chunk):
            usage = chunk.get("usage") or {}
            if usage.get("prompt_tokens"):
                rec["prompt"] = usage["prompt_tokens"]
                details = usage.get("prompt_tokens_details") or {}
                rec["cached"] = details.get("cached_tokens")
            for choice in chunk.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if t_first[0] is None:
                        t_first[0] = time.perf_counter()
                        rec["ttft"] = t_first[0] - t0
                    reply.append(delta)
            if barge_after and t_first[0] and time.perf_counter() - t_first[0] > barge_after:
                rec["barged"] = True
                return False  # user interrupts: hang up

        try:
            post(args, body, on_chunk)
        except Exception as e:
            rec["err"] = repr(e)
        records.append(rec)
        barged = rec["barged"]
        msgs += [user, {"role": "assistant", "content": "".join(reply) or "..."}]
        if not barged:  # after a barge-in the user is already talking
            time.sleep(rng.uniform(1.0, 3.0))


def batch_load(args, done, stop):
    """Open-loop Poisson arrivals of long-prompt, best-effort requests."""
    rng = random.Random(10**6)

    def one(i):
        r = random.Random(10**6 + i)
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": f"[doc {i}] Summarize: " + text(r, args.batch_words)}],
            "max_tokens": args.batch_max_tokens,
            "ignore_eos": True,
        }
        try:
            out = post(args, body)
            if not stop.is_set():
                done.append(out["usage"]["completion_tokens"])
        except Exception:
            pass

    i = 0
    while not stop.is_set():
        threading.Thread(target=one, args=(i,), daemon=True).start()
        i += 1
        stop.wait(rng.expovariate(args.batch_rps))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def summarize(args, records, batch_done, dur):
    def attain(rs):
        ok = sum(1 for r in rs if r["ttft"] is not None and r["ttft"] * 1e3 <= args.slo_ms)
        return round(100 * ok / len(rs), 1) if rs else None

    ttft = [r["ttft"] * 1e3 for r in records if r["ttft"] is not None]
    with_usage = [r for r in records if r["prompt"] and r["cached"] is not None]
    return {
        "label": args.label,
        "slo_ms": args.slo_ms,
        "turns": len(records),
        "errors": sum(1 for r in records if r["err"]),
        "barge_ins": sum(1 for r in records if r["barged"]),
        "slo_pct": attain(records),
        "slo_pct_first_turn": attain([r for r in records if r["turn"] == 0]),
        "slo_pct_after_barge": attain([r for r in records if r["after_barge"]]),
        "ttft_p50_ms": round(pct(ttft, 50), 1),
        "ttft_p95_ms": round(pct(ttft, 95), 1),
        "ttft_p99_ms": round(pct(ttft, 99), 1),
        "cached_prompt_pct": round(100 * sum(r["cached"] for r in with_usage)
                                   / sum(r["prompt"] for r in with_usage), 1) if with_usage else None,
        "batch_reqs_done": len(batch_done),
        "batch_tok_per_s": round(sum(batch_done) / dur, 1),
        "duration_s": round(dur, 1),
    }


COLS = ["label", "turns", "errors", "barge_ins", "slo_pct", "slo_pct_first_turn",
        "slo_pct_after_barge", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
        "cached_prompt_pct", "batch_reqs_done", "batch_tok_per_s"]


def report(summaries):
    width = max(len(c) for c in COLS)
    for c in COLS:
        print(c.ljust(width), *(str(s.get(c)).rjust(14) for s in summaries))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", nargs="+", help="print a table from result files and exit")
    ap.add_argument("--label", default="run")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--slo-ms", type=int, default=300, help="TTFT SLO for voice turns")
    ap.add_argument("--deadlines", action="store_true", help="send the SLO in `priority` (voice scheduler only)")
    ap.add_argument("--prefill", action="store_true", help="speculative prefill on partial transcripts")
    ap.add_argument("--prefill-every", type=int, default=3, help="words between speculative prefills")
    ap.add_argument("--prefill-slack-ms", type=int, default=1000)
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--stagger-s", type=float, default=10.0)
    ap.add_argument("--sys-words", type=int, default=1500, help="per-session system prompt size")
    ap.add_argument("--word-s", type=float, default=0.3, help="seconds per spoken word")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--barge-p", type=float, default=0.4, help="probability the user interrupts a reply")
    ap.add_argument("--batch-rps", type=float, default=2.0, help="background batch arrivals per second (0 = none)")
    ap.add_argument("--batch-words", type=int, default=3000)
    ap.add_argument("--batch-max-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.report:
        return report([json.load(open(f))["summary"] for f in args.report])

    records, batch_done, stop = [], [], threading.Event()
    t0 = time.perf_counter()
    if args.batch_rps > 0:
        threading.Thread(target=batch_load, args=(args, batch_done, stop), daemon=True).start()
    sessions = [threading.Thread(target=voice_session, args=(args, s, records))
                for s in range(args.sessions)]
    for t in sessions:
        t.start()
    for t in sessions:
        t.join()
    stop.set()
    summary = summarize(args, records, batch_done, time.perf_counter() - t0)

    os.makedirs("results", exist_ok=True)
    path = f"results/{args.label}.json"
    with open(path, "w") as f:
        json.dump({"summary": summary, "args": vars(args), "turns": records}, f, indent=1)
    report([summary])
    print("saved", path)


if __name__ == "__main__":
    main()
