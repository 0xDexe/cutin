#!/usr/bin/env python3
"""Plot the load sweep: python plot_sweep.py [--dir results/sweep] [--out sweep.png]

Reads <dir>/rps*/<arm>.json, writes the figure and prints markdown tables.
Each turn is scored against its own SLO class (--slo-ms overrides it for every turn).
Turns that hit the TTFT timeout count as misses and as TTFT = timeout in percentiles.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# arm -> (legend name, color, line style). Stock-vLLM baselines are dashed
# (FCFS in neutral gray); the rest take categorical slots 1-4 in fixed order.
ARMS = {
    "fcfs": ("stock vLLM, FCFS", "#8f8e89", "--"),
    "static": ("stock vLLM, static priority", "#2a78d6", "--"),
    "voice-nopin": ("deadlines (EDF)", "#eb6834", "-"),
    "voice": ("+ KV pinning", "#1baf7a", "-"),
    "voice+prefill": ("+ partial-ASR prefill", "#eda100", "-"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def load(path, slo_ms):
    d = json.load(open(path))
    cap = d["args"].get("ttft_timeout_s") or 600
    default = d["args"].get("slo_ms", 300)
    turns = d["turns"]
    t = [r["ttft"] if r["ttft"] is not None else cap for r in turns]
    slo = [slo_ms or r.get("slo_ms", default) for r in turns]
    hit = [x * 1e3 <= s for x, s in zip(t, slo)]
    by_class = {c: 100 * sum(h for h, s in zip(hit, slo) if s == c) / slo.count(c) for c in set(slo)}
    return {
        "slo": 100 * sum(hit) / len(hit),
        "classes": by_class,
        "p50": pct(t, 50) * 1e3,
        "p95": pct(t, 95) * 1e3,
        "batch": d["summary"]["batch_tok_per_s"],
        "cache": d["summary"].get("cached_prompt_pct"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sweep")
    ap.add_argument("--slo-ms", type=int, default=None, help="score every turn against this SLO")
    ap.add_argument("--out", default="sweep.png")
    args = ap.parse_args()

    data = {}  # arm -> {rps: metrics}
    for f in glob.glob(os.path.join(args.dir, "rps*", "*.json")):
        rps = float(re.search(r"rps([\d.]+)", f).group(1))
        arm = os.path.basename(f)[:-5]
        if arm in ARMS:
            data.setdefault(arm, {})[rps] = load(f, args.slo_ms)
    if not data:
        raise SystemExit(f"no results under {args.dir}/rps*/")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    classes = sorted({c for arm in data.values() for m in arm.values() for c in m["classes"]})
    slo_desc = " / ".join(f"{c}" for c in classes) + " ms" + (" classes" if len(classes) > 1 else "")
    panels = [
        ("slo", f"TTFT SLO attainment ({slo_desc})", "% of voice turns", "linear"),
        ("p95", "p95 TTFT", "ms (log scale)", "log"),
        ("batch", "Batch throughput", "output tokens / s", "linear"),
    ]
    for ax, (key, title, ylabel, scale) in zip(axes, panels):
        for arm, (name, color, ls) in ARMS.items():
            if arm not in data:
                continue
            xs = sorted(data[arm])
            ys = [data[arm][x][key] for x in xs]
            ax.plot(xs, ys, ls, color=color, lw=2, marker="o", ms=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=name, zorder=3)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("background batch load (requests / s)")
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)
        if key == "slo":
            ax.set_ylim(0, 105)
        elif key == "batch":
            ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), frameon=False,
               fontsize=10, bbox_to_anchor=(0.01, 1.0))
    fig.text(0.01, 0.01, "Qwen2.5-7B-Instruct, vLLM 0.32.0, 16 voice sessions x 6 turns per point. "
             "Turns with no token after the timeout count as misses (p95 is capped there).",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(args.out, dpi=160)
    print("saved", args.out)

    loads = sorted({r for arm in data.values() for r in arm})
    for key, label, fmt in [("slo", f"SLO attainment ({slo_desc})", "{:.0f}%"),
                            ("p50", "TTFT p50 (ms)", "{:,.0f}"), ("p95", "TTFT p95 (ms)", "{:,.0f}"),
                            ("batch", "batch output tok/s", "{:.0f}")]:
        print(f"\n**{label}**\n")
        print("| batch req/s | " + " | ".join(ARMS[a][0] for a in ARMS if a in data) + " |")
        print("|---" * (1 + len(data)) + "|")
        for r in loads:
            cells = [fmt.format(data[a][r][key]) if r in data[a] else "-" for a in ARMS if a in data]
            print(f"| {r:g} | " + " | ".join(cells) + " |")
    if len(classes) > 1:
        for c in classes:
            print(f"\n**SLO attainment, {c} ms class**\n")
            print("| batch req/s | " + " | ".join(ARMS[a][0] for a in ARMS if a in data) + " |")
            print("|---" * (1 + len(data)) + "|")
            for r in loads:
                cells = [f"{data[a][r]['classes'][c]:.0f}%" if r in data[a] else "-" for a in ARMS if a in data]
                print(f"| {r:g} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
