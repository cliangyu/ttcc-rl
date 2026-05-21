"""Plot GRPO/RLOO reward, IBSReward, and exploration std over training steps.

Parses ms-swift training logs (which emit json-like dicts on each
logging_step line) and produces a multi-panel figure.

Usage:
    python reward_curves.py \\
        --runs label=path/to/grpo.log ... \\
        --out figs/reward_curves.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Pull "key: 'value'" pairs from the dict-style log lines.
KV = re.compile(r"'(\w[\w/]*)': '([^']+)'")


def parse_run(path: Path) -> dict[str, list[float]]:
    """Extract per-step series from a swift training log."""
    series: dict[str, list[float]] = {}
    seen_steps: set[int] = set()
    with open(path) as f:
        for raw in f:
            if "global_step/max_steps" not in raw:
                continue
            kv = dict(KV.findall(raw))
            step_str = kv.get("global_step/max_steps", "")
            try:
                step = int(step_str.split("/")[0])
            except Exception:
                continue
            if step in seen_steps:
                continue
            seen_steps.add(step)
            for k, v in kv.items():
                if k == "global_step/max_steps":
                    continue
                try:
                    series.setdefault(k, []).append(float(v))
                except ValueError:
                    pass
            series.setdefault("_step", []).append(step)
    return series


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs: dict[str, dict[str, list[float]]] = {}
    for spec in args.runs:
        label, p = spec.split("=", 1)
        runs[label] = parse_run(Path(p))
        n = len(runs[label].get("_step", []))
        print(f"  {label}: {n} steps parsed from {p}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Panel A: total reward
    ax = axes[0, 0]
    for label, s in runs.items():
        if "reward" not in s:
            continue
        ax.plot(s["_step"], s["reward"], label=label, marker=".", markersize=3, linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("total reward (max=1.2)")
    ax.set_title("Total reward over training")
    ax.axhline(1.2, color="gray", linestyle=":", linewidth=0.8, label="max=1.2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: IBSReward mean (the policy improvement signal)
    ax = axes[0, 1]
    for label, s in runs.items():
        k = "rewards/TTCCIBSReward/mean"
        if k not in s:
            continue
        ax.plot(s["_step"], s[k], label=label, marker=".", markersize=3, linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("mean IBS-reward  (= 1 − IBS per rollout)")
    ax.set_title("IBSReward mean — policy improvement signal")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="max=1.0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: IBSReward std — exploration / mode-collapse warning
    ax = axes[1, 0]
    for label, s in runs.items():
        k = "rewards/TTCCIBSReward/std"
        if k not in s:
            continue
        ax.plot(s["_step"], s[k], label=label, marker=".", markersize=3, linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("IBS-reward std across rollouts")
    ax.set_title("IBSReward std — exploration (→0 = mode collapse)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel D: completions/clipped_ratio — length-saturation
    ax = axes[1, 1]
    for label, s in runs.items():
        k = "completions/clipped_ratio"
        if k not in s:
            continue
        ax.plot(s["_step"], s[k], label=label, marker=".", markersize=3, linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction of rollouts hitting max_completion_length")
    ax.set_title("Clipped ratio — length saturation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
