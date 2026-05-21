"""Visualize ground-truth vs SFT vs GRPO predictions on diverse example ads.

Picks one ad from each novelty bucket (Q1 = closest to B1, Q2+Q3 = moderate,
Q4 = farthest) plus a few extras chosen by IBS gap GRPO-vs-SFT to highlight
where RL helps and where it hurts. Overlays curves with per-ad IBS labels.

Usage:
    python example_curves.py \\
        --preds preds_sft.parquet:SFT preds_grpo.parquet:GRPO ... \\
        --b1-preds preds_b1.parquet \\
        --gt /home/ssm-user/work/data/ttcc_swift/ttcc_test.jsonl \\
        --out figs/example_curves.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def load_preds(path: Path) -> dict[str, list[float]]:
    t = pq.read_table(path)
    return dict(zip(t["ad_id"].to_pylist(), t["R_hat"].to_pylist()))


def load_gt(path: Path) -> dict[str, tuple[int, list[float]]]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["ad_id"]] = (r["T"], r["R_true"])
    return out


def ibs(hat: list[float], true: list[float], T: int) -> float:
    a = np.array(hat[: T + 1])
    b = np.array(true[: T + 1])
    return float(np.mean((a - b) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--b1-preds", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    method_preds: dict[str, dict] = {}
    for spec in args.preds:
        p, m = spec.rsplit(":", 1)
        method_preds[m] = load_preds(Path(p))
    b1 = load_preds(args.b1_preds)
    gt = load_gt(args.gt)
    methods = list(method_preds.keys())

    common = set(gt) & set(b1)
    for m in methods:
        common &= set(method_preds[m])
    ads = sorted(common)

    # Per-ad metrics for selection
    nov: dict[str, float] = {}
    ibs_per: dict[str, dict[str, float]] = {m: {} for m in methods}
    ibs_per["B1"] = {}
    for ad in ads:
        T, true = gt[ad]
        nov[ad] = ibs(b1[ad], true, T)
        ibs_per["B1"][ad] = nov[ad]
        for m in methods:
            ibs_per[m][ad] = ibs(method_preds[m][ad], true, T)

    nov_arr = np.array([nov[a] for a in ads])
    q1, q2, q3 = np.quantile(nov_arr, [0.25, 0.5, 0.75])

    # Pick 6 ads: 1 each from Q1/Q2/Q3/Q4 by median novelty within bucket,
    # plus the ad where GRPO most beats SFT, and where GRPO most loses to SFT.
    def bucket(ad: str) -> str:
        v = nov[ad]
        if v <= q1: return "Q1"
        if v <= q2: return "Q2"
        if v <= q3: return "Q3"
        return "Q4"

    by_bucket: dict[str, list[str]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for ad in ads:
        by_bucket[bucket(ad)].append(ad)
    # Sort each bucket by novelty and pick the median ad
    pick = {}
    for b, lst in by_bucket.items():
        lst.sort(key=lambda a: nov[a])
        pick[b] = lst[len(lst) // 2]

    # Extras: largest GRPO win and loss vs SFT
    if "SFT" in methods and "GRPO" in methods:
        diff = {a: ibs_per["GRPO"][a] - ibs_per["SFT"][a] for a in ads}
        pick["RL_win"]  = min(diff, key=diff.get)
        pick["RL_lose"] = max(diff, key=diff.get)

    chosen = list(pick.items())
    n = len(chosen)
    cols = 3
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(cols * 5, rows_ * 3.5))
    axes = axes.flatten()

    colors = {"GT": "black", "B1": "tab:gray", "SFT": "tab:orange", "GRPO": "tab:blue"}

    for ax_idx, (tag, ad) in enumerate(chosen):
        ax = axes[ax_idx]
        T, true = gt[ad]
        ts = np.arange(T + 1)
        ax.plot(ts, true,        color=colors["GT"], linewidth=2.0, label=f"GT")
        ax.plot(ts, b1[ad][:T+1], color=colors["B1"], linewidth=1.2, linestyle="--",
                label=f"B1 (IBS={ibs_per['B1'][ad]:.4f})")
        for m in methods:
            color = colors.get(m, "tab:green")
            ax.plot(ts, method_preds[m][ad][:T+1], color=color, linewidth=1.6,
                    label=f"{m} (IBS={ibs_per[m][ad]:.4f})")
        ax.set_xlabel("second t")
        ax.set_ylabel("R(t)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{tag}: ad …{ad[-6:]}  T={T}s  ν={nov[ad]:.4f}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    # hide any unused panels
    for i in range(len(chosen), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
