"""Final comparison viz for SFT vs SFT-noCoT vs SFT-extended vs GRPO-extended.

Reads predictions parquet for each method and renders:
  Panel A: IBS bar (lower=better) with B1 reference line
  Panel B: calibration slope bar (target=1.0)
  Panel C: integrated-retention scatter (true vs predicted) for the best & worst method
  Panel D: paired BCa table (delta IBS vs SFT) as text

Usage:
    python -m scripts.viz.extended \
        --preds work-out/preds_b1.parquet:B1 \
                work-out/preds_sft.parquet:SFT \
                work-out/preds_sft_nocot.parquet:SFT_noCoT \
                work-out/preds_sft_extended.parquet:SFT_ext \
                work-out/preds_grpo_extended.parquet:GRPO_ext \
        --gt work-out/data/ttcc_test.jsonl \
        --out work-out/figs/extended.png
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
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["ad_id"]] = (r["T"], r["R_true"])
    return out


def ibs_per_ad(R_hat: list[float], R_true: list[float], T: int) -> float:
    a = np.array(R_hat[: T + 1])
    b = np.array(R_true[: T + 1])
    return float(np.mean((a - b) ** 2))


def calib_slope(R_hat_all: list[list[float]], R_true_all: list[list[float]]) -> float:
    xs, ys = [], []
    for h, t in zip(R_hat_all, R_true_all):
        n = min(len(h), len(t))
        xs.extend(h[:n])
        ys.extend(t[:n])
    xs, ys = np.array(xs), np.array(ys)
    return float(np.polyfit(xs, ys, 1)[0])


def bca_paired(d: np.ndarray, B: int = 10000, alpha: float = 0.05, rng: int = 0) -> tuple[float, float, float]:
    rs = np.random.default_rng(rng)
    n = len(d)
    boots = np.array([np.mean(d[rs.integers(0, n, n)]) for _ in range(B)])
    z0 = np.percentile(boots <= np.mean(d), 100) / 100
    from scipy.stats import norm
    z0 = norm.ppf(np.mean(boots < np.mean(d)))
    jk = np.array([np.mean(np.delete(d, i)) for i in range(n)])
    jbar = np.mean(jk)
    a = np.sum((jbar - jk) ** 3) / (6 * (np.sum((jbar - jk) ** 2) ** 1.5) + 1e-12)
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    al = norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    au = norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    lo = np.percentile(boots, 100 * al)
    hi = np.percentile(boots, 100 * au)
    return float(np.mean(d)), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="parquet:method pairs, e.g. preds.parquet:SFT")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    methods, paths = [], []
    for spec in args.preds:
        p, m = spec.rsplit(":", 1)
        methods.append(m)
        paths.append(Path(p))

    gt = load_gt(args.gt)
    method_ibs: dict[str, list[tuple[str, float]]] = {}
    method_slope: dict[str, float] = {}
    for m, p in zip(methods, paths):
        preds = load_preds(p)
        rows, hs, ts = [], [], []
        for ad, hat in preds.items():
            if ad not in gt:
                continue
            T, true = gt[ad]
            rows.append((ad, ibs_per_ad(hat, true, T)))
            hs.append(hat[: T + 1])
            ts.append(true[: T + 1])
        method_ibs[m] = rows
        method_slope[m] = calib_slope(hs, ts)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: IBS bar
    ax = axes[0, 0]
    mibs = [(m, np.mean([x[1] for x in method_ibs[m]])) for m in methods]
    ax.bar([m for m, _ in mibs], [v for _, v in mibs], color="steelblue")
    ax.set_ylabel("IBS (lower=better)")
    ax.set_title("Integrated Brier Score")
    if "B1" in [m for m, _ in mibs]:
        b1v = dict(mibs)["B1"]
        ax.axhline(b1v, color="red", linestyle="--", linewidth=1, label=f"B1={b1v:.4f}")
        ax.legend()
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # Panel B: calibration slope
    ax = axes[0, 1]
    ax.bar(methods, [method_slope[m] for m in methods], color="darkorange")
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1, label="ideal=1.0")
    ax.set_ylabel("calibration slope")
    ax.set_title("Calibration slope (regress R_true on R_hat)")
    ax.legend()
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # Panel C: integrated-retention scatter, best & worst method
    ax = axes[1, 0]
    ranks = sorted(mibs, key=lambda x: x[1])
    for m, color in [(ranks[0][0], "tab:green"), (ranks[-1][0], "tab:red")]:
        xs, ys = [], []
        for ad, _ in method_ibs[m]:
            T, true = gt[ad]
            hat = dict(load_preds([p for mm, p in zip(methods, paths) if mm == m][0]))[ad]
            xs.append(np.mean(true[: T + 1]))
            ys.append(np.mean(hat[: T + 1]))
        ax.scatter(xs, ys, alpha=0.6, label=m, color=color, s=20)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("true mean retention (per ad)")
    ax.set_ylabel("predicted mean retention")
    ax.set_title(f"best ({ranks[0][0]}) vs worst ({ranks[-1][0]})")
    ax.legend()

    # Panel D: paired BCa table vs SFT
    ax = axes[1, 1]
    ax.axis("off")
    if "SFT" in methods:
        sft_ibs = {ad: v for ad, v in method_ibs["SFT"]}
        lines = ["paired BCa ΔIBS vs SFT (negative = better than SFT):", ""]
        for m in methods:
            if m == "SFT":
                continue
            mibsd = {ad: v for ad, v in method_ibs[m]}
            ads = sorted(set(sft_ibs) & set(mibsd))
            d = np.array([mibsd[a] - sft_ibs[a] for a in ads])
            mean, lo, hi = bca_paired(d)
            sig = " ✓" if (lo > 0 or hi < 0) else ""
            lines.append(f"  {m:18s}  Δ={mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}")
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, family="monospace",
                fontsize=10, verticalalignment="top")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
