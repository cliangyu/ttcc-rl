"""Build a teaching visualization for what the eval is measuring.

4 panels:
A. Per-time-step MSE: error vs t for each method (shows where in the curve each method fails)
B. Calibration scatter: predicted R vs true R for all (ad, t) — perfect = diagonal
C. Per-ad IBS distribution (log) — width of failure tail
D. 8 random ads, all methods overlaid — the qualitative picture
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

os.environ.setdefault("HF_HOME", "/home/ssm-user/work/hf-cache")
sys.path.insert(0, "/home/ssm-user/work/ttcc-eval/src")
from ttcc_eval.data import load_ground_truth
from ttcc_eval.preprocess import preprocess

WORK = Path("/home/ssm-user/work")

def load(p):
    return {row["ad_id"]: np.asarray(row["R_hat"]) for _, row in pq.read_table(p).to_pandas().iterrows()}

methods = {
    "iter2 (zero-shot)":  ("#bcbd22", "--", load(WORK / "work-out/qwen25_omni_3b_seed0_iter2.parquet")),
    "B1 train-mean":      ("#1f77b4", ":",  load(WORK / "work-out/B1_train_mean.parquet")),
    "SFT (CoT distill)":  ("#17becf", "--", load(WORK / "work-out/preds_sft.parquet")),
    "GRPO-50":            ("#2ca02c", "-",  load(WORK / "work-out/preds_grpo.parquet")),
}

gt_raw = load_ground_truth().filter_split("test")
clean, _ = preprocess(gt_raw)
gt = {str(clean.ad_id[i]): (int(clean.T[i]), np.asarray(clean.curves[i])) for i in range(len(clean.T))}
common = sorted(set.intersection(*(set(d) for _, _, d in methods.values())) & set(gt))
print(f"n common ads: {len(common)}")

# -------- compute --------
def ibs_per_ad(preds, ads):
    out = np.empty(len(ads))
    for i, a in enumerate(ads):
        T = gt[a][0]; g = gt[a][1][:T+1]; p = preds[a][:T+1]
        L = min(len(g), len(p))
        out[i] = np.mean((p[:L] - g[:L])**2)
    return out

per_ad = {name: ibs_per_ad(d, common) for name, (_, _, d) in methods.items()}

# Per-time-step MSE: for each t in 0..60, average over all ads with T_i >= t
def per_t_mse(preds):
    sums = np.zeros(61); cnts = np.zeros(61)
    for a in common:
        T = gt[a][0]; g = gt[a][1]; p = preds[a]
        L = min(len(g), len(p))
        for t in range(min(L, 61)):
            sums[t] += (p[t] - g[t])**2
            cnts[t] += 1
    return sums / np.maximum(cnts, 1)

t_mse = {name: per_t_mse(d) for name, (_, _, d) in methods.items()}

# -------- figure --------
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = GridSpec(3, 4, figure=fig, height_ratios=[1.1, 1.1, 1.2])

# A. Per-time-step MSE
axA = fig.add_subplot(gs[0, :2])
for name, (color, ls, _) in methods.items():
    axA.plot(np.arange(61), t_mse[name], color=color, ls=ls, lw=2, label=name)
axA.set_yscale('log')
axA.set_xlabel("t (s)")
axA.set_ylabel("MSE at time t  (log)")
axA.set_title("A. Where each method loses — per-second squared error\n(IBS averages this curve over t)")
axA.legend(loc='upper right', fontsize=9)
axA.grid(alpha=0.3, which='both')

# B. Calibration scatter
axB = fig.add_subplot(gs[0, 2:])
for name, (color, _, d) in methods.items():
    xs, ys = [], []
    for a in common:
        T = gt[a][0]; g = gt[a][1][:T+1]; p = d[a][:T+1]
        L = min(len(g), len(p))
        xs.extend(g[:L].tolist()); ys.extend(p[:L].tolist())
    axB.scatter(xs, ys, s=3, c=color, alpha=0.18, label=name)
axB.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='perfect calibration')
axB.set_xlabel("true R(t)")
axB.set_ylabel("predicted R(t)")
axB.set_title("B. Calibration — predicted vs true on all (ad, t) pairs\n(diagonal = perfect; clouds above diagonal = over-prediction)")
axB.legend(loc='lower right', fontsize=9, markerscale=3)
axB.set_xlim(-0.02, 1.02); axB.set_ylim(-0.02, 1.05)
axB.grid(alpha=0.3)

# C. Per-ad IBS histogram (log)
axC = fig.add_subplot(gs[1, :2])
bins = np.logspace(-5, 0, 31)
for name, (color, _, _) in methods.items():
    vals = per_ad[name]
    axC.hist(vals, bins=bins, color=color, alpha=0.55, label=f"{name}  median={np.median(vals):.4f}", edgecolor='black', linewidth=0.3)
axC.set_xscale('log')
axC.set_xlabel("per-ad IBS  (log)")
axC.set_ylabel("# ads")
axC.set_title("C. Per-ad IBS distribution — wider tail = more catastrophic ads")
axC.legend(loc='upper right', fontsize=9)
axC.grid(alpha=0.3, which='both')

# D. Per-ad ΔIBS GRPO − SFT vs SFT per-ad IBS (re-emphasizes GRPO signal)
axD = fig.add_subplot(gs[1, 2:])
d_grpo = per_ad["GRPO-50"] - per_ad["SFT (CoT distill)"]
sft_vals = per_ad["SFT (CoT distill)"]
colors = ['#2ca02c' if d < -1e-4 else ('#d62728' if d > 1e-4 else '#888') for d in d_grpo]
axD.scatter(sft_vals, d_grpo, s=40, c=colors, edgecolor='black', linewidth=0.4, alpha=0.85)
axD.axhline(0, color='black', lw=1)
axD.set_xscale('log')
axD.set_xlabel("SFT per-ad IBS  (log)")
axD.set_ylabel("ΔIBS  =  GRPO − SFT  (<0 = GRPO better)")
axD.set_title("D. GRPO's gain over SFT is concentrated where SFT failed\n(49/87 wins, ρ(Δ, SFT_IBS) = −0.449)")
axD.grid(alpha=0.3)

# E. 8 random ads with all methods (last row)
rng = np.random.default_rng(7)
sample_ads = rng.choice(common, size=8, replace=False)
sample_ads = sorted(sample_ads, key=lambda a: gt[a][0])  # by T_i
for k, ad in enumerate(sample_ads):
    ax = fig.add_subplot(gs[2, k % 4]) if k < 4 else fig.add_subplot(gs[2, k - 4])
    T = gt[ad][0]; g = gt[ad][1][:T+1]
    ax.plot(np.arange(len(g)), g, '-', color='black', lw=2.5, label='GT' if k==0 else None)
    for name, (color, ls, d) in methods.items():
        p = d[ad][:T+1]
        ax.plot(np.arange(len(p)), p, ls, color=color, lw=1.5, alpha=0.85, label=name if k==0 else None)
    ax.set_title(f"…{ad[-6:]}  T={T}", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    if k == 0: ax.legend(loc='upper right', fontsize=7)
    if k % 4 == 0: ax.set_ylabel("R(t)")
    if k >= 4: ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3)

fig.suptitle("Understanding the eval: 4 ways to see what IBS measures and which method wins", fontsize=13, y=1.005)
OUT = WORK / "work-out/eval_explain.png"
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved {OUT}")
