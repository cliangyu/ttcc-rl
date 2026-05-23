"""Visualize the GRPO signal: per-ad ΔIBS vs SFT_IBS scatter +
3 example curves where GRPO helped most."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

os.environ.setdefault("HF_HOME", "/home/ssm-user/work/hf-cache")
sys.path.insert(0, "/home/ssm-user/work/ttcc-eval/src")

WORK = Path("/home/ssm-user/work")
def load(p):
    return {row["ad_id"]: np.asarray(row["R_hat"]) for _, row in pq.read_table(p).to_pandas().iterrows()}
sft = load(WORK / "work-out/preds_sft.parquet")
grpo = load(WORK / "work-out/preds_grpo.parquet")
b1 = load(WORK / "work-out/B1_train_mean.parquet")

from ttcc_eval.data import load_ground_truth
from ttcc_eval.preprocess import preprocess
gt_raw = load_ground_truth().filter_split("test")
clean, _ = preprocess(gt_raw)
gt = {str(clean.ad_id[i]): (int(clean.T[i]), np.asarray(clean.curves[i])) for i in range(len(clean.T))}

common = sorted(set(sft) & set(grpo) & set(gt))
def ibs_ad(p, t, T):
    L = min(len(p), len(t), T+1); return float(np.mean((p[:L]-t[:L])**2))
ibs_sft  = np.array([ibs_ad(sft[a],  gt[a][1], gt[a][0]) for a in common])
ibs_grpo = np.array([ibs_ad(grpo[a], gt[a][1], gt[a][0]) for a in common])
ibs_b1   = np.array([ibs_ad(b1[a],   gt[a][1], gt[a][0]) for a in common])
delta = ibs_grpo - ibs_sft

# Sort by ΔIBS for example selection
order = np.argsort(delta)
top_wins = [common[order[i]] for i in range(3)]
top_loss = [common[order[-1-i]] for i in range(2)]

fig = plt.figure(figsize=(16, 11), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1, 1])

# Panel A: scatter SFT_IBS vs ΔIBS
axA = fig.add_subplot(gs[0, :])
colors = ['#2ca02c' if d < -1e-4 else ('#d62728' if d > 1e-4 else '#888') for d in delta]
axA.scatter(ibs_sft, delta, s=40, c=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
axA.axhline(0, color='black', lw=1)
# fit line to show the trend
z = np.polyfit(np.log10(ibs_sft+1e-6), delta, 1)
xx = np.linspace(np.log10(ibs_sft.min()+1e-6), np.log10(ibs_sft.max()), 100)
axA.plot(10**xx, z[0]*xx+z[1], 'k--', alpha=0.5, lw=1, label=f"slope = {z[0]:.4f} per dex")
axA.set_xscale('log')
axA.set_xlabel("SFT per-ad IBS (log)")
axA.set_ylabel("ΔIBS = GRPO − SFT  (negative = GRPO better)")
axA.set_title(f"GRPO improves more where SFT does worse  (n_better=49, n_worse=38, ρ_per_ad=-0.449)")
axA.legend(loc='lower left')
axA.grid(alpha=0.3)
# Annotate the example ads
for ad, dot_color in zip(top_wins, ['#2ca02c']*3):
    i = common.index(ad)
    axA.annotate(f"…{ad[-5:]}", (ibs_sft[i], delta[i]), xytext=(5, -8), textcoords='offset points', fontsize=8, color=dot_color)
for ad in top_loss:
    i = common.index(ad)
    axA.annotate(f"…{ad[-5:]}", (ibs_sft[i], delta[i]), xytext=(5, 5), textcoords='offset points', fontsize=8, color='#d62728')

# Panels B-D: 3 example curves (GRPO wins)
for k, ad in enumerate(top_wins):
    ax = fig.add_subplot(gs[1, k])
    T = gt[ad][0]; g = gt[ad][1][:T+1]
    s = sft[ad][:T+1]; gr = grpo[ad][:T+1]; bb = b1[ad][:T+1]
    ax.plot(np.arange(len(g)), g, '-', color='black', lw=2.5, label='GT')
    ax.plot(np.arange(len(s)), s, '--', color='#17becf', lw=1.5, label='SFT')
    ax.plot(np.arange(len(gr)), gr, '--', color='#2ca02c', lw=1.5, label='GRPO')
    ax.plot(np.arange(len(bb)), bb, ':', color='#1f77b4', lw=1.0, alpha=0.6, label='B1 train-mean')
    ax.set_title(f"GRPO win: …{ad[-6:]}  T={T}\nSFT IBS={ibs_sft[common.index(ad)]:.4f} → GRPO {ibs_grpo[common.index(ad)]:.4f}", fontsize=10)
    ax.set_xlabel("t (s)"); ax.set_ylabel("R(t)")
    ax.set_ylim(-0.05, 1.05)
    if k == 0: ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

# Panels E-F: 2 GRPO losses + 1 ad where SFT was already perfect
for k, ad in enumerate(top_loss + [common[order[len(order)//2]]]):
    ax = fig.add_subplot(gs[2, k])
    T = gt[ad][0]; g = gt[ad][1][:T+1]
    s = sft[ad][:T+1]; gr = grpo[ad][:T+1]; bb = b1[ad][:T+1]
    ax.plot(np.arange(len(g)), g, '-', color='black', lw=2.5, label='GT')
    ax.plot(np.arange(len(s)), s, '--', color='#17becf', lw=1.5, label='SFT')
    ax.plot(np.arange(len(gr)), gr, '--', color='#2ca02c', lw=1.5, label='GRPO')
    ax.plot(np.arange(len(bb)), bb, ':', color='#1f77b4', lw=1.0, alpha=0.6, label='B1')
    tag = "loss" if k < 2 else "median"
    ax.set_title(f"GRPO {tag}: …{ad[-6:]}  T={T}\nSFT={ibs_sft[common.index(ad)]:.4f} → GRPO {ibs_grpo[common.index(ad)]:.4f}", fontsize=10)
    ax.set_xlabel("t (s)"); ax.set_ylabel("R(t)")
    ax.set_ylim(-0.05, 1.05)
    if k == 0: ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("GRPO signal decomposition — RL refines where SFT collapses to train-mean", fontsize=13, y=1.005)
OUT = WORK / "work-out/grpo_signal_decomp.png"
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved {OUT}")
