"""Decompose GRPO's IBS gain over SFT:
   1. per-ad ΔIBS distribution (did all 87 ads improve, or just a tail?)
   2. correlation of ΔIBS with T_i and GT R(T_i) — which ads benefit?
   3. sample 5 completions side-by-side SFT vs GRPO to inspect content
   4. SFT IBS at different effective temperatures (eval temp=0 vs GRPO rollout temp=0.4)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

os.environ.setdefault("HF_HOME", "/home/ssm-user/work/hf-cache")
sys.path.insert(0, "/home/ssm-user/work/ttcc-eval/src")

WORK = Path("/home/ssm-user/work")

# Load preds
def load(p):
    return {row["ad_id"]: np.asarray(row["R_hat"]) for _, row in pq.read_table(p).to_pandas().iterrows()}
sft = load(WORK / "work-out/preds_sft.parquet")
grpo = load(WORK / "work-out/preds_grpo.parquet")
rloo = load(WORK / "work-out/preds_rloo.parquet")
b1 = load(WORK / "work-out/B1_train_mean.parquet")

# Load GT
from ttcc_eval.data import load_ground_truth
from ttcc_eval.preprocess import preprocess
gt_raw = load_ground_truth().filter_split("test")
clean, _ = preprocess(gt_raw)
gt = {str(clean.ad_id[i]): (int(clean.T[i]), np.asarray(clean.curves[i])) for i in range(len(clean.T))}

common = sorted(set(sft) & set(grpo) & set(rloo) & set(gt))
print(f"n common ads = {len(common)}")

def ibs_ad(pred, true, T):
    L = min(len(pred), len(true), T+1)
    return float(np.mean((pred[:L]-true[:L])**2))

ibs_sft  = np.array([ibs_ad(sft[a],  gt[a][1], gt[a][0]) for a in common])
ibs_grpo = np.array([ibs_ad(grpo[a], gt[a][1], gt[a][0]) for a in common])
ibs_rloo = np.array([ibs_ad(rloo[a], gt[a][1], gt[a][0]) for a in common])
ibs_b1   = np.array([ibs_ad(b1[a],   gt[a][1], gt[a][0]) for a in common])

# Per-ad deltas
d_grpo_sft = ibs_grpo - ibs_sft
d_rloo_sft = ibs_rloo - ibs_sft
d_grpo_b1  = ibs_grpo - ibs_b1

print()
print("=== Per-ad ΔIBS (negative = candidate better) ===")
print(f"  GRPO - SFT : mean={d_grpo_sft.mean():+.5f}  median={np.median(d_grpo_sft):+.5f}  std={d_grpo_sft.std():.5f}")
print(f"  RLOO - SFT : mean={d_rloo_sft.mean():+.5f}  median={np.median(d_rloo_sft):+.5f}  std={d_rloo_sft.std():.5f}")
print(f"  GRPO - B1  : mean={d_grpo_b1.mean():+.5f}  median={np.median(d_grpo_b1):+.5f}  std={d_grpo_b1.std():.5f}")
print()

print("=== GRPO wins / losses vs SFT per-ad ===")
n_better = int((d_grpo_sft < -1e-6).sum())
n_worse  = int((d_grpo_sft > 1e-6).sum())
n_same   = len(common) - n_better - n_worse
print(f"  GRPO strictly better on {n_better}/{len(common)} ads, worse on {n_worse}, ≈ same on {n_same}")
# Top 5 GRPO wins
order = np.argsort(d_grpo_sft)
print("\n  Top 5 GRPO improvements (most negative Δ):")
for j in order[:5]:
    a = common[j]; T = gt[a][0]
    print(f"    ad={a}  T={T}  SFT_IBS={ibs_sft[j]:.4f}  GRPO_IBS={ibs_grpo[j]:.4f}  Δ={d_grpo_sft[j]:+.4f}")
print("  Top 5 GRPO regressions (most positive Δ):")
for j in order[-5:][::-1]:
    a = common[j]; T = gt[a][0]
    print(f"    ad={a}  T={T}  SFT_IBS={ibs_sft[j]:.4f}  GRPO_IBS={ibs_grpo[j]:.4f}  Δ={d_grpo_sft[j]:+.4f}")
print()

# Correlation of ΔIBS with ad properties
T_arr = np.array([gt[a][0] for a in common])
RT_arr = np.array([gt[a][1][gt[a][0]] for a in common])
R3_arr = np.array([gt[a][1][3] for a in common])
mid_arr = np.array([gt[a][1][gt[a][0]//2] for a in common])

print("=== Where does GRPO help? (Spearman of |Δ| with ad properties) ===")
for name, vec in [("T_i", T_arr), ("GT R(3)", R3_arr), ("GT R(T/2)", mid_arr), ("GT R(T)", RT_arr), ("SFT_IBS", ibs_sft)]:
    rho, _ = spearmanr(vec, d_grpo_sft)
    print(f"  ρ(Δ_grpo, {name}) = {rho:+.3f}")
print()

# Per-ad |R_hat - R| over time — show 3 example ads where GRPO won
print("=== 3 example ads: SFT vs GRPO curves at t=0..T_i ===")
for j in order[:3]:
    a = common[j]; T = gt[a][0]
    g = gt[a][1][:T+1]
    s = sft[a][:T+1]
    gr = grpo[a][:T+1]
    print(f"\nad={a} T={T}  SFT_IBS={ibs_sft[j]:.4f} → GRPO_IBS={ibs_grpo[j]:.4f}")
    print(f"  GT:   {np.round(g, 3).tolist()}")
    print(f"  SFT:  {np.round(s, 3).tolist()}")
    print(f"  GRPO: {np.round(gr, 3).tolist()}")
