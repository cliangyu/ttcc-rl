"""Decisive tests for hypotheses H2-H6 about the TTCC eval protocol.

Output: one verdict per hypothesis with the supporting numbers.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr, rankdata

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_HOME", "/home/ssm-user/work/hf-cache")

WORK = Path("/home/ssm-user/work")

# ===== Load GT (same preprocessing as before) =====
T_MIN, T_MAX = 5, 60
def horizon(d, L):
    Td = round(float(d))
    if Td < T_MIN: return None
    Tc = L - 1
    if min(Td, T_MAX) - Tc > 1: return None
    T = min(Td, T_MAX, Tc)
    return T if T >= T_MIN else None

gt = {}
import pandas as pd
shard_rows = []
for shard in sorted((WORK / "data/ttcc/data").glob("train-*-of-*.parquet")):
    t = pq.read_table(shard, columns=["ad_id","duration","retention_curve","split"]).to_pandas()
    shard_rows.append(t)
df = pd.concat(shard_rows, ignore_index=True)
for _, r in df.iterrows():
    if r["split"] != "test":
        continue
    raw = r["retention_curve"]
    if raw is None or len(raw) == 0: continue
    c = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(c)) or c[0] <= 0: continue
    T = horizon(r["duration"], len(c))
    if T is None: continue
    c = c[:T+1] / c[0]
    ok = True
    for i in range(1, len(c)):
        if c[i] > c[i-1]:
            if c[i] - c[i-1] > 5e-3: ok=False; break
            c[i] = c[i-1]
    if ok: gt[str(r["ad_id"])] = (T, np.clip(c, 0, 1))

# Train-mean curve (for B1)
train_curves = []
for _, r in df.iterrows():
    if r["split"] != "train": continue
    raw = r["retention_curve"]
    if raw is None or len(raw) == 0: continue
    c = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(c)) or c[0] <= 0: continue
    T = horizon(r["duration"], len(c))
    if T is None: continue
    c = c[:T+1] / c[0]
    ok = True
    for i in range(1, len(c)):
        if c[i] > c[i-1]:
            if c[i] - c[i-1] > 5e-3: ok=False; break
            c[i] = c[i-1]
    if ok: train_curves.append(np.clip(c, 0, 1))
T_MAX_TR = max(len(c) for c in train_curves) - 1
train_mean = np.zeros(T_MAX_TR + 1); cnt = np.zeros(T_MAX_TR + 1)
for c in train_curves:
    for t, v in enumerate(c):
        train_mean[t] += v; cnt[t] += 1
train_mean /= np.maximum(cnt, 1)

# ===== Load all method predictions =====
def load(name, path):
    p = pq.read_table(path).to_pandas()
    return {row["ad_id"]: np.asarray(row["R_hat"]) for _, row in p.iterrows()}

methods = {
    "OLD":   load("OLD",   WORK / "work-out/qwen25_omni_3b_seed0_modeCollapse.parquet"),
    "iter1": load("iter1", WORK / "work-out/qwen25_omni_3b_seed0_iter1.parquet"),
    "iter2": load("iter2", WORK / "work-out/qwen25_omni_3b_seed0_iter2.parquet"),
}
# Build B1, B2 in-memory
methods["B1_train_mean"] = {ad: train_mean[:T+1].copy() for ad, (T, _) in gt.items()}
methods["B2_linear_T"]   = {ad: np.linspace(1, 0, T+1) for ad, (T, _) in gt.items()}

common = sorted(set.intersection(*(set(m.keys()) for m in methods.values())) & set(gt))
print(f"common ads across all methods + GT: n={len(common)}")
print()

# ===== Metric helpers =====
def ibs_per_ad(R_hat, R, T):
    L = min(len(R_hat), len(R), T+1)
    return float(np.mean((R_hat[:L] - R[:L])**2))

def iae_per_ad(R_hat, R, T):
    L = min(len(R_hat), len(R), T+1)
    return float(np.mean(np.abs(R_hat[:L] - R[:L])))

def per_ad_array(method_name, fn):
    arr = np.empty(len(common))
    for i, ad in enumerate(common):
        T, R = gt[ad]
        arr[i] = fn(methods[method_name][ad], R, T)
    return arr

# BCa CI helper
def bca(values, B=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(values)
    theta_hat = np.mean(values)
    # Bootstrap
    idx = rng.integers(0, n, size=(B, n))
    boot = values[idx].mean(axis=1)
    from scipy.stats import norm
    p = float((boot < theta_hat).mean())
    p = min(max(p, 1/(2*B)), 1 - 1/(2*B))
    z0 = norm.ppf(p)
    # acceleration via jackknife
    jk = np.array([np.mean(np.delete(values, i)) for i in range(n)])
    jk_mean = jk.mean()
    num = ((jk_mean - jk)**3).sum()
    den = 6 * ((jk_mean - jk)**2).sum()**1.5
    a = num / den if den > 0 else 0.0
    # BCa endpoints
    z_lo = norm.ppf(alpha/2); z_hi = norm.ppf(1-alpha/2)
    alpha1 = norm.cdf(z0 + (z0 + z_lo) / (1 - a*(z0 + z_lo)))
    alpha2 = norm.cdf(z0 + (z0 + z_hi) / (1 - a*(z0 + z_hi)))
    lo = float(np.percentile(boot, 100*alpha1))
    hi = float(np.percentile(boot, 100*alpha2))
    return theta_hat, lo, hi

def paired_bca(values_a, values_b, B=10000, alpha=0.05, seed=0):
    """Paired BCa CI on mean(a) - mean(b) using identical resample indices."""
    rng = np.random.default_rng(seed)
    diff = values_a - values_b
    return bca(diff, B=B, alpha=alpha, seed=seed)

# ============================================================
# H2: ρ_hook is dominated by T_i confound
# ============================================================
print("="*70)
print("H2: rho_hook is dominated by T_i confound")
print("    decisive: partial-Spearman( R(3), R_hat(3) ) controlling T_i")
print("              should be substantially smaller than raw rho_hook")
print("    AND: B2 (linear 1-t/T) raw rho_hook should be > 0 (despite zero content)")
print()

T_arr = np.array([gt[ad][0] for ad in common])
def partial_spearman(x, y, z):
    """partial Spearman of x,y controlling z, on ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    # Residualize ranks
    def resid(a, b):
        slope = ((a - a.mean()) * (b - b.mean())).sum() / ((b - b.mean())**2).sum()
        return a - slope * b
    rx_z = resid(rx, rz)
    ry_z = resid(ry, rz)
    num = ((rx_z - rx_z.mean()) * (ry_z - ry_z.mean())).sum()
    den = float(np.sqrt(((rx_z - rx_z.mean())**2).sum() * ((ry_z - ry_z.mean())**2).sum()))
    return float(num/den) if den > 0 else float("nan")

gt_R3 = np.array([gt[ad][1][3] for ad in common])
print(f"{'method':<16} {'raw rho_hook':>14} {'partial (out T_i)':>18} {'attenuation':>12}")
for name in ["OLD", "iter1", "iter2", "B1_train_mean", "B2_linear_T"]:
    pred_R3 = np.array([methods[name][ad][3] for ad in common])
    if np.std(pred_R3) < 1e-9:
        print(f"{name:<16} {'nan (const)':>14} {'-':>18} {'-':>12}")
        continue
    raw, _ = spearmanr(pred_R3, gt_R3)
    pr = partial_spearman(pred_R3, gt_R3, T_arr)
    print(f"{name:<16} {raw:>+14.4f} {pr:>+18.4f} {raw-pr:>+12.4f}")
print()

# ============================================================
# H3: iter2 IBS is WORSE than B1's IBS (paired BCa)
# ============================================================
print("="*70)
print("H3: iter2 LOSES to B1 (train-mean curve) on IBS")
print("    decisive: IBS(iter2) > IBS(B1), paired BCa CI of diff excludes 0")
print()
ibs = {m: per_ad_array(m, ibs_per_ad) for m in methods}
print(f"{'method':<16} {'IBS':>8} {'95% BCa CI':>22}")
for name in ["B1_train_mean", "B2_linear_T", "OLD", "iter1", "iter2"]:
    point, lo, hi = bca(ibs[name])
    print(f"{name:<16} {point:>8.4f}  [{lo:>+7.4f},{hi:>+7.4f}]")
print()
for cand in ["iter2", "iter1", "OLD"]:
    d_point, d_lo, d_hi = paired_bca(ibs[cand], ibs["B1_train_mean"])
    sign = "WORSE than B1" if d_point > 0 and d_lo > 0 else ("BETTER than B1" if d_point < 0 and d_hi < 0 else "no significant diff")
    print(f"  paired IBS({cand}) - IBS(B1) = {d_point:+.4f}  [{d_lo:+.4f},{d_hi:+.4f}]   <- {sign}")
print()

# ============================================================
# H4: ordering of methods differs between rank-based and IBS
# ============================================================
print("="*70)
print("H4: method ordering changes between rho_comp (rank) and IBS (proper)")
print()
rho_comp = {}
for name in methods:
    pred_RT = np.array([methods[name][ad][gt[ad][0]] for ad in common])
    if np.std(pred_RT) < 1e-9:
        rho_comp[name] = float("nan")
    else:
        gt_RT = np.array([gt[ad][1][gt[ad][0]] for ad in common])
        r,_ = spearmanr(pred_RT, gt_RT)
        rho_comp[name] = r
order_by_rho = sorted(methods.keys(), key=lambda m: -rho_comp[m] if np.isfinite(rho_comp[m]) else 999)
order_by_ibs = sorted(methods.keys(), key=lambda m: ibs[m].mean())
print(f"  rank by rho_comp (higher=better): {order_by_rho}")
print(f"  rank by IBS      (lower=better):  {order_by_ibs}")
print(f"  identical orderings? {order_by_rho == order_by_ibs}")
print()

# ============================================================
# H5: iter2 fails by SYSTEMATIC over-prediction (slope, bias)
# ============================================================
print("="*70)
print("H5: iter2's failure is systematic over-prediction (not random noise)")
print("    decisive: bias = mean(R_hat - R) over all (ad, t) > 0")
print("              calibration slope (regress R on R_hat) ~ 1 if calibrated")
print()
print(f"{'method':<16} {'mean(R_hat-R)':>14} {'cal slope':>12} {'cal intercept':>14}")
for name in ["B1_train_mean", "iter1", "iter2"]:
    all_pred, all_gt = [], []
    for ad in common:
        T, R = gt[ad]
        Rh = methods[name][ad]
        L = min(len(Rh), len(R), T+1)
        all_pred.extend(Rh[:L].tolist())
        all_gt.extend(R[:L].tolist())
    all_pred = np.array(all_pred); all_gt = np.array(all_gt)
    bias = float((all_pred - all_gt).mean())
    if np.std(all_pred) > 1e-9:
        slope = float(np.cov(all_gt, all_pred, ddof=0)[0,1] / np.var(all_pred))
        intercept = float(all_gt.mean() - slope * all_pred.mean())
    else:
        slope = float("nan"); intercept = float("nan")
    print(f"{name:<16} {bias:>+14.4f} {slope:>+12.4f} {intercept:>+14.4f}")
print()

# ============================================================
# H7: CI width
# ============================================================
print("="*70)
print("H7: N=87 -> CI half-width on rho is ~0.2, larger than typical method gaps")
print()
print(f"  theoretical SE(rho) ~ 1/sqrt(N-1) = {1/np.sqrt(len(common)-1):.3f}")
print(f"  -> 95% CI half-width ~ {1.96/np.sqrt(len(common)-1):.3f}")
print(f"  observed iter2 rho_comp CI half-width: {(0.4481 - 0.0298)/2:.3f}")
print(f"  observed gap iter2 vs iter1 rho_comp:  {0.2543 - 0.1786:.4f}")
print(f"  -> gap << CI half-width: methods indistinguishable on rho_comp alone")
print()
