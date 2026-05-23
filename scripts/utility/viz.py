"""Build a one-figure visualization comparing OLD (mode-collapse) and NEW
(demo-prompted) Qwen2.5-Omni-3B predictions against TTCC ground truth.

Panels:
  A (top, 2x3 grid)   — 6 example ads, each panel: GT curve + OLD pred + NEW pred
  B (mid-left)        — scatter R_hat(3) vs R(3) for OLD and NEW
  C (mid-right)       — scatter R_hat(T_i) vs R(T_i) for OLD and NEW
  D (bottom, wide)    — per-ad std(R_hat) histogram, OLD vs NEW (highlights constants)

Saves to /home/ssm-user/work/work-out/qwen25_omni_3b_compare.png
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

WORK = Path("/home/ssm-user/work")
NEW = WORK / "work-out/qwen25_omni_3b_seed0.parquet"
OLD = WORK / "work-out/qwen25_omni_3b_seed0_modeCollapse.parquet"
OUT = WORK / "work-out/qwen25_omni_3b_compare.png"

new_df = pq.read_table(NEW).to_pandas()
old_df = pq.read_table(OLD).to_pandas()

# Build GT lookup
T_MIN, T_MAX = 5, 60
def compute_horizon(duration, curve_len):
    T_dur = round(float(duration))
    if T_dur < T_MIN: return None
    T_curve = curve_len - 1
    if min(T_dur, T_MAX) - T_curve > 1: return None
    T = min(T_dur, T_MAX, T_curve)
    return T if T >= T_MIN else None

gt = {}
for shard in sorted((WORK / "data/ttcc/data").glob("train-*-of-*.parquet")):
    t = pq.read_table(shard, columns=["ad_id","duration","retention_curve","split"]).to_pandas()
    t = t[t["split"] == "test"]
    for _, r in t.iterrows():
        ad_id = str(r["ad_id"]); raw = r["retention_curve"]
        if raw is None or len(raw) == 0: continue
        c = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(c)) or c[0] <= 0: continue
        T = compute_horizon(r["duration"], len(c))
        if T is None: continue
        c = c[:T+1] / c[0]
        for i in range(1, len(c)):
            if c[i] > c[i-1]:
                if c[i] - c[i-1] > 5e-3: c = None; break
                c[i] = c[i-1]
        if c is None: continue
        gt[ad_id] = np.clip(c, 0.0, 1.0)
print(f"GT loaded: {len(gt)} ads")

# Align
new_by_id = {row["ad_id"]: np.asarray(row["R_hat"]) for _, row in new_df.iterrows()}
old_by_id = {row["ad_id"]: np.asarray(row["R_hat"]) for _, row in old_df.iterrows()}
common = sorted(set(new_by_id) & set(old_by_id) & set(gt))
print(f"intersection: {len(common)} ads")

# Pick 6 representative ads: span GT R(T_i) low-to-high to show shape variety
items = [(ad, gt[ad]) for ad in common]
items.sort(key=lambda x: x[1][-1])  # by completion retention
idx = np.linspace(0, len(items)-1, 6).astype(int)
example_ads = [items[i][0] for i in idx]

# === FIGURE ===
fig = plt.figure(figsize=(15, 12), constrained_layout=True)
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1.1, 1.1])

# Panel A — 6 example ads (2x3 grid in top half)
for i, ad in enumerate(example_ads):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    g = gt[ad]; o = old_by_id[ad]; n = new_by_id[ad]
    T = len(g) - 1
    ax.plot(np.arange(len(g)), g, "-", lw=2.5, color="black", label="GT")
    ax.plot(np.arange(len(o)), o, "--", lw=1.7, color="#d62728", label="OLD (mode-collapse)", alpha=0.9)
    ax.plot(np.arange(len(n)), n, "--", lw=1.7, color="#2ca02c", label="NEW (demo-prompted)", alpha=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(T, len(o)-1, len(n)-1))
    ax.set_title(f"ad …{ad[-6:]}  T={T}", fontsize=10)
    ax.set_xlabel("t (s)")
    if i % 3 == 0:
        ax.set_ylabel("R(t)")
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8)

# Panel B+C — cross-ad scatter at t=3 and t=T_i
def scatter(ax, df_by_id, color, label, t_picker):
    xs, ys = [], []
    for ad in common:
        g = gt[ad]; p = df_by_id[ad]
        t = t_picker(g)
        if t < len(g) and t < len(p):
            xs.append(g[t]); ys.append(p[t])
    xs, ys = np.asarray(xs), np.asarray(ys)
    rho, _ = spearmanr(xs, ys)
    ax.scatter(xs, ys, s=18, color=color, alpha=0.6, label=f"{label}  ρ={rho:+.3f}")
    return xs, ys

axB = fig.add_subplot(gs[2, :2])
scatter(axB, old_by_id, "#d62728", "OLD", lambda g: 3)
scatter(axB, new_by_id, "#2ca02c", "NEW", lambda g: 3)
axB.plot([0,1],[0,1], "k:", lw=0.8, alpha=0.5)
axB.set_xlabel("GT R(3)"); axB.set_ylabel("predicted R(3)")
axB.set_xlim(-0.02, 1.02); axB.set_ylim(-0.02, 1.05)
axB.set_title("Cross-ad hook (t=3 s) — predicted vs true")
axB.legend(loc="lower right"); axB.grid(alpha=0.3)

axC = fig.add_subplot(gs[2, 2])
scatter(axC, old_by_id, "#d62728", "OLD", lambda g: len(g)-1)
scatter(axC, new_by_id, "#2ca02c", "NEW", lambda g: len(g)-1)
axC.plot([0,1],[0,1], "k:", lw=0.8, alpha=0.5)
axC.set_xlabel("GT R(T_i)"); axC.set_ylabel("predicted R(T_i)")
axC.set_xlim(-0.02, 1.02); axC.set_ylim(-0.02, 1.05)
axC.set_title("Cross-ad completion (t=T_i)")
axC.legend(loc="upper left", fontsize=8); axC.grid(alpha=0.3)

# Panel D — per-ad std(R_hat) histogram (left) + per-ad MAE (right)
axD = fig.add_subplot(gs[3, :2])
old_stds = np.array([np.std(R) for R in old_by_id.values()])
new_stds = np.array([np.std(R) for R in new_by_id.values()])
bins = np.linspace(0, 0.5, 41)
axD.hist(old_stds, bins=bins, color="#d62728", alpha=0.55, label=f"OLD  median std={np.median(old_stds):.3f}")
axD.hist(new_stds, bins=bins, color="#2ca02c", alpha=0.55, label=f"NEW  median std={np.median(new_stds):.3f}")
n_const_new = int((new_stds < 1e-6).sum())
n_const_old = int((old_stds < 1e-6).sum())
axD.axvline(0.01, color="gray", ls="--", lw=0.8)
axD.text(0.01, axD.get_ylim()[1]*0.9, f"  NEW constants: {n_const_new}/{len(new_stds)}\n  OLD constants: {n_const_old}/{len(old_stds)}",
         fontsize=10, va="top")
axD.set_xlabel("per-ad std of R_hat (0 = constant curve)")
axD.set_ylabel("# ads")
axD.set_title("Prediction-vector diversity — does the model commit to any decay at all?")
axD.legend(loc="upper right"); axD.grid(alpha=0.3)

# Panel D right — per-ad shape Spearman histogram
axE = fig.add_subplot(gs[3, 2])
def per_ad_spearman(df_by_id):
    rhos = []
    for ad in common:
        g = gt[ad]; p = df_by_id[ad]
        m = min(len(g), len(p))
        if m < 4 or np.std(p[:m]) < 1e-9 or np.std(g[:m]) < 1e-9:
            rhos.append(np.nan)
        else:
            r, _ = spearmanr(g[:m], p[:m]); rhos.append(r)
    return np.asarray(rhos)
ro = per_ad_spearman(old_by_id); rn = per_ad_spearman(new_by_id)
axE.hist(ro[~np.isnan(ro)], bins=np.linspace(-1,1,21), color="#d62728", alpha=0.55, label=f"OLD")
axE.hist(rn[~np.isnan(rn)], bins=np.linspace(-1,1,21), color="#2ca02c", alpha=0.55, label=f"NEW")
axE.set_xlabel("per-ad Spearman(pred, GT)")
axE.set_ylabel("# ads")
axE.set_title(f"Within-ad shape rank\n(constants → nan: OLD={int(np.isnan(ro).sum())}, NEW={int(np.isnan(rn).sum())})")
axE.legend(); axE.grid(alpha=0.3)

fig.suptitle("Qwen2.5-Omni-3B zero-shot on TTCC test (n=87 ads) — OLD vs NEW prompt", fontsize=13, y=1.005)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved {OUT}  ({OUT.stat().st_size//1024} KB)")
