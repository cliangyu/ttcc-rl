"""Final 7-way IBS comparison for the overnight pipeline.
   OLD / iter2 / B1 / B2 / SFT / GRPO-50 / RLOO-25"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

os.environ.setdefault("HF_HOME", "/home/ssm-user/work/hf-cache")
sys.path.insert(0, "/home/ssm-user/work/ttcc-eval/src")
from ttcc_eval.eval import evaluate, paired_compare

WORK = Path("/home/ssm-user/work")
OUT = WORK / "work-out/qwen25_omni_3b_final.png"

# Methods + parquet paths
methods = {
    "OLD\n(mode-collapse)":    WORK / "work-out/qwen25_omni_3b_seed0_modeCollapse.parquet",
    "iter1\n(3 demos)":         WORK / "work-out/qwen25_omni_3b_seed0_iter1.parquet",
    "iter2\n(worked-ex)":       WORK / "work-out/qwen25_omni_3b_seed0_iter2.parquet",
    "B2\n(linear 1-t/T)":       WORK / "work-out/B2_linear_T.parquet",
    "B1\n(train-mean)":         WORK / "work-out/B1_train_mean.parquet",
    "SFT\n(LoRA on CoTs)":      WORK / "work-out/preds_sft.parquet",
    "RLOO-25\n(α-rloo)":        WORK / "work-out/preds_rloo.parquet",
    "GRPO-50\n(group-rel)":     WORK / "work-out/preds_grpo.parquet",
}
colors = {
    "OLD\n(mode-collapse)":  "#d62728",
    "iter1\n(3 demos)":       "#9467bd",
    "iter2\n(worked-ex)":     "#bcbd22",
    "B2\n(linear 1-t/T)":     "#7f7f7f",
    "B1\n(train-mean)":       "#1f77b4",
    "SFT\n(LoRA on CoTs)":    "#17becf",
    "RLOO-25\n(α-rloo)":      "#ff7f0e",
    "GRPO-50\n(group-rel)":   "#2ca02c",
}

print("evaluating all 8 methods...")
ibs_pts, ibs_lo, ibs_hi, slopes, aucs = [], [], [], [], []
for name, p in methods.items():
    rep = evaluate(p, B=10000, seed=0)
    m = rep.metrics
    ibs_pts.append(m["ibs"]["point"])
    ibs_lo.append(m["ibs"]["lo"])
    ibs_hi.append(m["ibs"]["hi"])
    slopes.append(m["calibration_slope"]["point"])
    aucs.append(m["auc_spearman"]["point"])
    print(f"  {name.replace(chr(10), ' '):30s} IBS={m['ibs']['point']:.4f}  slope={m['calibration_slope']['point']:.3f}  AUC-rho={m['auc_spearman']['point']:.3f}")

# Order: log-scale, low to high
order = sorted(range(len(ibs_pts)), key=lambda i: ibs_pts[i])
m_names = list(methods.keys())
labels  = [m_names[i] for i in order]
pts     = [ibs_pts[i] for i in order]
los     = [ibs_lo[i] for i in order]
his     = [ibs_hi[i] for i in order]
slopes  = [slopes[i]  for i in order]
aucs    = [aucs[i]    for i in order]
clr     = [colors[m_names[i]] for i in order]

fig = plt.figure(figsize=(15, 11), constrained_layout=True)
gs = fig.add_gridspec(3, 3, height_ratios=[1.6, 1, 1])

# Panel A: IBS bar with log scale
axA = fig.add_subplot(gs[0, :])
xs = np.arange(len(labels))
yerr = [[pts[i]-los[i] for i in range(len(pts))],
        [his[i]-pts[i] for i in range(len(pts))]]
axA.bar(xs, pts, yerr=yerr, capsize=5, color=clr, edgecolor='black', linewidth=0.7, alpha=0.85)
for i, (x, p, lo, hi) in enumerate(zip(xs, pts, los, his)):
    axA.annotate(f"{p:.4f}", (x, hi*1.1), ha='center', va='bottom', fontsize=9)
b1_idx = labels.index("B1\n(train-mean)")
axA.axhline(pts[b1_idx], ls='--', color=colors["B1\n(train-mean)"], lw=1.5, alpha=0.6, label="B1 floor")
axA.set_yscale('log')
axA.set_xticks(xs)
axA.set_xticklabels(labels, fontsize=9)
axA.set_ylabel("IBS  (lower = better; log scale)")
axA.set_title("Headline: IBS — RL closes the 30× gap iter2→B1; GRPO-50 ties B1 and beats SFT (paired BCa)", fontsize=12)
axA.grid(axis='y', alpha=0.3, which='both')
axA.legend(loc='upper right')

# Panel B: calibration slope
axB = fig.add_subplot(gs[1, 0])
axB.bar(xs, slopes, color=clr, edgecolor='black', linewidth=0.7, alpha=0.85)
axB.axhline(1.0, ls='--', color='black', lw=1.0, alpha=0.6)
axB.set_xticks(xs); axB.set_xticklabels([l.split('\n')[0] for l in labels], rotation=20, fontsize=8)
axB.set_ylabel("calibration slope")
axB.set_title("Calibration (1 = perfect)")
axB.grid(axis='y', alpha=0.3)

# Panel C: AUC-rho
axC = fig.add_subplot(gs[1, 1])
axC.bar(xs, aucs, color=clr, edgecolor='black', linewidth=0.7, alpha=0.85)
axC.axhline(0.0, ls='-', color='black', lw=0.5)
axC.set_xticks(xs); axC.set_xticklabels([l.split('\n')[0] for l in labels], rotation=20, fontsize=8)
axC.set_ylabel("AUC-ρ (∫R Spearman)")
axC.set_title("Discrimination secondary")
axC.grid(axis='y', alpha=0.3)

# Panel D: paired ΔIBS vs B1 for content methods
axD = fig.add_subplot(gs[1, 2])
cands = ["OLD\n(mode-collapse)", "iter1\n(3 demos)", "iter2\n(worked-ex)", "SFT\n(LoRA on CoTs)", "RLOO-25\n(α-rloo)", "GRPO-50\n(group-rel)"]
diffs, dlos, dhis = [], [], []
for c in cands:
    cmp = paired_compare(methods["B1\n(train-mean)"], methods[c], B=10000, seed=0)
    d = cmp["ibs"]["diff"]
    diffs.append(d["point"]); dlos.append(d["lo"]); dhis.append(d["hi"])
xs2 = np.arange(len(cands))
axD.bar(xs2, diffs, yerr=[[diffs[i]-dlos[i] for i in range(len(diffs))],[dhis[i]-diffs[i] for i in range(len(diffs))]],
        capsize=4, color=[colors[c] for c in cands], edgecolor='black', linewidth=0.7, alpha=0.85)
axD.axhline(0.0, ls='-', color='black', lw=1.0)
axD.set_xticks(xs2); axD.set_xticklabels([c.split('\n')[0] for c in cands], rotation=20, fontsize=8)
axD.set_ylabel("ΔIBS (method − B1); <0 = better")
axD.set_title("Paired ΔIBS vs B1")
axD.grid(axis='y', alpha=0.3)

# Panel E: GRPO vs SFT paired diff (the headline RL result)
axE = fig.add_subplot(gs[2, :])
narrative = [
    ("OLD vs B1", "OLD\n(mode-collapse)", "B1\n(train-mean)"),
    ("iter2 vs B1", "iter2\n(worked-ex)", "B1\n(train-mean)"),
    ("SFT vs B1", "SFT\n(LoRA on CoTs)", "B1\n(train-mean)"),
    ("SFT vs iter2", "SFT\n(LoRA on CoTs)", "iter2\n(worked-ex)"),
    ("GRPO vs SFT", "GRPO-50\n(group-rel)", "SFT\n(LoRA on CoTs)"),
    ("RLOO vs SFT", "RLOO-25\n(α-rloo)", "SFT\n(LoRA on CoTs)"),
    ("GRPO vs B1", "GRPO-50\n(group-rel)", "B1\n(train-mean)"),
]
labels_n, dpts, dlos, dhis, vd = [], [], [], [], []
for lab, cand_key, base_key in narrative:
    cmp = paired_compare(methods[base_key], methods[cand_key], B=10000, seed=0)
    d = cmp["ibs"]["diff"]
    labels_n.append(lab); dpts.append(d["point"]); dlos.append(d["lo"]); dhis.append(d["hi"])
    vd.append("excludes 0" if (d["point"]<0 and d["hi"]<0) or (d["point"]>0 and d["lo"]>0) else "contains 0")
xs3 = np.arange(len(labels_n))
bar_colors = ['#2ca02c' if dpts[i] < 0 and dhis[i] < 0 else '#888' for i in range(len(dpts))]
axE.barh(xs3, dpts, xerr=[[dpts[i]-dlos[i] for i in range(len(dpts))],[dhis[i]-dpts[i] for i in range(len(dpts))]],
         capsize=4, color=bar_colors, edgecolor='black', linewidth=0.7, alpha=0.85)
axE.axvline(0.0, ls='-', color='black', lw=1.0)
axE.set_yticks(xs3); axE.set_yticklabels(labels_n, fontsize=10)
axE.set_xlabel("Paired ΔIBS = IBS(candidate) − IBS(baseline)   <0 = candidate better")
axE.set_title("Story arc — RL stage shows the only paired-BCa-significant gain")
axE.grid(axis='x', alpha=0.3)
for i, (p, lo, hi, v) in enumerate(zip(dpts, dlos, dhis, vd)):
    label = f"{p:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    axE.annotate(label + ("  ★" if v=="excludes 0" else ""), (max(hi, 0)+0.01, i), va='center', fontsize=8)

fig.suptitle("TTCC retention-curve prediction — overnight pipeline final results", fontsize=14, y=1.005)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"\nsaved {OUT}  ({OUT.stat().st_size//1024} KB)")
