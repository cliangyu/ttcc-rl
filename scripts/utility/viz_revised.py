"""Build IBS-headline visualization for the revised eval protocol.

5 methods + 1 mandatory floor line (B1). Bar chart with BCa CIs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

WORK = Path("/home/ssm-user/work")
OUT_PNG = WORK / "work-out/qwen25_omni_3b_revised_eval.png"

methods_order = ["B1", "B2", "OLD", "iter1", "iter2"]
colors = {
    "B1": "#1f77b4",
    "B2": "#9467bd",
    "OLD": "#d62728",
    "iter1": "#ff7f0e",
    "iter2": "#2ca02c",
}
labels = {
    "B1": "B1\n(train-mean)",
    "B2": "B2\n(linear 1-t/T)",
    "OLD": "OLD\n(mode collapse)",
    "iter1": "iter1\n(3-demo prompt)",
    "iter2": "iter2\n(worked example)",
}

reports = {}
for m in methods_order:
    reports[m] = json.loads((WORK / f"work-out/report_revised_{m}.json").read_text())

fig = plt.figure(figsize=(16, 11), constrained_layout=True)
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

# --- A. IBS bar chart with CIs ---
axA = fig.add_subplot(gs[0, :])
xs = np.arange(len(methods_order))
points = [reports[m]["metrics"]["ibs"]["point"] for m in methods_order]
los = [reports[m]["metrics"]["ibs"]["lo"] for m in methods_order]
his = [reports[m]["metrics"]["ibs"]["hi"] for m in methods_order]
yerr = [[points[i] - los[i] for i in range(5)],
        [his[i] - points[i] for i in range(5)]]
bars = axA.bar(xs, points, yerr=yerr, capsize=6,
               color=[colors[m] for m in methods_order],
               edgecolor='black', linewidth=0.6, alpha=0.85)
# Annotate values
for i, (x, p, lo, hi) in enumerate(zip(xs, points, los, his)):
    axA.annotate(f"{p:.4f}\n[{lo:.3f},{hi:.3f}]", (x, hi + 0.015),
                 ha='center', va='bottom', fontsize=8)
# Reference line at B1
axA.axhline(points[0], ls='--', color=colors["B1"], lw=1, alpha=0.6)
axA.text(4.4, points[0] + 0.005, f"B1 floor = {points[0]:.4f}",
         color=colors["B1"], fontsize=9)
axA.set_xticks(xs)
axA.set_xticklabels([labels[m] for m in methods_order])
axA.set_ylabel("IBS  (lower = better; 0 = perfect)")
axA.set_title("Headline: Integrated Brier Score — all 3 Qwen runs LOSE to the train-mean baseline B1",
              fontsize=12)
axA.grid(axis='y', alpha=0.3)
axA.set_ylim(0, max(his) * 1.18)

# --- B. Calibration slope ---
axB = fig.add_subplot(gs[1, 0])
pts = [reports[m]["metrics"]["calibration_slope"]["point"] for m in methods_order]
los = [reports[m]["metrics"]["calibration_slope"]["lo"] for m in methods_order]
his = [reports[m]["metrics"]["calibration_slope"]["hi"] for m in methods_order]
axB.bar(xs, pts, yerr=[[pts[i]-los[i] for i in range(5)],
                       [his[i]-pts[i] for i in range(5)]],
        capsize=4, color=[colors[m] for m in methods_order],
        edgecolor='black', linewidth=0.6, alpha=0.85)
axB.axhline(1.0, ls='--', color='black', lw=1.0, alpha=0.7)
axB.text(4.4, 1.02, "target = 1.0", color='black', fontsize=8)
axB.set_xticks(xs)
axB.set_xticklabels([m for m in methods_order], rotation=0, fontsize=8)
axB.set_ylabel("Calibration slope")
axB.set_title("Calibration slope (1.0 = calibrated)")
axB.grid(axis='y', alpha=0.3)

# --- C. AUC Spearman ---
axC = fig.add_subplot(gs[1, 1])
pts = [reports[m]["metrics"]["auc_spearman"]["point"] for m in methods_order]
los = [reports[m]["metrics"]["auc_spearman"]["lo"] for m in methods_order]
his = [reports[m]["metrics"]["auc_spearman"]["hi"] for m in methods_order]
axC.bar(xs, pts, yerr=[[pts[i]-los[i] for i in range(5)],
                       [his[i]-pts[i] for i in range(5)]],
        capsize=4, color=[colors[m] for m in methods_order],
        edgecolor='black', linewidth=0.6, alpha=0.85)
axC.axhline(0.0, ls='--', color='black', lw=0.8, alpha=0.5)
axC.set_xticks(xs)
axC.set_xticklabels([m for m in methods_order], rotation=0, fontsize=8)
axC.set_ylabel("ρ on ∫R dt  (secondary discrimination)")
axC.set_title("auc_spearman  (secondary)")
axC.grid(axis='y', alpha=0.3)

# --- D. Paired ΔIBS vs B1 ---
axD = fig.add_subplot(gs[1, 2])
# Recompute paired diffs from raw IBS arrays
import sys; sys.path.insert(0, '/home/ssm-user/work/ttcc-eval/src')
import os; os.environ.setdefault("HF_HOME", str(WORK / "hf-cache"))
from ttcc_eval.eval import paired_compare

cmps = {}
for m in ["OLD", "iter1", "iter2", "B2"]:
    cmps[m] = paired_compare(WORK / "work-out/B1_train_mean.parquet",
                              WORK / f"work-out/{'B2_linear_T' if m == 'B2' else 'qwen25_omni_3b_seed0_modeCollapse' if m == 'OLD' else 'qwen25_omni_3b_seed0_iter1' if m == 'iter1' else 'qwen25_omni_3b_seed0_iter2'}.parquet",
                              B=10000, seed=0)
cand_order = ["B2", "OLD", "iter1", "iter2"]
xs = np.arange(len(cand_order))
diffs = [cmps[m]["ibs"]["diff"]["point"] for m in cand_order]
los = [cmps[m]["ibs"]["diff"]["lo"] for m in cand_order]
his = [cmps[m]["ibs"]["diff"]["hi"] for m in cand_order]
axD.bar(xs, diffs, yerr=[[diffs[i]-los[i] for i in range(4)],
                          [his[i]-diffs[i] for i in range(4)]],
        capsize=4, color=[colors[m] for m in cand_order],
        edgecolor='black', linewidth=0.6, alpha=0.85)
axD.axhline(0.0, ls='-', color='black', lw=1.0)
axD.set_xticks(xs)
axD.set_xticklabels(cand_order)
axD.set_ylabel("ΔIBS (cand − B1); >0 = worse")
axD.set_title("Paired ΔIBS vs B1 (must be < 0 with CI excluding 0 to claim 'beats baseline')")
axD.grid(axis='y', alpha=0.3)

# --- E. Legacy comparison: rho_comp vs IBS — show the inversion ---
axE = fig.add_subplot(gs[2, 0])
rho_comp = [reports[m]["metrics"]["completion_spearman"]["point"] for m in methods_order]
ibs_pts = [reports[m]["metrics"]["ibs"]["point"] for m in methods_order]
axE.scatter(ibs_pts, rho_comp, s=120, c=[colors[m] for m in methods_order], edgecolors='black')
for i, m in enumerate(methods_order):
    axE.annotate(m, (ibs_pts[i], rho_comp[i]), xytext=(6, 6), textcoords='offset points', fontsize=10)
axE.set_xlabel("IBS (lower better)")
axE.set_ylabel("ρ_comp (legacy; higher better)")
axE.set_title("Inverse orderings between rank metric and proper scoring rule")
axE.grid(alpha=0.3)
axE.axhline(0, ls=':', color='gray')

# --- F. Bias bar chart ---
axF = fig.add_subplot(gs[2, 1])
# Compute mean(R_hat - R) per method
def gt_dict():
    out = {}
    sys.path.insert(0, '/home/ssm-user/work/ttcc-eval/src')
    from ttcc_eval.data import load_ground_truth
    from ttcc_eval.preprocess import preprocess
    gt = load_ground_truth().filter_split("test")
    clean, _ = preprocess(gt)
    for i in range(len(clean.T)):
        out[str(clean.ad_id[i])] = (int(clean.T[i]), np.asarray(clean.curves[i], dtype=np.float64))
    return out
gt_lookup = gt_dict()
bias_by = {}
for m in methods_order:
    fname = "B1_train_mean" if m == "B1" else "B2_linear_T" if m == "B2" else \
            "qwen25_omni_3b_seed0_modeCollapse" if m == "OLD" else \
            "qwen25_omni_3b_seed0_iter1" if m == "iter1" else "qwen25_omni_3b_seed0_iter2"
    preds = pq.read_table(WORK / f"work-out/{fname}.parquet").to_pandas()
    diffs = []
    for _, row in preds.iterrows():
        ad = row["ad_id"]
        if ad not in gt_lookup: continue
        T, g = gt_lookup[ad]
        p = np.asarray(row["R_hat"], dtype=np.float64)
        L = min(len(p), len(g), T+1)
        diffs.extend((p[:L] - g[:L]).tolist())
    bias_by[m] = float(np.mean(diffs))
xs = np.arange(len(methods_order))
axF.bar(xs, [bias_by[m] for m in methods_order], color=[colors[m] for m in methods_order],
        edgecolor='black', linewidth=0.6, alpha=0.85)
axF.axhline(0, ls='-', color='black', lw=1.0)
axF.set_xticks(xs)
axF.set_xticklabels(methods_order)
axF.set_ylabel("mean(R̂ − R)  over all (ad, t)")
axF.set_title("Systematic bias (0 = unbiased)")
axF.grid(axis='y', alpha=0.3)

# --- G. Example 3 ads ---
axG = fig.add_subplot(gs[2, 2])
# Pick 3 example ads spanning GT completion span
ads_sorted = sorted(gt_lookup.keys(), key=lambda a: gt_lookup[a][1][-1])
example_ads = [ads_sorted[i] for i in [0, len(ads_sorted)//2, -1]]
ex_offset = [0.0, 0.25, 0.50]  # vertical offset for visibility
for k, ad in enumerate(example_ads):
    T, g = gt_lookup[ad]
    axG.plot(np.arange(len(g)), g + ex_offset[k] * 0, '-', color='black', lw=1.5, alpha=0.9, label="GT" if k == 0 else None)
    for m in ["B1", "iter2"]:
        fname = "B1_train_mean" if m == "B1" else "qwen25_omni_3b_seed0_iter2"
        preds = pq.read_table(WORK / f"work-out/{fname}.parquet").to_pandas()
        row = preds[preds["ad_id"] == ad].iloc[0]
        p = np.asarray(row["R_hat"], dtype=np.float64)
        axG.plot(np.arange(len(p)), p, '--', color=colors[m], lw=1.2, alpha=0.7, label=m if k == 0 else None)
axG.legend(fontsize=9, loc='upper right')
axG.set_xlabel("t (s)")
axG.set_ylabel("R(t)")
axG.set_title("3 example ads: GT vs B1 vs iter2\n(B1 nails the magnitude; iter2 over-predicts)")
axG.grid(alpha=0.3)
axG.set_ylim(-0.02, 1.05)

fig.suptitle("Revised eval (docs/07): IBS as proper-scoring-rule headline, with mandatory B1 floor",
             fontsize=13, y=1.005)
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"saved {OUT_PNG}  ({OUT_PNG.stat().st_size//1024} KB)")
