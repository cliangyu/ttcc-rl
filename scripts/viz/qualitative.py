"""Qualitative example: GT curve vs all method predictions, with each method's CoT text shown."""
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

AD = "7585042709691629585"
WORK = Path("/home/ssm-user/work")

# GT
gt_raw = load_ground_truth().filter_split("test")
clean, _ = preprocess(gt_raw)
gt_idx = list(clean.ad_id).index(AD)
T = int(clean.T[gt_idx])
gt_curve = np.asarray(clean.curves[gt_idx])

# All method predictions
def load_curve(p):
    tbl = pq.read_table(p).to_pandas()
    row = tbl[tbl["ad_id"] == AD]
    if len(row) == 0: return None
    return np.asarray(row.iloc[0]["R_hat"])

curves = {
    "iter2 (zero-shot)":     ("#bcbd22", load_curve(WORK / "work-out/qwen25_omni_3b_seed0_iter2.parquet")),
    "B1 (train-mean)":       ("#1f77b4", load_curve(WORK / "work-out/B1_train_mean.parquet")),
    "SFT (after CoT distill)": ("#17becf", load_curve(WORK / "work-out/preds_sft.parquet")),
    "GRPO-50":               ("#2ca02c", load_curve(WORK / "work-out/preds_grpo.parquet")),
    "RLOO-25":               ("#ff7f0e", load_curve(WORK / "work-out/preds_rloo.parquet")),
}

# CoT text from swift JSONL outputs
def find_text(path):
    target_vid = f"/home/ssm-user/work/data/videos/{AD}.mp4"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            v = r.get("videos") or [r.get("video", "")]
            if v and target_vid == v[0]:
                return r.get("response", "")
    return ""

cots = {
    "SFT":  find_text("/tmp/ttcc_infer_LnFg.jsonl"),
    "GRPO": find_text("/tmp/ttcc_infer_KKWx.jsonl"),
    "RLOO": find_text("/tmp/ttcc_infer_REQR.jsonl"),
}

# per-ad IBS
def ibs(p):
    L = min(len(p), len(gt_curve), T+1)
    return float(np.mean((p[:L] - gt_curve[:L])**2))

fig = plt.figure(figsize=(15, 12), constrained_layout=True)
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1.5])

# A. Curves
axA = fig.add_subplot(gs[0, :])
axA.plot(np.arange(T+1), gt_curve[:T+1], '-', color='black', lw=3, label=f"GT  (ad …{AD[-6:]}, T={T}, R(1)={gt_curve[1]:.2f})")
for name, (color, c) in curves.items():
    if c is None: continue
    L = min(len(c), T+1)
    axA.plot(np.arange(L), c[:L], '--', color=color, lw=2, alpha=0.9, label=f"{name}  IBS={ibs(c):.4f}")
axA.set_xlabel("t (s)")
axA.set_ylabel("R(t)")
axA.set_ylim(-0.05, 1.05)
axA.set_title(f"Ad …{AD[-6:]}: GT vs all methods\nVideo: two men in a forest, one with a blue lightsaber, choreographed fight; text overlays JP+EN", fontsize=12)
axA.legend(loc='upper right', fontsize=10)
axA.grid(alpha=0.3)
# Annotate the key drop
axA.annotate(f"GT R(1)={gt_curve[1]:.2f}\n(strong hook)", (1, gt_curve[1]), xytext=(2.5, 0.85),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
sft_r1 = curves["SFT (after CoT distill)"][1][1] if curves["SFT (after CoT distill)"][1] is not None else 0
grpo_r1 = curves["GRPO-50"][1][1] if curves["GRPO-50"][1] is not None else 0
axA.annotate(f"SFT R(1)={sft_r1:.2f}\n(crash like train-mean)", (1, sft_r1), xytext=(2, 0.05),
             fontsize=10, color='#17becf', arrowprops=dict(arrowstyle='->', color='#17becf'))
axA.annotate(f"GRPO R(1)={grpo_r1:.2f}\n(RL lifts toward GT)", (1, grpo_r1), xytext=(5, 0.45),
             fontsize=10, color='#2ca02c', arrowprops=dict(arrowstyle='->', color='#2ca02c'))

# B-D. CoT texts side by side
for col, (tag, color) in enumerate([("SFT", '#17becf'), ("GRPO", '#2ca02c'), ("RLOO", '#ff7f0e')]):
    ax = fig.add_subplot(gs[1, col])
    ax.axis('off')
    ax.text(0.5, 0.97, tag, ha='center', va='top', fontsize=14, fontweight='bold', color=color, transform=ax.transAxes)
    # Wrap text manually since matplotlib doesn't do that by default
    import textwrap
    txt = cots.get(tag, "(missing)")
    # Bold-ish the Content/Drops/Reasoning/Curve labels
    formatted = ""
    for line in txt.split("\n"):
        if line.startswith(("Content:", "Drops:", "Reasoning:", "Curve:")):
            label, _, rest = line.partition(":")
            wrapped = textwrap.fill(rest.strip(), width=42)
            formatted += f"\n● {label}:\n{wrapped}\n"
        else:
            formatted += textwrap.fill(line, width=42) + "\n"
    ax.text(0.02, 0.92, formatted, ha='left', va='top', fontsize=9, family='monospace',
            transform=ax.transAxes, wrap=True)

fig.suptitle("Qualitative example: same ad, same CoT description, different R predictions", fontsize=13, y=1.005)
OUT = WORK / "work-out/qualitative_example.png"
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved {OUT}")
