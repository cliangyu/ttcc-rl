"""Plot SFT/GRPO/RLOO training curves to answer: did training saturate?"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

W = Path("/home/ssm-user/work/work-out")

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                if "loss" in r and "global_step/max_steps" in r:
                    step = int(str(r["global_step/max_steps"]).split("/")[0])
                    rows.append({"step": step, **r})
            except Exception:
                pass
    return sorted(rows, key=lambda x: x["step"])

sft = load_jsonl(W / "ttcc_sft/v5-20260520-102834/logging.jsonl")
grpo = load_jsonl(W / "ttcc_grpo/v3-20260520-131102/logging.jsonl")
rloo = load_jsonl(W / "ttcc_rloo/v2-20260520-141946/logging.jsonl")

print(f"SFT rows={len(sft)}, GRPO rows={len(grpo)}, RLOO rows={len(rloo)}")

fig = plt.figure(figsize=(16, 9), constrained_layout=True)
gs = GridSpec(2, 3, figure=fig)

# --- SFT loss & token_acc ---
axL = fig.add_subplot(gs[0, 0])
xs = [r["step"] for r in sft]
ys = [r["loss"] for r in sft]
axL.plot(xs, ys, "-o", color="#17becf", lw=2, ms=5)
axL.set_xlabel("SFT step")
axL.set_ylabel("loss")
axL.set_title(f"SFT loss  (start={ys[0]:.3f} → end={ys[-1]:.3f})")
axL.grid(alpha=0.3)
# Trend: split first/second half
mid = len(ys) // 2
axL.axvline(xs[mid], ls=":", color="gray", alpha=0.5)
axL.text(xs[mid] + 2, max(ys), "↓ first half  |  second half ↓", fontsize=9, color="gray", va="top")

axA = fig.add_subplot(gs[0, 1])
ta = [r["token_acc"] for r in sft]
axA.plot(xs, ta, "-o", color="#17becf", lw=2, ms=5)
axA.set_xlabel("SFT step"); axA.set_ylabel("token_acc")
axA.set_title(f"SFT token_acc  ({ta[0]:.3f} → {ta[-1]:.3f})")
axA.grid(alpha=0.3)

axG = fig.add_subplot(gs[0, 2])
gn = [r["grad_norm"] for r in sft]
axG.plot(xs, gn, "-o", color="#17becf", lw=2, ms=5)
axG.set_xlabel("SFT step"); axG.set_ylabel("grad_norm")
axG.set_title(f"SFT grad_norm  ({gn[0]:.3f} → {gn[-1]:.3f})")
axG.grid(alpha=0.3)

# --- GRPO + RLOO IBSReward/mean and reward_std ---
def get_field(rows, *keys):
    for r in rows:
        for k in keys:
            if k in r:
                yield r["step"], r[k]
                break

axR = fig.add_subplot(gs[1, 0])
g_steps, g_rew = zip(*get_field(grpo, "rewards/TTCCIBSReward/mean"))
r_steps, r_rew = zip(*get_field(rloo, "rewards/TTCCIBSReward/mean"))
axR.plot(g_steps, g_rew, "-o", color="#2ca02c", lw=2, ms=5, label=f"GRPO ({len(g_rew)} steps)")
axR.plot(r_steps, r_rew, "-^", color="#ff7f0e", lw=2, ms=5, label=f"RLOO ({len(r_rew)} steps)")
axR.set_xlabel("RL step")
axR.set_ylabel("IBSReward mean  (= 1 − IBS)")
axR.set_title("RL training-time reward (higher = better)")
axR.legend(loc="lower right")
axR.grid(alpha=0.3)

axS = fig.add_subplot(gs[1, 1])
g_steps2, g_std = zip(*get_field(grpo, "rewards/TTCCIBSReward/std"))
r_steps2, r_std = zip(*get_field(rloo, "rewards/TTCCIBSReward/std"))
axS.plot(g_steps2, g_std, "-o", color="#2ca02c", lw=2, ms=5, label="GRPO")
axS.plot(r_steps2, r_std, "-^", color="#ff7f0e", lw=2, ms=5, label="RLOO")
axS.set_xlabel("RL step")
axS.set_ylabel("σ across rollouts  (TTCCIBSReward/std)")
axS.set_title("Rollout reward spread  (group-relative advantage divides by this)")
axS.legend(loc="upper right")
axS.grid(alpha=0.3)

axRL = fig.add_subplot(gs[1, 2])
g_steps3, g_loss = zip(*get_field(grpo, "loss"))
r_steps3, r_loss = zip(*get_field(rloo, "loss"))
axRL.plot(g_steps3, g_loss, "-o", color="#2ca02c", lw=2, ms=5, label="GRPO")
axRL.plot(r_steps3, r_loss, "-^", color="#ff7f0e", lw=2, ms=5, label="RLOO")
axRL.set_xlabel("RL step")
axRL.set_ylabel("policy loss")
axRL.set_title("RL policy loss")
axRL.legend(loc="upper right")
axRL.grid(alpha=0.3)

fig.suptitle("Did SFT and RL training saturate?", fontsize=14, y=1.005)
OUT = W / "saturation.png"
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved {OUT}")

# Quick text summary
print()
print("=== SFT signals ===")
half = len(sft) // 2
loss_first = np.mean([r["loss"] for r in sft[:half]])
loss_late = np.mean([r["loss"] for r in sft[half:]])
print(f"  mean loss first half ({half} steps) = {loss_first:.4f}")
print(f"  mean loss last half ({len(sft)-half}) = {loss_late:.4f}")
print(f"  Δ = {loss_late - loss_first:+.4f}  ({'plateau' if abs(loss_late-loss_first)<0.05 else 'descending'})")
ta = [r["token_acc"] for r in sft]
print(f"  token_acc first half mean={np.mean(ta[:half]):.4f}, last half mean={np.mean(ta[half:]):.4f}")

print("\n=== GRPO signals ===")
print(f"  IBSReward mean first 10 steps = {np.mean(list(g_rew)[:10]):.4f}")
print(f"  IBSReward mean last 10 steps = {np.mean(list(g_rew)[-10:]):.4f}")
print(f"  reward_std mean first 10 = {np.mean(list(g_std)[:10]):.4f}  (= effective σ_g)")
print(f"  reward_std mean last 10  = {np.mean(list(g_std)[-10:]):.4f}")
print(f"  σ_g collapse? {'yes' if np.mean(list(g_std)[-10:]) < 0.005 else 'no'}")

print("\n=== RLOO signals ===")
print(f"  IBSReward mean first {min(10,len(r_rew))} = {np.mean(list(r_rew)[:10]):.4f}")
print(f"  IBSReward mean last  {min(10,len(r_rew))} = {np.mean(list(r_rew)[-10:]):.4f}")
