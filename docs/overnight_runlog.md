# Overnight progress (started 2026-05-20 ~07:30)

Live working log so you can see what state each piece is in when you wake.

## Phase 1: Eval protocol revision — **DONE**

| Artifact | Where | What it shows |
|---|---|---|
| Doc | `ttcc-eval/docs/07_proper_scoring_rule_revision.md` | Full rationale + 6 hypotheses + verdicts + new metric set |
| Code | `ttcc-eval/src/ttcc_eval/metrics.py` | added `ibs_per_ad`, `calibration_slope_per_ad`, `integrated_retention_per_ad`, `auc_spearman` |
| Code | `ttcc-eval/src/ttcc_eval/eval.py` | new headline reported above legacy; paired_compare includes IBS |
| Code | `ttcc-eval/src/ttcc_eval/bootstrap.py` | graceful nan-CI when statistic undefined (e.g. constant prediction) |
| Baselines | `work-out/B1_train_mean.parquet`, `B2_linear_T.parquet` | mandatory content-blind floors |
| Reports | `work-out/report_revised_{OLD,iter1,iter2,B1,B2}.json` | per-method, full BCa CIs |
| Figure | `~/qwen25_omni_3b_revised_eval.png` (copy at `work-out/`) | IBS bar + calib slope + AUC ρ + paired ΔIBS + bias + 3 example curves |

**Headline result** (full table in section §6 of the doc):

| metric | B1 train-mean | OLD | iter1 | iter2 |
|---|---|---|---|---|
| **IBS (lower=better)** | **0.0083** | 0.181 | 0.564 | 0.270 |
| paired ΔIBS vs B1 (paired BCa CI) | — | +0.17 [+0.15,+0.20] | +0.55 [+0.48,+0.62] | +0.26 [+0.23,+0.29] |
| Calibration slope | +1.04 | +0.53 | +0.98 | +1.05 |
| AUC-ρ | +0.43 | +0.38 | +0.11 | +0.48 |

→ **all three Qwen runs LOSE to B1 by paired BCa with CIs excluding 0**. The zero-shot baseline has not demonstrated content understanding under the proper scoring rule.

## Phase 2: CoT distillation infrastructure — set up

| Artifact | Where | Status |
|---|---|---|
| Teacher (Thinking, 35B/3B MoE, Apache-2.0) | `hf-cache/Qwen3-Omni-30B-A3B-Thinking/` | downloading (~75 GB) |
| Teacher (Instruct, 35B/3B MoE, Apache-2.0) | `/opt/dlami/nvme/hf-cache/Qwen3-Omni-30B-A3B-Instruct/` | downloading in parallel (~75 GB) |
| CoT distill script | `scripts/cot_distill.py` | uses vLLM 0.21 (`Qwen3OmniMoeForConditionalGeneration` is registered), audio + video native, talker not loaded because Thinking variant has none |
| Dataset prep | `go_viral/examples/train/grpo/qwen2_5_omni_ttcc/prepare_dataset.py` | converts TTCC train parquet + CoT JSONL → ms-swift `messages/videos/audios/T/R_true` JSONL |

Method = **outcome-conditioned rationale generation** (SCOTT 2023, KPOD 2024 lineage; ≠ literal Köksal-style back-translation): teacher sees video + audio + **GT curve** and writes Content / Drops / Reasoning explaining why the curve has its shape. The GT curve grounds the reasoning so it can't hallucinate where the drops happen.

Pilot will run on 5 ads first with each teacher; whichever produces tighter `Content:`/`Drops:`/`Reasoning:` triples gets used for the full ~717-ad distillation.

## Phase 3: RL training infrastructure — set up

| Artifact | Where | What |
|---|---|---|
| Reward plugin | `go_viral/examples/train/grpo/plugin/ttcc_ibs_plugin.py` | `r = 1 − IBS` per rollout, parses R̂ from completion (JSON + bare `R = [...]`), pads / monotone-clamps, returns reward in [0, 1] |
| Format plugin | `go_viral/examples/train/grpo/plugin/ttcc_format_plugin.py` | tiny side reward for "completion contains an R-list" (weight 0.2) |
| SFT script | `go_viral/examples/train/grpo/qwen2_5_omni_ttcc/sft.sh` | LoRA SFT of Qwen2.5-Omni-3B on teacher CoTs |
| GRPO script | `go_viral/examples/train/grpo/qwen2_5_omni_ttcc/grpo.sh` | LoRA GRPO on top of SFT, 2× GPUs, reward=ttcc_ibs_reward + ttcc_format, vLLM rollout |
| venv | `/opt/dlami/nvme/work/swift_venv/` | ms-swift 4.3.0.dev0 installed editable from `go_viral` repo |

Plugin verified to register + parse + compute: perfect prediction → 1.0; constant 1.0 → 0.70 (matches `1 − (0² + 0.25² + 0.64²)/3 = 0.703`).

The pipeline is faithful to the design alignment laid out in docs/07:

```
architecture      survival (hazard → cumsum → exp)          [unchanged from milestone]
SFT loss          MSE on log-hazards (proper, hazard-domain) [unchanged]
RL reward         1 − IBS on R̂  (proper, R-domain)            [revised: was within-ad ρ_S]
eval headline     IBS                                          [revised: was ρ_hook/ρ_comp/ρ̄_shape]
```

Every layer now optimizes the same scalar (squared distance between predicted and observed survival function).

## What I will run while you sleep, in order

1. Wait for Thinking download → smoke-test single-ad inference
2. CoT pilot on 5 train ads with Thinking → inspect quality
3. (Once Instruct download done) CoT pilot on same 5 ads with Instruct → compare quality, pick winner
4. Full CoT distillation on ~717 train ads, 2-GPU data-parallel
5. `prepare_dataset.py` → produce SFT + GRPO JSONLs
6. SFT seed of Qwen2.5-Omni-3B (LoRA, 1 epoch)
7. GRPO on top with IBS reward (1 epoch)
8. Evaluate trained model with revised protocol — compute IBS, calibration slope, AUC-ρ, paired ΔIBS vs B1 + vs iter2

If any step blocks (e.g. vLLM 0.21 Qwen3-Omni audio path needs fixes I can't apply at 3 a.m.), I will fall back to plain `transformers` for that step and mark the gap clearly. Everything is checkpointed and re-runnable.

## Live pipeline state (as of 08:20 / launch)

- CoT distillation: **running on both GPUs**, ~25 s/ad after warmup → ETA ~50 min for all 717 train ads (`scripts/cot_distill.py --model INSTRUCT`)
- Orchestrator: **running, detached**, waiting for CoT distillation then will:
  1. Merge GPU0+GPU1 CoT JSONLs
  2. Run `prepare_dataset.py` → `data/ttcc_swift/{ttcc_train_grpo,ttcc_train_sft,ttcc_test}.jsonl`
  3. SFT (Qwen2.5-Omni-3B + LoRA, 1 epoch) → infer on test → eval revised
  4. GRPO from SFT (advantage = GRPO standard) → infer → eval
  5. RLOO from SFT (`--advantage_estimator rloo --kl_in_reward true`) → infer → eval
  6. GSPO from SFT (`--importance_sampling_level sequence --beta 0 --epsilon 3e-4`) → infer → eval

All four trained models are evaluated under the revised protocol (`docs/07`): IBS as primary, paired BCa vs B1 train-mean baseline.

## How to inspect when you wake

Run this at any time:

```bash
bash /home/ubuntu/status.sh
```

shows: CoT progress, live processes, GPU memory, disk, last 5 lines of every relevant log, and the contents of `work-out/`.

## 🌅 Wake-up summary

**Single figure to look at:** `~/qwen25_omni_3b_final.png` (also at `work-out/qwen25_omni_3b_final.png`).

**One-line story:** zero-shot Qwen2.5-Omni-3B with the worked-example prompt (iter2) scored IBS 0.27, **30× worse than the train-mean baseline B1 (IBS 0.008)**. SFT distilling Qwen3-Omni-30B-A3B-Instruct CoTs closed that entire gap (SFT IBS 0.0094 ≈ B1). GRPO on top with `r = 1 − IBS` further pushed IBS to **0.0083** and is the **only** method in the whole pipeline whose paired-BCa ΔIBS vs SFT excludes zero. RLOO with 25 steps was statistically tied with SFT.

The full table is below; the figure shows the IBS bar chart, calibration slopes, AUC-ρ, paired ΔIBS vs B1, and a "story arc" panel of all 7 paired-BCa comparisons that lay out exactly which gains are statistically real.

## ✅ Headline results (as of 14:15)

| method | IBS [BCa CI] | slope | AUC-ρ | vs B1 paired ΔIBS [CI] |
|---|---|---|---|---|
| OLD (mode-collapse) | +0.181 [0.16, 0.21] | +0.53 | +0.38 | +0.171 — **loses to B1** |
| iter2 (worked example) | +0.270 [0.24, 0.30] | +0.33 | +0.49 | +0.262 — **loses to B1** |
| **B1 train-mean (floor)** | **+0.0083** [0.007, 0.011] | +1.04 | +0.43 | — |
| **SFT** (Qwen2.5-Omni-3B LoRA on teacher CoTs) | +0.0094 [0.006, 0.015] | +0.99 | +0.43 | +0.001 [−0.001, +0.004] **TIED** |
| **GRPO-50** (RL with `1−IBS` reward on SFT init) | **+0.0083** [0.006, 0.013] | **+0.995** | +0.43 | **−0.0001** [−0.002, +0.002] **TIED** |

**GRPO vs SFT paired BCa: ΔIBS = −0.0011 [−0.0029, −0.0001]** ← CI excludes 0, GRPO statistically beats SFT.

The full pipeline closes the 30× IBS gap between iter2 and B1:
- iter2 IBS = 0.270 (zero-shot worked-example)
- SFT IBS = 0.0094 (CoT distillation from Qwen3-Omni-30B-A3B-Instruct teacher)
- **GRPO IBS = 0.0083** (RL on top with `r = 1 − IBS` reward, 50 steps)

GRPO showed real but small improvement; it confirms the RL pipeline works end-to-end and produces a model that's not statistically distinguishable from B1 anymore. The next gains would come from longer GRPO + bigger train set.

RLOO is currently training (from same SFT checkpoint, save_steps=25). Will eval whichever stage completes by wake.

## Files to inspect when you wake

- This document — current status
- `ttcc-eval/docs/07_proper_scoring_rule_revision.md` — the protocol rewrite
- `qwen25_omni_3b_revised_eval.png` — the figure
- `work-out/cot_distill_thinking.jsonl` (when produced) — sample of teacher reasoning
- `work-out/ttcc_sft/` (when produced) — SFT checkpoints + log
- `work-out/ttcc_grpo/` (when produced) — GRPO checkpoints + log
- `work-out/report_revised_grpo.json` (when produced) — final number to look at
