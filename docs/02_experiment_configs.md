# Experiment config audit (every hyperparameter + assumption)

Goal: surface every implicit choice in the pipeline so we can review them
together before extending training. Each row cites the file:line where the
value lives, the value itself, and the assumption / rationale behind it.

Branch: `cliangyu/go_viral:ttcc-rl`. All training scripts live under
`examples/train/grpo/qwen2_5_omni_ttcc/`. All Python helpers live under
`cliangyu/ttcc-rl` (separate repo, not yet pushed).

---

## 0. Data pipeline

### Source
- Source parquets: `/home/ssm-user/work/data/ttcc/data/train-*.parquet`
- Columns used: `ad_id`, `duration`, `retention_curve`, `split`,
  `video_local_path`
- Split filter: `split == "train"` for SFT/GRPO, `split == "test"` for eval

### Per-ad filters (`prepare_dataset.py:79–106`)
| filter | rule | assumption |
|---|---|---|
| $T_{\min} = 5$ | drop ads shorter than 5s | very short ads have too few timesteps for meaningful $R(t)$ curve |
| $T_{\max} = 60$ | clip horizon to 60s | per-token cost grows with $T$; 60s covers most ads |
| $\lvert \text{retention\_curve}\rvert \ge T+1$ | drop if curve has fewer points than $\text{round}(\text{duration})$ allows | data integrity |
| $c[0] > 0$ | drop ads with zero initial retention | data integrity |
| monotone-non-increasing tolerance | reject ad if any positive increment $> 5\!\times\!10^{-3}$, else clamp the increment | small numerical noise in GT is acceptable; bigger spikes look like bad data |
| $T = \min(\text{round}(\text{duration}),\,T_{\max},\,L-1)$ where $L = \lvert \text{retention\_curve}\rvert$ | unify three notions of length | one source of truth for $T_i$ |

After filters: **717 train ads** (with CoT seed), **87 test ads** (after dropping `7625656718593097746` for corrupted audio).

### CoT seeds (`docs/01_method.md`, `scripts/cot_distill.py`)
- Teacher: `Qwen3-Omni-30B-A3B-Instruct` (Apache-2.0, local), talker disabled.
- Prompt strategy: **outcome-conditioned rationale generation** — teacher
  sees the GT retention curve in its prompt and explains *why* it has that
  shape (Content / Drops / Reasoning, no curve emitted by teacher).
- Output file: `/home/ssm-user/work/work-out/cot_distill_thinking.jsonl`
  (one row per train ad with `{ad_id, raw}` where `raw` is the teacher's
  Content+Drops+Reasoning string).

**Assumption:** teacher CoT is informative for the student. **Not tested**
directly — the no-CoT SFT experiment (running now) will tell us how much
of the IBS comes from CoT vs from numeric tokens alone.

### Prompts (`prepare_dataset.py:39–54`)

System (identical across all variants):
```
You are an expert in short-form video advertising. You forecast
second-by-second audience retention curves. R(t) is the fraction of
viewers still watching at second t, with R(0) = 1 by definition.
R(t) is monotone non-increasing. Use the video and audio content to
estimate where viewers drop off.
```

User (per-ad, with-CoT variant):
```
This ad is {T} seconds long. Watch and listen to it, then write your
analysis on three labeled lines and finish with the JSON curve.
Content: <one sentence describing the ad>.
Drops: <one or two sentences naming SPECIFIC seconds where retention
falls fastest, with reasons tied to what happens on screen or audio>.
Reasoning: <one sentence summarizing the overall shape>.
Curve: {"R": [1.0, R(1), R(2), ..., R({T})]}
Rules: the Curve line MUST be a valid JSON object exactly of the form
{"R": [...]}, exactly {T+1} numbers in R, R(0) = 1.0, every value in
[0, 1], monotone non-increasing.
```

Assistant (CoT SFT target): teacher's `raw` field + `\nCurve: {"R": [1.0000, 0.3327, ..., 0.0103]}` with all numerical values filled in to 4 decimal places.

Assistant (no-CoT SFT target): just `Curve: {"R": [1.0000, ...]}` — **no Content/Drops/Reasoning preceding**. User message also stripped of the CoT structure.

**Assumption to audit:** the prompt embeds the answer schema verbatim. This is a strong inductive bias toward producing structured output, but it also means the model sees the GT length `T` in the user prompt — *which we've decided is fine because at test time we'd know how long the ad is anyway*. Still worth noting.

---

## 1. SFT — base (`sft.sh`)

This was the one already-run experiment producing the SFT predictions in `preds_sft.parquet`.

| param | value | rationale / assumption |
|---|---|---|
| `--model` | `Qwen2.5-Omni-3B` (local) | milestone-aligned student model; native audio+video; Apache-2.0 |
| `--tuner_type lora` | LoRA | full FT on 3B would saturate 96GB GPUs at bs=1; LoRA cheap |
| `--lora_rank` | 16 | community default for 3B; **not tuned** |
| `--lora_alpha` | 32 | 2× rank, standard scaling; **not tuned** |
| `--target_modules` | `all-linear` | swift convention; covers attn + MLP; **not tuned** |
| `--torch_dtype` | bfloat16 | A100 native; saves 50% memory vs fp32 |
| `--max_length` | 8192 | enough for full assistant target (CoT ≈400 tokens + curve ≈ 250 tokens) plus video tokens (~4096) |
| `--max_pixels` | 49152 (=192×256) | empirical OOM bound at FPS_MAX_FRAMES=32 |
| `VIDEO_MAX_PIXELS=49152` env | matches `--max_pixels` | swift uses both; we set both to be safe |
| `FPS_MAX_FRAMES=32` env | 32 frames cap at FPS=1.0 | covers ads up to 32s; longer ads get truncated. Our T_MAX=60 means ads up to 60s pass the data filter but get visually truncated at 32 frames in training. **This is a known asymmetry.** Inference uses 24 → even more truncated. |
| `FPS=1.0` env | 1 frame per second | the prediction is per-second, so 1 fps lines up with target granularity |
| `--num_train_epochs` | 1 | base experiment; "extended" variant is 3 |
| `--per_device_train_batch_size` | 1 | bs>1 OOMs at video length |
| `--gradient_accumulation_steps` | 4 | effective batch 4×2 GPUs = 8 |
| `--learning_rate` | 1e-4 | LoRA standard; **not tuned** |
| `--warmup_ratio` | 0.05 | 5% of 90 steps = 4 steps warmup |
| `--lr_scheduler` | cosine (swift default when not set) | reaches 0 at end of 1 epoch — for **extended runs this means LR effectively re-decayed** |
| `--gradient_checkpointing` | true | saves activation memory at ~30% compute cost |
| `--deepspeed` | zero2 | zero3 + LoRA had device-mismatch errors with multimodal |
| `--logging_steps` | 5 | reasonable for ~90-step run |
| `--save_steps / --eval_steps` | 50 | only save once mid-train + final |
| `--save_total_limit` | 3 | disk budget |

**Yielded:** `ttcc_sft/v0-.../checkpoint-90` (1 epoch, 90 steps, train loss 0.6673 → produced $\text{IBS} \approx 0.0094$, calibration slope $\approx 1.0$).

---

## 2. SFT-noCoT (`sft_nocot.sh`) — RUNNING NOW

**Goal:** ablate the CoT contribution. Identical hyperparams to base SFT, only the dataset differs (`ttcc_swift_nocot/ttcc_train_sft.jsonl` — assistant target = `Curve: {"R": [...]}` only).

Differences from base SFT:
| param | value | note |
|---|---|---|
| `--dataset` | `ttcc_swift_nocot/ttcc_train_sft.jsonl` | no-CoT target |
| `FPS_MAX_FRAMES=32` | unchanged | apples-to-apples |

**Expected outcome:** if IBS is similar to base SFT, CoT is not load-bearing — the model learns the curve from numeric tokens alone. If IBS is worse, the Content/Drops/Reasoning preamble adds signal.

**Assumption being tested:** that CoT improves prediction quality (the implicit premise of CoT distillation). If false, the cot_distill step can be skipped entirely.

---

## 3. SFT-Extended (`sft_extended.sh`) — pending

| param | value | difference from base SFT |
|---|---|---|
| `--num_train_epochs` | 3 | from 1 → 3 to test saturation |
| `--save_steps` | 90 | every epoch instead of every 50 steps |
| `--save_total_limit` | 4 | keep 3 epoch boundaries + final |
| `FPS_MAX_FRAMES` | **24** (just lowered from 32) | match inference budget, save wall time |

**Other params unchanged**, including `--learning_rate 1e-4` and the cosine
schedule. Cosine over 270 steps will decay slower per-step than over 90,
so effective late-stage LR is similar to base SFT.

**Assumption to audit:** that 3 epochs is enough to test saturation. Base
SFT loss was 0.59 at step 90 with token_acc 0.80 — still descending. If
extended SFT plateaus by epoch 2 we have saturation; otherwise we need
more epochs.

---

## 4. GRPO base (`grpo.sh`) — already ran (28% of an epoch ≈ 50 steps)

| param | value | rationale / assumption |
|---|---|---|
| `--rlhf_type grpo` | GRPO | RL flavor: group-relative policy optimization |
| `--model` | Qwen2.5-Omni-3B + SFT adapter | warm-start from SFT |
| `--reward_funcs` | `ttcc_ibs_reward ttcc_format` | composite reward |
| `--reward_weights` | `1.0 0.2` | IBS dominates, format is a small bonus |
| `--num_generations` | **2** | very small group (typical GRPO uses 4–8). chose 2 because rollout cost is high (vLLM colocate + 30s videos) — **may hurt advantage estimate quality** |
| `--temperature` | 0.4 | low for GRPO rollouts; reduces variance but limits exploration |
| `--top_p` | 0.95 | nucleus sampling; standard |
| `--beta` | 0.04 | KL coefficient against SFT reference; default in swift examples |
| `--max_completion_length` | 384 | response budget; CoT+curve typically ~250 tokens, 384 is comfortable |
| `--max_length` | 8192 | same as SFT |
| `--max_pixels` | 49152 | same |
| `FPS_MAX_FRAMES` | 24 | tighter than SFT 32 — vLLM needs prompt budget for video tokens |
| `--use_vllm` true | yes | rollout speed |
| `--vllm_mode colocate` | colocate | shares GPU with policy; ~35% mem util |
| `--vllm_gpu_memory_utilization` | 0.35 | leaves 65% for policy gradient |
| `--per_device_train_batch_size` | 1 | OOM otherwise |
| `--gradient_accumulation_steps` | 4 | effective batch 4×2 = 8 prompts per step |
| `--learning_rate` | 5e-6 | 20× lower than SFT — RL with KL anchor needs gentle updates |
| `--warmup_ratio` | 0.05 | same |
| `--logging_steps` | 2 | more frequent than SFT (RL has more interesting per-step signal) |
| `--save_steps / --eval_steps` | 25 | save every 25 steps so we have intermediate checkpoints |

**Reward $r = 1 - \text{IBS}$**, clipped to $[0, \infty)$:
- `ttcc_ibs_plugin.py:18-24` — `rewards.append(max(0.0, 1.0 - ibs))`
- Why clip to $\ge 0$? An $\text{IBS} > 1$ prediction is degenerate (e.g. constant $0$ against a curve that starts at $1$); we don't want to compound the gradient signal with arbitrarily negative rewards. **Assumption: clipping doesn't bias the policy because $\text{IBS} > 1$ happens only for malformed parses, where we already give $0$.**
- Format reward (`ttcc_format_plugin.py`): $+1$ if the completion contains a parseable `{"R": [...]}` somewhere, else $0$. Weighted $0.2$ → up to $+0.2$ added to the IBS reward.

**Yielded:** GRPO-50 beats base SFT by paired BCa $\Delta\text{IBS} = -0.0011$, CI $[-0.0029,\,-0.0001]$ (excludes 0, $n=87$, $B=10{,}000$).

---

## 5. GRPO-Extended (`grpo_extended.sh`) — pending

Differences from base GRPO:
| param | value | note |
|---|---|---|
| `--adapters` | SFT-Extended checkpoint | start from longer-trained SFT, not base SFT |
| `--num_train_epochs` | 1 | full single epoch = 179 steps (vs ~50 we ran before) |
| `--save_steps` | 25 | unchanged |

**Other params unchanged** — same `num_generations=2`, `temperature=0.4`,
`beta=0.04`, `lr=5e-6`.

**Assumption to audit:** that more GRPO steps continue to improve IBS
without $\sigma_g$ (reward std across rollouts) collapsing. Original 50-step run
showed std oscillating but not collapsing. Extension to 179 steps is a
$\sim 3.6\times$ extrapolation; we don't *know* it stays healthy.

---

## 6. Inference (`scripts/infer_trained.sh`)

| param | value | note |
|---|---|---|
| `--infer_backend vllm` | vLLM | fast batch inference |
| `--max_new_tokens` | 1024 | generous — longest output observed ≈ 700 tokens for 60s ads |
| `--temperature` | 0.0 | **greedy** — deterministic eval. Earlier sanity check showed temp 0.4 gave paired $\Delta\text{IBS} = +0.0015$ vs temp 0.0 with CI containing 0 (i.e. consistent) |
| `--top_p` | 1.0 | unused at temp 0 |
| `FPS_MAX_FRAMES` | 24 | matches GRPO; this is what we evaluate against |
| `--max_pixels` | 49152 | same |

**Assumption:** greedy = "best possible single guess." For a proper-scoring-rule metric (IBS), the policy mean would be ideal, but greedy is the standard inference for LMs and is what milestone reports use.

---

## 7. Eval protocol (`ttcc-eval/docs/07_proper_scoring_rule_revision.md`)

Headline metric: **IBS** (Brier 1950; Graf 1999 for survival; Sonabend 2024 for RSBS training).

$$
\text{IBS}_i = \frac{1}{T_i + 1} \sum_{t=0}^{T_i} \big(\hat R_i(t) - R_i(t)\big)^2
$$

Aggregate: $\overline{\text{IBS}} = \frac{1}{n}\sum_{i=1}^{n} \text{IBS}_i$ (unweighted).

Diagnostic: **calibration slope** — $\text{slope}\big(\text{regress } R_{\text{true}} \text{ on } \hat R_{\text{hat}}\big)$, ideal $= 1.0$.

Significance: **paired BCa bootstrap** (Efron 1987), $B = 10{,}000$, $n = 87$ ads.

**Assumptions baked in:**
1. We use the **un-weighted** mean across ads. Alternative: weight by $T_i$ (longer ads contribute more datapoints). Currently no.
2. IBS treats every timestep equally. Alternative: time-weighted IBS (Hartman 2023). Currently no.
3. We do **not** use a censoring-aware version — every test ad has full $T_i$ seconds of data; no right-censoring. Safe.
4. Bootstrap is paired on `ad_id`, $B = 10{,}000$. Bias-corrected accelerated (BCa) intervals.

---

## 8. Parser (`src/ttcc_rl/parser.py`)

How we extract $\hat R$ from model text output:
1. Pass 1: balanced JSON `{"R": [...]}` — find the deepest balanced match.
2. Pass 2: fallback to bare `R = [...]` / `R: [...]` / `"R": [...]`.
3. Always: pad/truncate to $T+1$, force $R[0] = 1.0$, enforce monotone non-increasing by cumulative $\min$, clip to $[0, 1]$.

**Assumption:** if model produces nothing parseable, we record `None` and the ad gets dropped from the eval (not padded to all-1.0 which would game IBS). At test time on base SFT, 0 ads have parse failures.

---

## 9. Known limitations / open audit questions

1. **`num_generations = 2` is unusually small** for GRPO. Real-world recipes use 4–16. We chose 2 for compute. Worth re-running with 4 if budget allows.
2. **Frame budget asymmetry** — SFT base trained at 32 frames, inference and GRPO at 24. After the change we just made, SFT-Extended will also be at 24, so the chain is consistent going forward. Original SFT was trained on slightly more frames than it's evaluated on.
3. **No held-out validation set** within train. Train = 717, test = 87. Hyperparams chosen by training-data inspection + early-run intuition, not val-set tuning. Risk of overfitting eval-time choices to test set.
4. **Only one seed per method.** Variance across seeds unknown.
5. **CoT distillation budget** — teacher Qwen3-Omni was run once at temp 0 (greedy), so CoTs are deterministic; we didn't sample multiple CoTs per ad nor pick the best.
6. **The "no-CoT" SFT target preserves the system prompt but strips the analysis structure from the *user prompt*.** This means student sees a different task description, not just a different target. A cleaner ablation would keep the user prompt identical and only strip the assistant CoT. (Currently this is what `prepare_dataset_nocot.py` does.)

---

## Snapshot of pending/running

- **SFT-noCoT** (sft_nocot.sh): DONE. **66/87 parsed** (vs 87/87 for base SFT). $\text{IBS} = 0.0115$ on the 66; paired $\Delta\text{IBS}$ vs base SFT $= +0.0010$, CI $[-0.0002,\,+0.0022]$ (tied). Headline: CoT scaffolding is load-bearing for **format adherence** (21 ads lost to unparseable output), not for the numbers themselves.
- **SFT-Extended** (sft_extended.sh): DONE. Three ckpts evaluated; **ckpt-270 (epoch 3, $\text{IBS} = 0.0093$) ties base SFT** at $\Delta\text{IBS} = -0.0001$, CI $[-0.0012,\,+0.0010]$. ckpt-90/180 lost to base SFT — `FPS_MAX_FRAMES=24` penalty needed 3 epochs to overcome. Training loss saturated around epoch 2 (0.530) and bounced up slightly at epoch 3 (0.538) but IBS continued improving.
- **GRPO-Extended** (grpo_extended.sh): RUNNING from sft_ext ckpt-270 (179 steps full-epoch ≈ 2-3 hr).
