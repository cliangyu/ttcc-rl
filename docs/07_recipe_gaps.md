# Recipe gaps — what we haven't checked or what we're doing wrong

Follow-up to `docs/06_config_audit.md`. That doc covered the obvious training knobs (β, num_generations, FPS, max_completion, freeze_vit). This one covers everything else that could be silently degrading our recipe or biasing our reported numbers.

Audited against ms-swift `args.json` (authoritative record of what training actually used), the base Qwen2.5-Omni-3B config, the training scripts, and the data prep scripts.

---

## Verified-correct (so we can stop worrying about these)

- **Train/test split:** 0 video overlap between `ttcc_train_sft.jsonl` (717 ads) and `ttcc_test.jsonl` (87 ads). ✓
- **`max_grad_norm = 1.0`** (ms-swift default, recorded in args.json). Stable training. ✓
- **`seed = 42, data_seed = 42`** pinned. Runs reproducible. ✓
- **Optimizer:** `adamw_torch_fused`, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight_decay $= 0.1$. Matches Qwen's published values. ✓
- **LoRA target_modules regex:** matches `thinker.model.*` only (text decoder). Audio_tower + visual encoder never had LoRA attached. ✓
- **Audio gradient flow:** verified yesterday — audio flows forward through frozen audio_tower into the LLM; encoder weights stay pretrained (good for our 717-ad scale). ✓

---

## Gaps that probably matter — prioritized

### 1. 🚩 Talker + token2wav are loaded but unused

**Finding:** `/home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B/config.json` has `enable_talker: True, enable_audio_output: True`. Param breakdown of the full-FT checkpoint we already have:

| sub-module | scalar params | % of model | used for our task? |
|---|---|---|---|
| `thinker.audio_tower` | 638 M | 11.5% | yes (forward features into LLM) |
| `thinker.visual` | 669 M | 12.1% | yes (forward features into LLM) |
| `thinker.model` (LLM) | 3.09 B | 55.7% | yes (output) |
| `thinker.lm_head` | 311 M | 5.6% | yes (output) |
| **`talker.*`** | **385 M** | **7.0%** | **NO — speech generation** |
| **`token2wav.*`** | **449 M** | **8.1%** | **NO — vocoder** |

So **15% of the model (833 M params, ≈ 1.5 GB at bf16) is loaded into GPU memory but never used**. This likely contributed to the CUDA OOM we hit on the first full-FT GRPO attempt (which we worked around by lowering `vllm_gpu_memory_utilization` from 0.35 to 0.20).

**Fix:** load thinker-only. ms-swift has `model_type=qwen2_5_omni_thinker` for this exact case. Or override the config at load time (`enable_talker=False, enable_audio_output=False`) before swift instantiates the model. **Verify which mechanism actually works** before next training run.

**Expected impact:** ~1.5 GB GPU memory saved per device → we can raise `vllm_gpu_memory_utilization` back to 0.35-0.40 (more rollout budget), possibly fit larger `num_generations`, possibly use larger frame budgets.

### 2. 🚩 No validation set — we've been doing model selection on the test set

**Finding:** we have only `ttcc_train_sft.jsonl` (717) and `ttcc_test.jsonl` (87). No `ttcc_val.jsonl`.

But across yesterday's experiments we:
- evaluated SFT vs SFT-Extended ckpt-90 vs ckpt-180 vs ckpt-270 → picked ckpt-270 as "best" by test IBS
- evaluated GRPO-Extended ckpts 25/100/125/150/175/179 → reported ckpt-150 as best by test IBS

**This is test-set leakage by model selection.** The reported "best" numbers are biased upward (in the model's favor) because we explicitly searched over checkpoints using test feedback.

**Fix:** carve a $\sim 60$-ad val split from the 717 training ads, retrain, select checkpoints by val IBS, evaluate **once** on test for the headline. Cost: 1× training run + redoing analysis on the reduced train set. Some win-loss numbers may shift.

**Why this matters most:** if we ever want to claim "GRPO beats SFT" in writing, the comparison must be on a held-out test we did NOT use for selection. Currently we cannot make that claim defensibly.

### 3. 🟡 CoT quality never audited

**Finding:** the teacher (Qwen3-Omni-30B-A3B-Instruct) was run once at temp=0, outputs trusted. We never sampled CoTs to verify the teacher is grounding its "Drops at second X" claims in real on-screen events vs hallucinating.

**Risk:** if teacher hallucinates `Drops: retention falls at second 17 when the product appears` when nothing happens at second 17, the student learns to associate fabricated events with fabricated drops. This would explain why SFT methods are well-calibrated *in aggregate* (RES preserved) but **add calibration error vs B1** in the Murphy decomposition: the model learns the marginal distribution of "where drops typically happen" rather than per-ad content.

**Cheap audit:** sample 10 ads, watch the videos with the GT curve and the teacher CoT side by side. Cost: ~30 minutes of researcher time. High insight density.

### 4. 🟡 Test set $n = 87$ is borderline for the BCa CIs we're reporting

**Finding:** Bradley 2008 (we cited it in `06_metrics_feynman.md` §4a) recommends "a few hundred" pairs for reliable Brier-skill significance. Our paired BCa CI widths are $\pm 0.002\text{-}0.004$ — borderline for $\Delta\text{IBS}$ values in the $\pm 0.001$ range.

A pointer from `ttcc-eval/docs/06_drop_logic.md`: we previously reported "cap-at-60 instead of drop recovered 16% of corpus". Has that been applied to the test set too? Verify the test-set drop logic and see if the test n can grow to ~120 by relaxing borderline drops.

**Fix (cheap):** recompute test set with the relaxed-drop rules. If recovered ads have valid curves, $n \to \sim 100\text{-}120$ tightens all our CIs by $\sim 1.1\times$.

### 5. 🟡 Only one seed per method

**Finding:** every reported number is from a single seed=42 run.

**Risk:** the GRPO-50 vs base SFT win (paired BCa $\Delta\text{IBS}$ CI excluded 0) was on one seed. A second seed could easily flip the conclusion if our $n = 87$ is borderline.

**Fix:** run 2 additional seeds of base GRPO with the new config (β=0.001, num_gens=4). Cost: ~1 hour per seed. Report mean ± std across 3 seeds.

### 6. 🟡 Parse-failure bias on SFT-noCoT

**Finding:** SFT-noCoT reports IBS over only 66/87 ads because 21 ads had unparseable outputs. The 66 "successfully parsed" ads are not a random subset — they're correlated with ad characteristics that make parseable output easier (likely shorter ads, simpler content).

**Implication:** the comparison "SFT-noCoT IBS = 0.0115 on 66 ads vs base SFT IBS = 0.0094 on 87 ads" mixes two effects: (a) actual prediction quality, (b) which subset of ads got measured.

**Honest treatment:** report SFT-noCoT IBS as `0.0115 on the 66/87 it could parse; 21 ads silently degraded to no prediction`. If we want a single comparable number, define a default prediction for parse failures (B1's curve is a reasonable choice — at least nondegenerate) and recompute SFT-noCoT IBS over all 87 with that fallback.

### 7. 🟡 Cosine LR schedule confound across runs

**Already in `06_config_audit.md` §3.3**, repeating for completeness because it interacts with item #2:

- Base SFT: cosine 1e-4 → 0 over 90 steps
- SFT-Extended: cosine 1e-4 → 0 over 270 steps

So at the same global step 90, SFT-Extended has LR ≈ 7.8e-5 (still warm) while base SFT has LR = 0 (cosine bottom). They're not the same checkpoint at step 90 even if data + model + seed are identical. The "more SFT helps RL" comparison is confounded.

**Fix:** use `--lr_scheduler constant_with_warmup` or save ckpt at matched LR across runs.

---

## Diagnostic gaps (already in `TOMORROW.md` §11)

These are concrete checks that could change our interpretation of past results:

- **§11.1 Audio actually used.** Forward path verified; whether the LLM attends to it is unverified. Run inference with audio replaced by silence and compare IBS. **Yesterday I described the ablation; needs to be run.**
- **§11.2 Truncation drops.** With `--truncation_strategy delete` + `max_length=8192` + the new `FPS_MAX_FRAMES=60` and `MAX_PIXELS=200704`, how many ads silently drop in the v2 full-FT round? If more than $\sim 5\%$, we have a coverage problem.
- **§11.3 vLLM LoRA merge for old checkpoints.** Old LoRA targets (text decoder only, no audio_tower) — vLLM should merge cleanly. New runs are full-FT so this stops mattering after we cut the LoRA path.

---

## Things I am NOT worried about (so we don't burn cycles)

- **Decimal precision in curves** (4 places): introduces noise floor at $\sim 10^{-7}$. Five orders of magnitude below our IBS values. Negligible.
- **Monotone enforcement in parser:** the parser fixes non-monotone outputs after extraction. So evaluated IBS reflects the post-fixed curve, which is fine — that's the prediction we'd deploy. Loss doesn't see this, but loss is token-CE on the raw text, which is what we want.
- **`prepare_dataset.py` peak normalization:** verified previously. R[0] = 1.0 by construction.
- **Tokenizer/template drift between train and infer:** swift writes `chat_template.jinja` into every checkpoint. infer uses the same.

---

## Recommended prioritization

If we have time for only one fix before the next training run: **#1 (disable talker/token2wav)**. It's the cheapest, most-mechanical, and most-impactful (saves 1.5 GB / device, may unlock larger num_generations or frame budgets).

If we have time for two: **#1 + #2 (carve val set from train)**. The val set is the only way to make our future "X beats Y" claims defensible.

If three: **#1 + #2 + #6 (parse-failure fallback for fair IBS aggregation)**. Removes a known reporting bias.

Items #3 (CoT audit), #4 (recover test ads), #5 (multi-seed) all add credibility without changing fundamentals — defer to after the next training round.

---

## Suggested sequence for next session

1. Disable talker + token2wav. Verify GPU memory drops by ~3 GB on each device.
2. Re-prep data: carve 60-ad val from train (stratified by `T_i` so val covers the full duration range).
3. Re-run base SFT with the new config (β=0.001, num_gens=4 doesn't apply to SFT, but FPS_MAX_FRAMES=60 + freeze_vit explicit + max_completion_length=1024 do).
4. Re-run base GRPO from the new SFT, with new defaults.
5. Evaluate **once** on test for the headline numbers.
6. Run the audio ablation (§11.1) on the new GRPO ckpt.

Total compute: SFT + GRPO + 2 inference runs ≈ 1 + 1.5 + 0.2 = 2.7 hours.
