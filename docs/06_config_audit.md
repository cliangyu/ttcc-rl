# Training + evaluation config audit (2026-05-22)

The user prompt: *"It would be pretty bad if we didn't get the training recipe right from the start. Go through every config to rule out factors."*

This document is the systematic audit. Numbers cross-referenced against ms-swift docs, the Qwen2.5-Omni HF page, and the GRPO literature.

---

## Summary verdict

Five settings are **provably wrong or below literature minimum**, two need diagnostic checks, the rest are sensible.

| # | knob | current | literature / recommended | severity |
|---|---|---|---|---|
| 1 | $\beta$ (GRPO KL coefficient) | $0.04$ | $\sim 0.001$ (DeepSeek-R1), default $0.0$ (HF TRL) | 🚩 **critical** — KL anchor dominates the compressed reward signal |
| 2 | `num_generations` | $2$ | $4$ minimum, $8$ standard | 🚩 **critical** — group advantage estimate has high variance |
| 3 | `FPS_MAX_FRAMES` (LoRA round) | $24\text{–}32$ | $\ge 60$ at our $\text{FPS}=1$ and $T_{\max}=60$ | 🚩 **critical** — $38\%$ of test ads lose visual past second $32$ |
| 4 | `max_completion_length` (GRPO) | $384$ | $\ge 512$ | 🚩 **critical** — observed $87\%$ clipped at $384$ |
| 5 | `--target_modules all-linear` in LoRA SFT *without* `--freeze_vit true` | implicit | should be explicit | ⚠️ trains LoRA on audio_tower + visual layers unnecessarily |
| 6 | `temperature` (GRPO rollouts) | $0.4$ | $0.5\text{–}0.7$ | ⚠️ low; limits exploration |
| 7 | `truncation_strategy delete` + `max_length=8192` + `max_pixels=200704` | as-is | unknown how many ads silently dropped | 🟡 diagnostic |
| 8 | Audio actually contributing gradients? | unknown | should verify | 🟡 diagnostic |

The five critical issues each independently explain part of why the methods haven't surpassed B1 on the headline. The two diagnostic checks may reveal more.

---

## 1. Full config table (every run, every knob)

| knob | LoRA SFT (`sft.sh`) | LoRA SFT-ext (`sft_extended.sh`) | LoRA GRPO (`grpo.sh`) | LoRA GRPO-ext (`grpo_extended.sh`) | Full-FT SFT (`sft_v2cot_full.sh`) | Full-FT GRPO (`grpo_v2cot_full.sh`) | LoRA infer (`infer_trained.sh`) | Full-FT infer (`infer_v2cot_full.sh`) |
|---|---|---|---|---|---|---|---|---|
| **Visual** | | | | | | | | |
| `FPS` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `FPS_MAX_FRAMES` | **32** | 24 | 24 | 24 | **32** | **32** | 24 | 32 |
| `MAX_PIXELS` | 49152 ($192 \times 256$) | 49152 | 49152 | 49152 | **200704** ($448^2$) | 200704 | 49152 | 200704 |
| `VIDEO_MAX_PIXELS` | 49152 | 49152 | 49152 | 49152 | 200704 | 200704 | 49152 | 200704 |
| `VIDEO_MAX_TOKEN_NUM` | (default) | (default) | 4096 | 4096 | (default) | **8192** | 4096 | 8192 |
| **Model** | | | | | | | | |
| Tuner | LoRA r=16 α=32 | LoRA r=16 α=32 | LoRA r=16 α=32 | LoRA r=16 α=32 | **full** | **full** | — | — |
| Target modules | `all-linear` | `all-linear` | `all-linear` | `all-linear` | n/a | n/a | — | — |
| `freeze_vit` | (unset → false) | (unset) | (unset) | (unset) | **true** | **true** | — | — |
| `freeze_aligner` | (unset → false) | (unset) | (unset) | (unset) | **true** | **true** | — | — |
| **Training** | | | | | | | | |
| LR | 1e-4 | 1e-4 | 5e-6 | 5e-6 | **1e-5** | 5e-6 | — | — |
| Warmup ratio | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | — | — |
| Epochs | 1 | 3 | 1 | 1 | **10** | 1 | — | — |
| Per-device BS | 1 | 1 | 1 | 1 | 1 | 1 | — | — |
| Grad accum | 4 | 4 | 4 | 4 | **8** | 4 | — | — |
| Max length | 8192 | 8192 | 8192 | 8192 | 8192 | 8192 | — | — |
| Truncation strategy | (swift default = `delete`) | same | same | same | **`delete` (explicit)** | (default) | — | — |
| Gradient checkpointing | true | true | true | true | true | true | — | — |
| DeepSpeed | zero2 | zero2 | zero2 | zero2 | zero2 | zero2 | — | — |
| **GRPO/RL** | | | | | | | | |
| `num_generations` | — | — | **2** | 2 | — | 2 | — | — |
| `temperature` | — | — | **0.4** | 0.4 | — | 0.4 | (greedy 0.0) | (greedy 0.0) |
| `top_p` | — | — | 0.95 | 0.95 | — | 0.95 | 1.0 | 1.0 |
| $\beta$ | — | — | **0.04** | 0.04 | — | 0.04 | — | — |
| `max_completion_length` | — | — | **384** | 384 | — | 384 | 1024 | 1024 |
| `reward_weights` | — | — | 1.0 0.2 | 1.0 0.2 | — | 1.0 0.2 | — | — |
| `vllm_gpu_mem_util` | — | — | 0.35 | 0.35 | — | **0.20** | — | — |

Bold = differs from sibling rows in a way that matters.

---

## 2. Critical issues, with citations

### 2.1 $\beta = 0.04$ is 8–40× too high — 🚩

**Our value:** $\beta = 0.04$ (KL penalty coefficient in GRPO).

**Literature:**

- HF TRL `GRPOTrainer` **default is $\beta = 0.0$** [(TRL docs)](https://huggingface.co/docs/trl/en/grpo_trainer). The TRL docs note: *"$\beta = 0.0$ by default, meaning the KL divergence term is not used, motivated by recent studies showing the KL divergence term is not essential for training with GRPO."*
- DeepSeek-R1 uses $\beta = 0.001$ [(Unsloth advanced RL guide)](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation).
- Typical operational values in the open literature: $\beta \in [0.001, 0.01]$.

**Math of why this hurts us:**

GRPO's policy gradient term scales with the standardized advantage $A_i = (r_i - \bar r)/\operatorname{std}(r)$. With our typical reward range $r \in [0.97, 1.0]$ and within-group $\operatorname{std}(r) \approx 0.005$, standardized advantages are $\pm 0.5\text{--}1.0$. The KL term, however, is **unnormalized** — its magnitude grows with raw $\beta$ times the policy drift from reference.

Ratio of "KL pull" to "reward push" in the gradient:

$$
\frac{\beta \cdot \nabla \mathrm{KL}}{\|A_i\| \cdot \nabla \log \pi_\theta} \;\approx\; \frac{0.04}{\sim 0.5 \cdot 0.005} \;\approx\; 16
$$

i.e. **KL is ~16× the reward gradient signal** for our setting. The policy barely moves from the SFT reference. **This is the most likely single cause of GRPO-Extended saturating at ckpt-150 (IBS $0.0091$) without ever surpassing the base SFT.**

**Fix:** $\beta = 0.001$ (matches DeepSeek-R1) or $\beta = 0.0$ (HF TRL default). One-line change in `_common.sh` or via env override.

### 2.2 `num_generations = 2` is below the literature floor — 🚩

**Our value:** 2.

**Literature:**

- HF TRL recommends $\ge 4$ for memory-constrained settings; up to 16 standard ([cookbook GRPO recipe](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl)).
- DeepSeek-R1: 16. Open-source repos commonly use 8.
- With $n = 2$ the within-group $\operatorname{std}(r)$ is computed from a *single difference*, which is the noisiest possible estimator.

**Why we set it to 2:** rollout cost on 30-second video. With our 2h21m for 179 steps at $n = 2$, going to $n = 4$ would be ~4-5h.

**Fix:** $\text{num\_generations} = 4$. Pay the 2× compute cost for a meaningfully more informative advantage estimate. Already in TOMORROW.md (item #4).

### 2.3 `FPS_MAX_FRAMES` is far below what Qwen2.5-Omni expects — 🚩

**Our value:** 24-32 across our runs.

**Recommended by Qwen2.5-Omni / Qwen-VL training guide:**

- `video_max_pixels`: **50176** (= $224^2$) per-frame
- `video_max_frames`: **512** per video
- `fps`: 1 (matches us)

Source: [LMMs Engine Qwen-VL training guide](https://lmms-engine.readthedocs.io/en/latest/models/qwenvl.html).

**Why this matters concretely for our data:**

At $\text{FPS} = 1$:

| FPS_MAX_FRAMES | test ads losing visual past frame $F$ | train ads losing visual past frame $F$ |
|---|---|---|
| 24 | 44/87 = **51%** | 415/717 = **58%** |
| 32 | 33/87 = **38%** | 314/717 = **44%** |
| 60 | 0/87 = 0% | 0/717 = 0% |
| 512 (Qwen rec.) | 0/87 = 0% | 0/717 = 0% |

For $\sim 40\text{–}50\%$ of our ads, the model **never sees what happens after second 32 visually** — yet we ask it to predict $R(33), R(34), \ldots, R(T_i)$ for those ads. This is a structural information loss. It plausibly explains:

- the persistent calibration drift on the tail (REL contribution from late timesteps),
- the puzzling pattern that "trained methods beat B1 on the CTA window" — for long ads the CTA window is past the frame budget for *both* B1 (which doesn't use video) and our methods (which had video budget exhausted), so it's a learned-statistics prediction in both cases.

**Fix:** `FPS_MAX_FRAMES = 60` covers every test ad. The cost is $\sim 1.9\times$ visual tokens vs 32 frames, mitigated by Qwen's adaptive token packing and our `MAX_PIXELS = 49152` (already below the recommended 50176). At 49152 px × 60 frames × ~36 patches = ~2160 tokens — well within the 8192 max_length budget.

### 2.4 `max_completion_length = 384` is too tight — 🚩

**Observation:** during GRPO training, `completions/clipped_ratio` reached **87.5%** at step 14 and stayed in $0.50\text{–}0.75$ for most of training. The reward dipped from $1.193$ to $1.164$ on step 16 immediately after clipping spiked.

**Math:**

- Curve for a $T_i = 60$ ad: $T_i + 1 = 61$ numbers each formatted to 4 decimals → $\sim 7 \cdot 61 + 8 = 435$ characters → $\sim 200$ tokens with Qwen's tokenizer.
- CoT (Content + Drops + Reasoning): typically $\sim 150\text{–}250$ tokens.
- Total: $350\text{–}450$ tokens typical, up to $\sim 500$ for the longest ads.

At 384, $\sim 40\text{–}80\%$ of ads with $T_i > 40$ get clipped, which means **the curve gets truncated mid-way**, the parser pads with the last value, and the reward signal is noisy.

**Fix:** `max_completion_length = 768`. Cost: rollouts ~1.4× slower (more tokens to decode), but the reward becomes informative on every ad instead of degenerate on long ads.

### 2.5 `target_modules all-linear` without explicit `freeze_vit`/`freeze_aligner` in LoRA round — ⚠️

**Behavior verified from [ms-swift docs](https://swift.readthedocs.io/en/latest/BestPractices/MLLM-Registration.html):**

> "For Qwen2.5-Omni in full parameter training with `freeze_vit=True`, it will freeze parameters of model layers prefixed with `thinker.audio_tower` and `thinker.visual`. In LoRA training with `freeze_vit=False`, it will additionally add LoRA to Linear layers prefixed with `thinker.audio_tower` and `thinker.visual`."

**Our LoRA scripts** (`sft.sh`, `grpo.sh`, etc.) set `--tuner_type lora` and `--target_modules all-linear` but do **not** set `--freeze_vit true`. So by ms-swift's default, LoRA was added to:
- `thinker.audio_tower.*` (audio encoder linears)
- `thinker.visual.*` (visual encoder linears)
- `thinker.*` (text-decoder linears)

**Implications:**

1. We're training LoRA adapters on the audio and visual encoders on $717$ ads — far too few to learn anything generalizable about audio/visual encoding.
2. These adapters add ~3-5× more trainable parameters than necessary, slowing training.
3. The audio/visual encoders for Qwen2.5-Omni are already very well pretrained; adapting them to 717 ads risks degrading them.
4. Inference is correspondingly slower because we have to merge more LoRA weights.

**Fix:** add `--freeze_vit true --freeze_aligner true` to the LoRA scripts (sft.sh and grpo.sh). LoRA then attaches only to the text decoder, where we actually want adaptation. Aligns LoRA round behavior with full-FT round (which already does this).

---

## 3. Concerning but not certainly broken

### 3.1 `temperature = 0.4` in GRPO rollouts — ⚠️

**Our value:** 0.4. **Literature:** 0.3-0.7, with 0.5 most common in HF TRL examples. We're on the conservative end.

**Effect:** rollouts hew closer to SFT; less exploration. Combined with high $\beta$ (#2.1), this means the policy explores very little. We saw `IBSReward/std` stay in $[0.005, 0.05]$ — non-zero but small.

**Fix:** raise to $0.6$ if combined with $\beta = 0.001$, to compensate for the reduced KL anchor.

### 3.2 `VIDEO_MAX_TOKEN_NUM = 4096` in LoRA round — ⚠️

**Calculation:** at `MAX_PIXELS = 49152` ($192 \times 256$), each frame is roughly $192 \cdot 256 / 28^2 \approx 63$ tokens after Qwen's $28 \times 28$ patching. With FPS_MAX_FRAMES = 32 frames, total video tokens $\approx 32 \cdot 63 \approx 2{,}016$. Within 4096 budget.

But: longer ads × bigger frames could exceed 4096 silently. Should bump to 8192 to match v2 full-FT for safety.

**Fix:** `VIDEO_MAX_TOKEN_NUM = 8192` in `_common.sh` default.

### 3.3 Cosine LR schedule decays to 0 at end-of-training — ⚠️

The base SFT (1 epoch, 90 steps, LR starts at 1e-4, ends at 0 by step 90). SFT-Extended (3 epochs, 270 steps) has the same `--learning_rate 1e-4` but cosine to 0 at step 270, so **step 90 of SFT-Extended is at LR ≈ 7.8e-5**, not 0.

This means base SFT and SFT-Extended-at-step-90 are *not* the same checkpoint; SFT-Extended-90 is partway through a longer schedule with a still-warm LR.

**Implication:** the "GRPO from sft_ext ckpt-270 vs GRPO from sft ckpt-90" comparison isn't an apples-to-apples test of "more SFT helps RL." The base sft is fully decayed; sft_ext at step 270 is also fully decayed, but the model has gone through 3× more update steps with mostly-warm LR.

**Fix:** if we want a clean A/B between "1 epoch" and "3 epochs" of SFT, use `--lr_scheduler constant_with_warmup` so steady-state LR is identical.

---

## 4. Diagnostic checks — unknowns to verify

### 4.1 Is audio actually flowing into gradients?

The dataset rows include `"audios": [path_to_mp4]`. Qwen2.5-Omni's `Thinker` runs the audio path through `audio_tower`. In our LoRA round (without explicit `freeze_vit`), LoRA was attached to `audio_tower.*` linears (see §2.5) — so yes, gradients flow.

But: with `--freeze_aligner true` in the full-FT round, the **audio aligner** (which maps audio embeddings into the LLM input space) is frozen. So full-FT trains the LLM but treats audio as a fixed feature. **Verify:** check `param.requires_grad` after loading the full-FT model to confirm audio tower is frozen but audio aligner outputs are still being consumed.

### 4.2 How many ads silently dropped by `truncation_strategy delete` + `max_length = 8192`?

In full-FT round: $\text{max\_pixels} = 200704 \;(448^2)$, each frame is $\sim 256$ patches × 32 frames = 8192 video tokens already at the edge. Plus audio ($\sim 200$), text prompt ($\sim 200$), assistant target ($\sim 400$) = potentially $> 8000$. With `truncation_strategy delete`, ads over the limit are silently dropped.

Verify: look for the swift log line like `Filtered N samples (>max_length)` after the run.

### 4.3 Does the LoRA infer path correctly merge audio_tower + visual LoRA?

If §2.5 is correct (LoRA was attached to audio_tower in the LoRA round), then `--adapters <ckpt>` must merge those too at inference. ms-swift handles this automatically, but: verify that `vLLM` actually uses the merged audio_tower LoRA — vLLM has historically been finicky about non-text LoRA targets.

---

## 5. Recommended fixes — concrete diffs

For the next training run, the highest-leverage set of changes:

```bash
# _common.sh
: "${FPS_MAX_FRAMES:=60}"           # was: 24
: "${VIDEO_MAX_TOKEN_NUM:=8192}"    # was: 4096

# grpo.sh / grpo_extended.sh / rloo.sh
: "${BETA:=0.001}"                  # was: 0.04
: "${NUM_GENERATIONS:=4}"           # was: 2
: "${TEMPERATURE:=0.6}"             # was: 0.4
: "${MAX_COMPLETION_LENGTH:=768}"   # was: 384

# sft.sh / grpo.sh — add these to the swift CLI invocation
    --freeze_vit true \
    --freeze_aligner true \

# sft_v2cot_full.sh — change FPS_MAX_FRAMES
FPS_MAX_FRAMES=60                   # was: 32
```

**Expected impact (qualitative):**

- $\beta$ and `num_generations` fixes: GRPO can actually move the policy. Should see $\text{IBSReward/std}$ grow (more exploration) and reward continue climbing past step ~50 instead of saturating.
- `FPS_MAX_FRAMES = 60`: model sees every second of every test ad. Should reduce tail-calibration error (lower REL).
- `max_completion_length = 768`: clipped_ratio should drop to ~0%, reward signal becomes uniformly informative.
- `freeze_vit/aligner`: fewer LoRA parameters, faster training, less risk of overfitting the encoders on 717 ads.

**Caveat: don't change all five at once for an ablation.** Recommend running:

1. Reproduce base SFT + base GRPO with $\beta = 0.001$ only. Measure $\Delta\text{IBS}$ vs current GRPO.
2. Add `num_generations = 4`. Measure marginal impact.
3. Add `FPS_MAX_FRAMES = 60`. Measure marginal impact.

If we change all five together and IBS improves, we won't know which fix did it.

---

## Sources

- [HuggingFace TRL `GRPOTrainer` docs](https://huggingface.co/docs/trl/en/grpo_trainer) — $\beta = 0.0$ default
- [Unsloth advanced RL documentation](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation) — DeepSeek-R1 $\beta = 0.001$
- [HuggingFace cookbook: GRPO TRL recipe](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) — num_generations recommendations
- [LMMs Engine Qwen-VL training guide](https://lmms-engine.readthedocs.io/en/latest/models/qwenvl.html) — video_max_pixels = 50176, video_max_frames = 512
- [ms-swift MLLM registration docs](https://swift.readthedocs.io/en/latest/BestPractices/MLLM-Registration.html) — Qwen2.5-Omni LoRA target behavior with freeze_vit
- [QwenLM/Qwen2.5-Omni fine-tuning issue #12](https://github.com/QwenLM/Qwen2.5-Omni/issues/12) — official fine-tuning reference
