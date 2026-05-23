# Session journal — 2026-05-22 afternoon

Picks up after `08_session_journal_20260522.md` (which ends at v38 completion: train_loss=0.036, eval_loss=1.82, 2h 7m wall, clean overfit signal). This doc tracks the post-v38 work: quantitative mode-collapse diagnosis (H1) and the training-config sweep (T1a–T2b) that's running as I write this.

---

## TL;DR (so future-me can skip the prose)

1. **Mode-collapse confirmed quantitatively.** v38 ckpt-80 has train_loss=0.036 but **R[1] correlation with GT ≈ 0** on both train and val. The "overfit" was memorization of a *generic decay shape* + length conditioning, not of ad-specific signal. CE token-loss and curve-MSE are weakly coupled — celebrating low train_loss was misleading.

2. **ckpt-50 (best CE eval) is the GRPO init candidate**, not ckpt-80. Marginal improvement in curve space: val per-ad MSE 0.0075 vs 0.0082 (−10% vs B1 train-mean baseline of 0.0083), but still mode-collapsed (std-across-ads ~0.04). 50× CE divergence between train and eval did NOT mean 50× quality divergence.

3. **FA3 wheel doesn't work on sm_120.** windreamer wheel ships only sm_80 + sm_90a kernels. CUDA "no kernel image" on smoke test. Compiling FA3 from source for Blackwell is risky (TMA semantics differ). Sticking with FA2.8.3.

4. **Config sweep in progress.** B0 baseline measured at 211 s/step, 77 GiB peak — matches prior session's "B" measurement. T1b (--use_liger_kernel true) launches next.

---

## Where the artifacts live (as of this writing — many in /tmp, will be moved)

| artifact | path | purpose |
|---|---|---|
| v38 ckpt-50 (best val CE) | `/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full/v35-20260522-150156/checkpoint-50/` | candidate GRPO init |
| v38 ckpt-80 (final, overfit) | `…/checkpoint-80/` | terminal overfit reference |
| v38 ckpt-80 train predictions | `/tmp/v38_preds_train.jsonl` | inference output (116 filtered train ads) |
| v38 ckpt-80 val predictions | `/tmp/v38_preds_val.jsonl` | inference output (20 filtered val ads) |
| v38 ckpt-50 train predictions | `/tmp/v38_ckpt50_preds_train.jsonl` | for H1 comparison |
| v38 ckpt-50 val predictions | `/tmp/v38_ckpt50_preds_val.jsonl` | for H1 comparison |
| v38 input data (matches above) | `/tmp/v38_input_{train,val}.jsonl` | aligned by ad_id; needed because filtered-train and filtered-val have different videos than HF dataset |
| Plots | `/tmp/v38_train.png`, `v38_val.png`, `v38_compare.png` | GT vs pred curves |
| Filter script | `/tmp/filter_long_rows.py` | row-length estimator that produced the 116/20 filtered sets |
| Plot script | `/tmp/plot_v38.py` | matplotlib grid of GT vs pred |
| Analysis script | `/tmp/analyze_v38.py`, `/tmp/analyze_h1.py` | per-ad MSE, R[1] corr, std-across-ads, mode-collapse proxy |
| Inference scripts | `/tmp/infer_v38.sh`, `/tmp/infer_v38_ckpt50.sh` | vLLM-backed inference on a JSONL input |
| Bench harness | `/tmp/bench_sft.sh` | 5-step micro-run with mem polling for config sweep |
| Bench launcher | `/tmp/launch_T1b.sh` | wrapper for T1b |
| B0 baseline log | `/tmp/bench_B0.log` | output of B0 benchmark |
| B0 output dir | `/opt/dlami/nvme/ssm-out/bench_B0_*/` | swift's args.json + bench.log + memory.csv |
| Summary doc (provisional) | `/tmp/overnight_summary.md` | v38-only TL;DR (predates this doc) |

**Cleanup note:** the /tmp files should move to `/home/ubuntu/ttcc-rl/scripts/` (or a sub-tree) and predictions / plots to a permanent `runs/` tree before the next overnight cycle. Currently scattered.

---

## H1: ckpt-50 vs ckpt-80 in curve space

### Why this experiment
v38 train_loss=0.036 was a token-CE measurement, not a curve-quality measurement. Prior project work (`04_final_report.md` §8) already established that token-CE and IBS / R[1] correlation can decouple — so we needed to actually evaluate v38 outputs in retention-curve space before deciding what to do next.

### Method
Ran vLLM inference (greedy, max_new_tokens=1024) on both filtered datasets (116 train, 20 val rows) using each checkpoint. Computed per-ad MSE, R[1] correlation with GT, std of predictions across ads (mode-collapse proxy), and curve-length match rate.

### Result table

| metric | ckpt-50 | ckpt-80 | winner |
|---|---:|---:|---|
| CE eval loss | 1.207 | 1.820 | ckpt-50 |
| val per-ad MSE (mean) | 0.0075 | 0.0082 | ckpt-50 |
| val per-ad MSE (median) | 0.0032 | 0.0043 | ckpt-50 |
| val R[1] correlation | −0.24 | −0.20 | both ≈ 0 |
| val std-across-ads | 0.040 | 0.045 | both very low (mode-collapsed) |
| length matches T+1 ±1 | 100% | 100% | both perfect |

### What this revises

I had described CE token-loss and curve-MSE as "completely decoupled, different units, can't compare." That was over-stated. They're **weakly coupled in our regime** — ckpt-50 won on both. But the coupling is small (CE differs 50×, MSE differs 10%), and **neither metric captures per-ad ranking signal** (R[1] correlation is ≈ 0 on both checkpoints). The model is producing a generic exponential decay shape that happens to be close to many ads' actual curves. Loss numbers don't expose this; only parsing the response and computing R-space metrics does.

### Decision
ckpt-50 is the GRPO init candidate. Skip ckpt-80.

### Anatomy of "token loss is digit loss" for v38

Each v38 assistant response (noCoT) is roughly `Curve: {"R": [1.0000, 0.1234, ..., 0.0103]}`. Tokenization breakdown:
- Structural prefix `Curve: {"R": [` → ~5 tokens
- ~T+1 numeric values, each ~4-5 BPE pieces (`0`, `.`, `12`, `34`) → 120-150 tokens for T=30
- `, ` separators and closer `]}` → ~30 tokens

**~85% of loss-bearing tokens are digit fragments.** Token CE *is* effectively digit CE here. But the **R[1] token is the only one carrying ad-specific signal** — all the others are predictable from autoregressive context once the model commits to a decay shape. A model that always emits `0.13` for R[1] gets near-perfect train_acc on a dataset where R[1] median is ~0.13, but R[1] correlation with GT = 0. That's what v38 is doing.

---

## Training-config sweep (T1a–T2b)

### Goal
Find the fastest config that fits the audit-ambitious recipe (FPS=1, FPS_MAX_FRAMES=60, max_pixels=200704, max_length=24576) on 2× sm_120 Blackwell. User asked for fastest config with ≥10 GB headroom.

### Hypotheses (ranked by expected speedup × confidence)

| # | Lever | Expected | Confidence | Effort | Status |
|---|---|---:|---|---|---|
| T1a | FlashAttention 3 | 1.3–1.8× | HIGH | LOW (one pip install) | **DEAD** — sm_120 unsupported |
| T1b | --use_liger_kernel true | 1.1–1.25× + 8-12 GB freed | MED | LOW | running |
| T1c | ZeRO-2 (no offload) instead of ZeRO-3 | 1.05–1.15× | MED | LOW | pending |
| T2a | vit_gradient_checkpointing=false | 1.15–1.30× | HIGH | LOW | pending (memory-gated) |
| T2b | per_device_batch_size=2 | 1.5–1.8× | MED | LOW | pending (memory-gated) |
| T3a | torch.compile | 1.1–1.2× | MED | LOW | optional |
| T3c | Ulysses SP (sp=2) | tested before, lost (halved effective batch) | — | — | known-bad |

Each test runs 5 steps on real ml=24576 audit-ambitious rows from `ttcc_swift_v2cot/ttcc_train_sft.jsonl`. Compares step-5 time + peak GPU memory to baseline.

### B0 baseline (measured)
- Config: ZeRO-3, no offload, vit_ckpt=true, FA2, bs=1×ga=8, ml=24576, FPS_MAX=60, max_pixels=200704, Liger=off
- **211.3 s/step, 77.16 GiB peak** (step 1)
- Reproduces prior session's "B" measurement.

### T1a (FA3) — negative result
windreamer's prebuilt wheel for cu130/torch2.11 only contains sm_80 + sm_90a cubins (confirmed via `strings` on `_C.abi3.so`). CUDA "no kernel image available" on smoke test. **Not viable without compiling from source for sm_120**, which is uncertain because FA3 hopper kernels use TMA semantics that differ on Blackwell. Uninstalled to prevent auto-detection.

### T1b (Liger fused CE) — in progress at time of writing
liger_kernel 0.8.0 already in venv. ms-swift has `--use_liger_kernel` flag + `_patch_liger_kernel()` hook that intercepts `transformers/loss/loss_utils.py:ForCausalLMLoss` (our known OOM point at vocab=152064 × seqlen × 4 bytes = up to 12 GB). Liger doesn't have a Qwen2.5-Omni-specific patch, but the LM loss path is shared with vanilla Qwen2, so the fused CE should apply regardless.

Expected: step time 1.1–1.25× faster, **peak memory drops by 8–12 GB** (logits never materialize at vocab×seqlen full size).

If the loss line in the bench log shows the same `'loss':` token-acc trajectory as B0, the patch is semantically correct.

---

## Open questions parked for after the sweep

1. **Why does R[1] correlation refuse to break above 0.3 even when train_loss drops to 0.036?** Hypothesis: the LLM's R[1] prediction is conditioned mostly on the *prompt format* (system message + "this ad is N seconds long") rather than on the visual/audio embeddings. A real test would be to ablate: feed the model an unrelated video and see if R[1] changes meaningfully. If it doesn't, the bottleneck is the projection from frozen vision encoder into the LLM, not the LLM itself.

2. **Should we unfreeze the aligner?** Currently `freeze_aligner=true`. Unfreezing adds ~100M params to the gradient buffer + opt state but lets the projection adapt to TTCC's specific feature distribution. Cheap to try once the sweep commits a config.

3. **GRPO from ckpt-50** is the known mode-collapse breaker (per `04_final_report.md`). Need to run it eventually but FIRST we need the config sweep to commit a viable training recipe, AND we should run an audit-ambitious SFT on full 717+101 to give GRPO a stronger init than v38 (which was filtered 116-only).

4. **Test set leakage.** `07_recipe_gaps.md §2'` calls out that prior project's "best ckpt" choices (SFT-Ext-270, GRPO-Ext-150) were test-tuned and invalidated. We have a real val split now and have been using it — but we should re-verify our workflow doesn't accidentally peek at test during ckpt selection.

5. **CoT quality never audited.** `07_recipe_gaps.md §3` flags that the v3 Gemini teacher was never sample-audited for grounding. If teacher hallucinates "Drops at second 17" when nothing happens there, the student learns to associate fabricated events with fabricated drops. This is a possible explanation for why CoT only helps format adherence and not numeric quality.

---

## What this session has and hasn't touched

**Touched today (afternoon):**
- v38 ckpt-50 and ckpt-80 inference + curve-space analysis (H1)
- FA3 install + smoke test (T1a) — negative
- Liger kernel benchmark setup (T1b) — running
- Bench harness `bench_sft.sh` written

**Not touched today:**
- The actual full audit-ambitious 717-row training run (H2 in the task list)
- GRPO from ckpt-50 (H3 in the task list)
- Aligner unfreeze experiment
- CoT teacher audit
- Test-set workflow review
