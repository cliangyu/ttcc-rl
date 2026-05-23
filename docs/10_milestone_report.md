# CS224R Milestone Report
### Per-Second Retention Curve Prediction with SFT + GRPO

**Author:** Liangyu Chen (liangyuch@stanford.edu) · **Course:** CS224R Spring 2026

---

## Setup (delta from proposal)

We predict a short-form video ad's per-second retention curve $R(t) \in [0,1]$, $t \in \{0,\dots,T_i\}$, monotone non-increasing, from raw audio+video. Two methodological deviations from the proposal, both justified in `docs/01_method.md`: (1) **architecture** — Qwen2.5-Omni-3B replaces VideoLLaMA2.1-7B-AV (smaller, native audio+video, Apache-2.0; LM emits the curve as JSON text and we parse + monotonize post-hoc, instead of a hazard head); (2) **GRPO reward** — $r_i = 1 - \text{IBS}_i$ (Brier-1950 / Graf-1999 proper scoring rule) replaces within-ad Spearman $\rho_S$, because any monotone-decreasing prediction scores $\bar\rho_S \approx +1.0$ and so $\rho_S$ cannot distinguish a content-aware model from a content-blind one. Train: 717 ads with CoT-distilled rationales; test: 87 ads held out. Baseline B1 = train-mean curve truncated to each ad's $T_i$; $\text{IBS}_{B_1} = 0.0083$.

## Experiment since proposal: full-FT capacity test + curve-space diagnosis (v38)

We trained Qwen2.5-Omni-3B **full-fine-tuned** (vs prior LoRA experiments) for 10 epochs on a 116-ad subset, with the audit-fixed recipe (FPS=1, FPS_MAX_FRAMES=60, max_pixels=200704, max_length=24576, Talker disabled). Then we **parsed the predicted JSON curves and computed metrics in retention-curve space**, not just token-CE space.

**Token-CE training looked like a clean overfit:** train_loss 0.99 → 0.036 in 80 update steps, eval_loss 1.21 → 1.82 (textbook overfit hinge), train-token-accuracy 99%. **But in curve space, the picture inverts:**

| metric | ckpt-50 (best val CE) | ckpt-80 (final) | B1 baseline |
|---|---:|---:|---:|
| val per-ad MSE (mean) | **0.0075** | 0.0082 | 0.0083 |
| val per-ad MSE (median) | 0.0032 | 0.0043 | — |
| val R[1] correlation with GT | −0.24 | −0.20 | 0 (constant) |
| std of predictions across ads | 0.040 | 0.045 | 0 |
| length matches $T_i + 1$ | 100% | 100% | 100% |

Figure 1 (`runs/v38_inference/figures/ckpt80_compare.png`) plots GT (blue) vs predicted (red) curves for 6 train + 6 val ads. On train, predictions track GT for some ads; **on val, they collapse to a near-identical exponential decay regardless of content** — average predicted curve at relative times {0, 0.25, 0.5, 0.75, 1.0} is essentially identical across train and val ($\approx \{1.00, 0.13, 0.07, 0.04, 0.03\}$). Per-ad R[1] correlation is ≈ 0 on both train and val; ckpt-50 ties B1 in per-ad MSE on val (0.0075 vs 0.0083). **Token-CE-trained SFT, even run to overfit, converges to a mode-collapsed prediction that is statistically indistinguishable from predicting the dataset average.** This corroborates two pre-existing observations from proposal-era LoRA experiments: (a) LoRA-SFT IBS = 0.0094 tied with B1 on the full 87-ad test set ($n=87$ paired BCa); (b) LoRA-GRPO-50, with the same SFT init, was the **only** method to produce a non-trivial across-ad ranking signal (Spearman $\rho = +0.22$ on hook-strength prediction). The signal that distinguishes ads came from RL, not from SFT.

## Hypothesis update

**Original (proposal):** SFT trains the curve policy; GRPO is a refinement step. **Revised:** SFT's token-CE objective is **fundamentally mode-collapsed** for this task — the loss minimum "always predict the dataset-average curve" is reachable without per-ad content understanding, so SFT does not learn ad-conditional structure no matter how long it trains. GRPO with $r = 1 - \text{IBS}$ is **load-bearing**, not optional polish: it is the only training signal that directly rewards curve fidelity. SFT supplies format adherence (100% parse rate, length matching) and a stable RL init. This is consistent with the audit's CoT ablation: removing CoT from SFT loses 21/87 ads to parse failures with no IBS gain on the 66 that parse — CoT earns its keep through format, not numeric quality.

## Concrete steps to completion

1. **Full-FT SFT on the 717-ad training set (H2, in progress as of this writing)** — audit-fixed recipe with the system configuration committed at `go_viral@dc6201f5` (T1–T6 sweep, see `docs/09_session_journal_*`). The sweep found that boring infrastructure knobs (dataloader_num_workers=4, persistent_workers, prefetch_factor=4) gave a 3.3× throughput win that no kernel-level change (FA3, Liger, ZeRO-2) could match — FA3/4 are also not supported on our sm_120 silicon. Expected wall: ~8 h. We will offline-evaluate periodic checkpoints in curve space (since H1 showed eval-loss is a weak proxy).
2. **GRPO from H2's best curve-quality checkpoint (H3)** with audit-fixed RL knobs: $\beta=0.001$, num_generations=4, max_completion_length=1024, temperature=0.5, reward $= 1 - \text{IBS} + 0.2 \cdot \text{format}$. Expected wall: ~4–6 h.
3. **Paired BCa-bootstrap evaluation** on the held-out 87-ad test set comparing (a) H2 full-FT SFT vs B1, (b) H3 full-FT GRPO vs B1, (c) H3 vs prior LoRA-GRPO-50. Headline IBS, conditional Q1/Q2+Q3/Q4 decomposition (the regime where content-awareness pays — prior LoRA-GRPO significantly beat B1 only on Q2+Q3), and Spearman ranking on hook-strength prediction.
4. **Fallback if GRPO does not break mode collapse:** aligner-unfreeze ablation (currently `freeze_aligner=true`), testing whether the visual/audio projection into the LLM is the bottleneck rather than the LLM itself.

## AI Tools Disclosure

Claude (Anthropic, claude-opus-4-7) was used for configuring training infrastructure (DataLoader knobs, DeepSpeed settings, FlashAttention build for sm_120 Blackwell), generating analysis scripts (JSON-curve parsers, per-ad MSE / R[1]-correlation computations, matplotlib panels), and drafting documentation including this report. The research design — mode-collapse diagnostic protocol, the $r = 1 - \text{IBS}$ reward and its critique of $\rho_S$, the conditional Q1/Q2+Q3/Q4 decomposition, and the hypothesis revision above — was developed by the author. All experiments were planned and interpreted by the author; AI assisted with code synthesis and writing clarity.
