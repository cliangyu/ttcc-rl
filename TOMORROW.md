# Open decisions — surface for next session

These were raised in today's work but explicitly **deferred** to tomorrow per the user's instruction. Each entry: the decision, the evidence, the recommendation, the cost.

---

## 1. ttcc-rl repo fate

**Decision:** local-only directory `/home/ubuntu/ttcc-rl/` — push as a 4th remote, fold into one of the others, or leave local?

**Evidence:** the user already has 3 remotes (`ttcc-inference`, `ttcc-eval`, `CS224R`). Today added training scaffolding here as a 4th codebase. Currently no `.git/` directory, no remote.

**Recommendation:** push as 4th remote `cliangyu/ttcc-rl`. The training pipeline is a separate concern from the eval protocol (which now correctly lives in `ttcc-eval`), and a separate concern from zero-shot inference (`ttcc-inference`). Keeping it as a self-contained repo prevents either of the existing repos from bloating.

**Cost:** 2 minutes (`git init` + create remote + push).

---

## ~~2. Reward redesign — bounded saturation problem~~ (still open)

Deferred — overshadowed by the lower-$\beta$ fix below, which is the same
root cause from a different angle.

---

## ~~3. Lower the KL coefficient $\beta$~~ ✓ APPLIED 2026-05-22

Done in `go_viral` commit `8136afe9` — $\beta$ dropped from 0.04 to 0.001
(DeepSeek-R1 value) across `grpo.sh`, `grpo_extended.sh`, `rloo.sh`,
`grpo_v2cot_full.sh`, `rloo_v2cot_full.sh`. See `docs/06_config_audit.md`
for the math (KL gradient was ~16× the reward gradient at $\beta = 0.04$).

---

## ~~4. Larger `num_generations`~~ ✓ APPLIED 2026-05-22

Done in `go_viral` commit `8136afe9` — `num_generations: 2 → 4` across
all GRPO/RLOO scripts. Cost: ~2× rollout time.

---

## NEW: 2'. Config audit — additional issues found ✓ ALL APPLIED 2026-05-22

`docs/06_config_audit.md` surfaced 5 critical config issues. All applied
in `go_viral` commit `8136afe9`:

- `FPS_MAX_FRAMES: 24-32 → 60` — was blinding the model for 38% of test
  ads on the tail
- `max_completion_length: 384 → 1024` — was 87% clipped during GRPO
- `--freeze_vit true --freeze_aligner true` — was implicitly attaching
  LoRA to audio_tower + visual encoder linears (ms-swift default)
- (β and num_generations: covered above)

The audit doc also flagged 3 diagnostics that remain unverified — see §11.

---

## ORIGINAL 2 (kept for reference): Reward redesign — bounded saturation

**Decision:** swap $r_i = 1 - \text{IBS}_i$ for an unbounded-above variant.

**Evidence:** during GRPO-Extended training, mean reward went from 1.188 (step 1) to 1.199 (step 179) — i.e. moved only 0.011 of the 0.012-wide headroom above 1.188. The reward signal saturates near the upper bound, producing weak policy gradient. Discussion in [`docs/04_final_report.md`](docs/04_final_report.md) §7 and in today's transcript section "Numerical optimization problems."

**Options:**
- (a) $r_i = -\text{IBS}_i$ — same gradient direction, no upper bound
- (b) $r_i = -\log(\text{IBS}_i + \epsilon)$ — stretches small differences exponentially
- (c) $r_i = \text{BSS}_i = 1 - \text{IBS}_i / \overline{\text{IBS}}_{B_1}$ — centered around 0, unit-free

**Recommendation:** (a) is the cheapest. Same gradient direction (constant additive shift), so the policy moves the same way; we just remove the wall. Costs nothing.

---

## 3. Lower the KL coefficient $\beta$

**Decision:** drop $\beta$ from 0.04 to 0.01 or 0.005.

**Evidence:** GRPO-Extended barely moved the policy past base GRPO-50 (IBS 0.0093 → 0.0091 over 179 steps vs base GRPO-50's 0.0083 over 50 steps). Consistent with KL anchor dominating when the reward signal is small. $\beta$ ratio to within-group reward std: $0.04 / 0.005 \approx 8$, meaning KL pull is ~8× the typical advantage magnitude. Strong regularization.

**Cost:** one hyperparameter line in `grpo.sh`. Re-run base GRPO with new $\beta$ for direct comparison: ~30 min.

---

## 4. Larger `num_generations`

**Decision:** try `num_generations=4` or `8` instead of 2.

**Evidence:** literature standard for GRPO is 4-16. We chose 2 due to rollout cost on 30-second video. With 2, group advantage estimates have very high variance: $\operatorname{std}(r)$ across 2 samples is unstable. Larger groups → more stable advantage → better gradient quality.

**Cost:** 2-4× rollout time. At 2h21m for 179 steps with num_gens=2, expect 4-9 hours with num_gens=4-8.

---

## 5. Post-hoc isotonic recalibration

**Decision:** fit isotonic regression on a held-out val slice; apply to GRPO outputs at inference time.

**Evidence:** Murphy decomposition (see [`docs/05_evaluation_protocol_v3.md`](docs/05_evaluation_protocol_v3.md) §4) shows trained methods have $\text{REL} \approx 0.002$ vs B1's $\text{REL} = 0.0002$ — a **10× calibration penalty** on the full set. RES (resolution / discrimination) is unchanged across methods. So the full-set tie with B1 is *entirely* a calibration problem. Isotonic regression should close most of the gap **without any retraining**.

**Cost:** ~1 hour. Carve out 10% of train as val, fit `sklearn.isotonic.IsotonicRegression()` on (R̂, R_true) pairs, apply at inference. Single high-leverage experiment.

**Why this is the most exciting open question:** if it works (which the Murphy decomposition predicts it should), we get a free $\text{BSS}$ boost on the full set, and the headline becomes "methods significantly beat B1 with paired BCa CI excluding 0" instead of "tied."

---

## 6. Routing / ensemble experiment

**Decision:** train a gate to classify each test ad as Q1 (predict B1) vs Q2+Q3+Q4 (predict GRPO).

**Evidence:** B1 dominates Q1 ($\text{IBS} = 0.0012$); GRPO dominates Q2+Q3 (BSS = +0.42 vs B1, CI excludes 0). A perfect oracle router would yield $\overline{\text{IBS}} \approx 0.0069$, ~17% improvement over B1's $0.0083$. See [`docs/03_conditional_results.md`](docs/03_conditional_results.md).

**Cost:** ~1-2 hours. Train logistic regression on simple features (SFT confidence + duration + audio loudness + scene-change count). Or: skip the gate, use a "novelty score" computed from SFT prediction confidence as a soft mixing weight.

---

## 7. Implement the milestone-spec baselines

**Decision:** implement the three milestone baselines we skipped.

**Evidence:** the milestone document called for:
1. **SFT-MSE** — per-second sigmoid head, MSE-only loss, no CoT
2. **SFT-Hazard+CoT** — softplus hazard head + CoT cross-entropy
3. **GRPO-$\rho_S$** — within-ad Spearman reward (the "broken" reward we critiqued in `ttcc-eval/docs/07`)

We replaced all three with text-domain SFT + LM-CE + $r = 1 - \text{IBS}$. These are honest "siblings, not replacements" — see [`docs/01_method.md`](docs/01_method.md).

**Recommendation:** if writing this up for the milestone report, implement at least **SFT-MSE** for an apples-to-apples comparison with the literature recipe (SnapUGC / VQualA winner). The other two require architecture work (hazard head) and a known-broken reward (GRPO-$\rho_S$); de-prioritize.

**Cost:** SFT-MSE — ~2 hours (sigmoid head + MSE loss + 1-epoch run + eval). Others — 1-2 days each.

---

## 8. Partial C-index controlling for $T_i$

**Decision:** deconfound the RMST C-index by stratifying by ad duration.

**Evidence:** [`docs/06_metrics_feynman.md`](https://github.com/cliangyu/ttcc-eval/blob/main/docs/08_metrics_feynman.md) §7a: B1 has C-index 0.74 on RMST mostly because $\widehat{\text{RMST}}_{B_1, i}$ is monotone in $T_i$, and true RMST also correlates with $T_i$ (longer ads → more total watch time mechanically). Trained methods have lower C-index (0.60-0.68) because they add content-specific signal that's also content-specific *noise*. A partial C-index — bucketed by $T_i$ quartile, then averaged — would tell us whether the trained methods discriminate *within* a duration bucket better than B1.

**Cost:** ~30 minutes. Pure analysis, no GPU.

---

## 11. Diagnostics still unverified from the config audit

From `docs/06_config_audit.md` §4:

- **4.1** Verify audio actually contributes gradients in v2 full-FT.
  `--freeze_aligner true` freezes the audio aligner; need to check
  whether the LLM still attends to audio embeddings or treats them as
  fixed features. Check `param.requires_grad` per layer.
- **4.2** Count ads silently dropped by `--truncation_strategy delete`
  + `max_length=8192` in v2 full-FT (video tokens at 200704 px × 60
  frames could push some ads over the limit).
- **4.3** Confirm vLLM correctly merges LoRA adapters that include
  audio_tower / visual encoder layers (relevant for the *old* LoRA
  checkpoints trained before the freeze_vit fix).

---

## 12. Engineering hygiene: deduplicate the eval scripts

**Decision:** decide whether `ttcc-eval/scripts/*.py` should call into `ttcc-eval/src/ttcc_eval/{metrics,bootstrap}.py` instead of reimplementing IBS, BCa, etc. inline.

**Evidence:** the scripts we shipped today (`minimal_eval.py`, `full_eval.py`, etc.) reimplement BCa, IBS, calibration_slope inline rather than importing from the `ttcc_eval` package. The package has cleaner versions but a different API (built around the `GroundTruth` dataclass). Two implementations of the same metric = drift risk.

**Recommendation:** consolidate next time we touch `ttcc-eval`. Lower priority than the experiments above.

**Cost:** ~2 hours.

---

## 10. Things I should not silently forget

- **`go_viral` branch `ttcc-rl` commit `122ab8b8`** — refactored launchers, cleaned plugins, deleted 4 dead files. The refactor changed how `sft_extended.sh` is invoked (now an `exec` of `sft.sh` with env overrides). If anyone resumes a hung training run from before this commit, the env-var contract is slightly different.
- **`ttcc-eval` commit `6899fd0`** — added 8 scripts to `scripts/` that reimplement metrics. See decision #9.
- **`scratch/` in ttcc-rl** — deleted in today's cleanup. ~30 one-off analysis scripts. If we need any of them, they're in the git history of `ttcc-eval` and prior conversations (`/home/ubuntu/.claude/projects/-home-ubuntu/*.jsonl`).
- **GRPO-Extended checkpoints** at `/home/ssm-user/work/work-out/ttcc_grpo_extended/v0-20260520-223244/{checkpoint-100,125,150,175,179}/` are the trained adapters. ckpt-150 is the best per IBS on the full test set.

---

*Compiled 2026-05-21. Read first thing tomorrow before resuming.*
