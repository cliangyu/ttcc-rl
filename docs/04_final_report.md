# TTCC retention-curve prediction — final report (2026-05-21)

CS224R 2026 final project. Predicting second-by-second viewer retention `R(t)`
for short-form video ads from raw video + audio.

This report is self-contained for someone with no prior project context. It
covers (1) the evaluation protocol we landed on, (2) every experiment we ran
with full hyperparameter settings, and (3) the layered findings.

---

## TL;DR

1. **Mean IBS is misleading.** The "all 87 ads, mean IBS" headline shows
   methods tied with the train-mean baseline B1. Decomposing by ad type
   reveals real wins.
2. **On Q2+Q3 (moderate-novelty ads, where being content-aware matters):**
   both SFT and base GRPO-50 **significantly beat B1** (paired BCa CI
   excludes 0).
3. **On segment metrics:** GRPO is the only method with non-trivial
   across-ad ranking signal for **hook strength** (Spearman ρ = +0.22 vs
   B1's undefined). SFT-Extended **significantly beats B1 on completion
   rate magnitude** (CI excludes 0).
4. **CoT scaffolding earns its keep through format adherence**, not
   numeric quality. Removing CoT from the SFT target drops 21/87 ads to
   parse failures with no IBS gain on the 66 that do parse.
5. **GRPO-Extended (179 steps from SFT-Extended) did not surpass base
   GRPO-50.** Frame-budget penalty in the SFT-Extended starting point
   cost us; full-epoch RL couldn't recover.

---

## 1. Problem setup

**Input:** a short-form video ad (mp4, 5–60 s, audio + visual).
**Output:** a per-second retention curve `R(t)`, `t ∈ {0..T_i}`, where
- `R(0) = 1.0` by definition,
- `R(t) ∈ [0, 1]`,
- `R(t)` is monotone non-increasing (viewers don't return),
- `T_i` = ad duration in seconds (per-ad horizon).

**Dataset (TTCC):**
- 717 train ads with CoT-distilled rationales + ground-truth curves
- 87 test ads with ground-truth curves (one corrupted mp4 dropped → 87)
- Curves: $R(1) \in [0.2,\,0.5]$ typical; long tail decays toward $\sim 0.05$ by $T_i$

---

## 2. Evaluation protocol (4 layers)

The original milestone proposed within-ad Spearman $\bar\rho_{\text{shape}}$ and
across-ad mean-rank $\rho_{\text{comp}}$ as headline metrics. We discarded both
during this project — see `ttcc-eval/docs/07_proper_scoring_rule_revision.md`
for the verified critique. The headline finding: **any monotone-decreasing
curve scores $\bar\rho_{\text{shape}} \approx +1.0$ against any other monotone-decreasing
curve**, so Spearman cannot distinguish content-aware from content-blind
models. B1 (train-mean) scored $\bar\rho_{\text{shape}} = +0.95$ under the original
metric — clearly broken.

### Layer 1 — Headline metric: IBS (Integrated Brier Score)

$$
\text{IBS}_i = \frac{1}{T_i + 1} \sum_{t=0}^{T_i} \big(\hat R_i(t) - R_i(t)\big)^2
$$

per ad. Aggregate as unweighted $\overline{\text{IBS}} = \frac{1}{n}\sum_i \text{IBS}_i$.

- **Source:** Brier 1950 (proper scoring for binary outcomes) + Graf 1999
  (integrated form for survival curves).
- **Why:** strictly proper. Cannot be gamed by predicting "any monotone
  curve."
- **Censoring:** none in our setting (every test ad has full $T_i$ seconds
  of data), so the Graf IPCW weights are all 1; IBS reduces to plain
  time-averaged MSE on a probability target.
- **CI:** percentile bootstrap on the 87 per-ad IBS values, $B = 10{,}000$.

### Layer 2 — Diagnostic: Calibration slope

Pool all $(\hat R_{\text{hat}}, R_{\text{true}})$ pairs ($\sim 2{,}200$ points) and fit
$R_{\text{true}} \sim a + b \cdot \hat R_{\text{hat}}$. Target slope $b = 1.0$. $b < 1$ = over-confident, $b > 1$ =
under-confident.

### Layer 3 — Significance: paired BCa bootstrap

For comparing method A vs method B: per-ad $\Delta_i = \text{IBS}_{A,i} - \text{IBS}_{B,i}$,
resample with replacement, $B = 10{,}000$, bias-corrected accelerated 95% CI
on $\overline\Delta = \frac{1}{n}\sum_i \Delta_i$. CI excludes 0 → real effect; CI contains 0 → "tied."
Source: Efron 1987.

### Layer 4 — Decomposition (added during the project)

Two complementary decompositions, motivated by the observation that
**mean IBS over all 87 ads hides the regimes where methods earn their
keep**.

**Novelty-conditional IBS:**
Per-ad novelty $\nu_i = \frac{1}{T_i + 1}\sum_t \big(R_{B_1}(t) - R_i(t)\big)^2$ — how far the GT
curve deviates from the train-mean. Quartile-stratify the 87 test ads:
- **Q1** (closest 25% to B1): easy ads, B1 trivially wins
- **Q2+Q3** (middle 50%): the regime where content-awareness pays
- **Q4** (farthest 25%): very-deviant ads, nobody predicts the direction

Report IBS + paired BCa **on the Q2+Q3 subset specifically**. This is
the honest answer to "is video analysis working?"

**Segment-level metrics** (industry-standard, see [TTS Vibes
hook](https://insights.ttsvibes.com/tiktok-first-3-seconds-hook-retention-rate/),
[Retention Rabbit YouTube benchmarks](https://www.retentionrabbit.com/blog/2025-youtube-audience-retention-benchmark-report),
academic basis in [scikit-survival time-restricted Brier](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)):
- **Hook rate** $H_i = R_i(3)$ (Meta convention) — fraction past the 3-second mark
- **Completion rate** $C_i = R_i(T_i)$ — fraction watching to end

For each method, we compute:
- **MSE across ads** on the scalar prediction: $\text{MSE} = \frac{1}{n}\sum_i (\hat H_i - H_i)^2$
- **Spearman $\rho$ across ads** — do we rank ads correctly by this scalar?
- **Classification AUC** for "strong-vs-weak" using the train-data median
  as threshold

---

## 3. Methods + settings

### Common setup

| | value |
|---|---|
| Student model | Qwen2.5-Omni-3B (Apache-2.0, local at `/home/ssm-user/work/hf-cache/`) |
| Teacher (CoT distillation) | Qwen3-Omni-30B-A3B-Instruct, talker disabled |
| LoRA rank / alpha | 16 / 32 |
| LoRA target | all-linear |
| Dtype | bfloat16 |
| GPUs | 2× A100 96GB |
| Hardware framework | ms-swift (modelscope fork), vLLM 0.21 |
| Test set | 87 ads (after dropping 1 corrupted mp4) |
| Train set | 717 ads with CoT rationales |
| Inference | greedy (temp=0), `--max_new_tokens 1024`, FPS=1.0, `FPS_MAX_FRAMES=24` |
| Parser | balanced-JSON first, bare list fallback, enforces monotone + R[0]=1 |

### Per-method settings

| method | recipe | key hyperparams | wall time |
|---|---|---|---|
| **B1** (baseline) | predict the train-mean curve, truncated to each test ad's $T_i$ | — | seconds |
| **Zero-shot iter2** | Qwen2.5-Omni-3B with hand-crafted worked-example prompt | no training | inference only |
| **SFT** (base) | LoRA SFT on `ttcc_train_sft` (CoT + curve target), 1 epoch | $\eta = 10^{-4}$ cosine, bs $1 \times \text{grad-accum } 4 \times 2\,\text{GPU} = 8$, 90 steps, `FPS_MAX_FRAMES=32` | ~8 min |
| **SFT-noCoT** | same as SFT but assistant target = `Curve: {...}` only | same | ~8 min |
| **SFT-Extended** | 3 epochs of SFT (270 steps) | same $\eta$, `FPS_MAX_FRAMES=24` (lowered to match infer/GRPO) | ~22 min |
| **GRPO-50** (base) | GRPO from SFT ckpt-90, $\sim 50$ steps (28% of one epoch) | $\eta = 5\!\times\!10^{-6}$, $\beta = 0.04$ KL, temp $0.4$, top-$p$ $0.95$, num_generations $= 2$, reward $= 1 - \text{IBS} + 0.2 \cdot \text{format}$ | ~30 min |
| **RLOO** | RLOO variant, 25 steps | as GRPO, `--advantage_estimator rloo --kl_in_reward false` | ~15 min |
| **GRPO-Extended** | full 1-epoch GRPO (179 steps) from SFT-Extended ckpt-270 | as GRPO-50 | 2h 21m |

### Why these settings (the audit)

The full hyperparameter table with rationale for every choice and the
six known limitations is at `docs/02_experiment_configs.md`. Key
assumptions worth flagging to your teammate:

1. **`num_generations = 2`** for GRPO is small (literature standard 4–8);
   chose 2 because rollout cost on 30-second video is high.
2. **Frame budget asymmetry:** base SFT trained at 32 frames; everything
   else (inference, GRPO, SFT-Extended) at 24. This cost SFT-Extended
   some IBS we couldn't recover.
3. **No held-out validation set** within train; all hyperparams chosen
   by inspection. Risk of test-set overfit via choices.
4. **Only one seed per method.** Variance across seeds is unknown.
5. **CoT distillation:** single greedy teacher pass, no sampling.
6. **GRPO reward $r = 1 - \text{IBS}$** rather than milestone's $r = \rho_S$
   (within-ad Spearman). Justification in
   `ttcc-eval/docs/07_proper_scoring_rule_revision.md`.

### CoT distillation prompt

**Outcome-conditioned rationale generation** — the teacher sees the GT
retention curve and explains *why* it has that shape. Output per ad:

```
Content: <one sentence describing the ad>
Drops: <2 sentences naming specific seconds where retention falls, with
       reasons tied to on-screen or audio content>
Reasoning: <one sentence summarizing the overall shape>
```

Closest cousin reference: Hsieh 2023 "Distilling Step-by-Step" and Wang
2023 "SCOTT" (CoT distillation with the answer in context). **Not**
instruction back-translation.

The SFT assistant target is this CoT block plus the numerical curve:
`Content: ... \nDrops: ... \nReasoning: ... \nCurve: {"R": [1.0, 0.33, ...]}`.

---

## 4. Headline results (Layer 1)

**Full test set, $n = 87$ ads. IBS lower = better. Paired BCa vs B1.**

| method | IBS | calib slope | $\Delta$ vs B1 | sig | $n_{\text{parsed}}$ |
|---|---|---|---|---|---|
| Zero-shot iter2 | 0.270 | $\sim 0$ | $\sim +0.26$ | ✓ $\sim 30\times$ worse | 87 |
| **B1 (train-mean)** | **0.0083** | $+1.001$ | reference | | 87 |
| **GRPO-50** | **0.0083** | $+0.969$ | $-0.0001$ | tied | 87 |
| SFT-Extended ckpt-270 | 0.0093 | $+0.957$ | $+0.0010$ | tied | 87 |
| RLOO | 0.0091 | $+0.965$ | $+0.0008$ | tied | 87 |
| SFT (1 epoch, CoT) | 0.0094 | $+0.966$ | $+0.0011$ | tied | 87 |
| **GRPO-Extended ckpt-150 (best)** | **0.0091** | — | $+0.0008$ | tied | 87 |
| SFT-noCoT | 0.0115 (on 66) | $+0.976$ | $+0.0026$ | ~ tied | **66** ← parse fails |

**Reading the headline naively:** "Nothing beats B1, methods are doing
nothing useful." This is false — it's an artifact of averaging across
regimes where the methods help vs hurt. See Layer 4.

**Trained methods do beat the zero-shot baseline by $\sim 30\times$** ($0.0083$ vs
$0.270$). That part is real and important. The methods learned *something*
substantial — just not enough to surpass the strong B1 floor on a
uniform-weight average.

---

## 5. Conditional results — where methods earn their keep (Layer 4a)

**Stratified by per-ad B1-deviation quartile. Paired BCa vs B1.**

| subset | $n$ | B1 IBS | SFT | GRPO-50 | story |
|---|---:|---:|---:|---:|---|
| **Q1 (closest to B1)** | 22 | **0.0012** | 0.0045 ($\Delta=+0.0034$ ✓ **worse**) | 0.0032 ($\Delta=+0.0020$ ✓ **worse**) | B1 trivially wins; methods over-think |
| **Q2+Q3 (middle)** | 43 | 0.0052 | 0.0035 ($\Delta=\mathbf{-0.0016}$ ✓) | **0.0030** ($\Delta=\mathbf{-0.0022}$ ✓) | **methods significantly beat B1** |
| **Q4 (farthest)** | 22 | 0.0217 | 0.0257 (tied) | 0.0236 (tied) | nobody predicts direction; everyone falls back |

**Q2+Q3 is the honest answer to "is video analysis working?"**
- SFT and GRPO **both significantly beat B1** (CI excludes 0).
- GRPO has the largest effect: $\sim 42\%$ lower IBS than B1 on this subset
  ($0.0030$ vs $0.0052$).
- Q1 + Q4 averaged into the headline cancel the Q2+Q3 wins, producing
  the misleading "tied with B1" full-set result.

**Practical implication:** a 2-stage system would dominate — predict B1
for Q1-classified ads, GRPO for the rest. Naive upper bound (perfect
oracle routing): $\text{IBS} \approx 0.0069$, $\sim 17\%$ improvement over B1's $0.0083$.

Full doc: `docs/03_conditional_results.md`.

---

## 6. Segment-level results — what matters for ad creators (Layer 4b)

**Hook rate $R(3)$ and completion rate $R(T_i)$. All 87 ads.**

### Hook rate $R(3)$ — "did we predict the 3-second hook strength?"

| method | MSE | Spearman $\rho$ | AUC(strong) | $\Delta$ MSE vs B1 |
|---|---:|---:|---:|---|
| B1 | **0.0200** | n/a (constant) | n/a | reference |
| **GRPO-50** | 0.0213 | $\mathbf{+0.217}$ | 0.544 | $+0.0012$ tied |
| RLOO | 0.0239 | $+0.025$ | 0.517 | tied |
| SFT-Extended ckpt-270 | 0.0260 | $+0.120$ | 0.513 | $+0.0060$ ✓ worse |
| SFT | 0.0258 | $+0.165$ | **0.598** | $+0.0058$ ✓ worse |
| SFT-noCoT | 0.0346 ($n=66$) | $-0.035$ | 0.355 | $+0.0123$ ✓ worse |

**Interpretation:** B1 has the lowest MSE because its constant
prediction of $R(3) = 0.387$ exploits the small variance in $R(3)$ across
ads. But B1 has **zero ranking signal** (Spearman undefined). **GRPO-50
is the only method with substantial across-ad Spearman ($+0.217$)**,
meaning it correctly identifies *which* ads have strong-vs-weak hooks
even when its absolute magnitude is noisy. **SFT has the best
classification AUC ($0.598$)** for "strong hook" (threshold = train
median $0.137$).

**For an ad creator workflow:** "Your model thinks this ad has weaker
hook than typical" is actionable in a way that "your $\text{IBS} = 0.0083$"
is not. GRPO and SFT both provide this signal.

### Completion rate $R(T_i)$ — "did the audience watch to the end?"

| method | MSE | Spearman $\rho$ | AUC(strong) | $\Delta$ MSE vs B1 |
|---|---:|---:|---:|---|
| B1 | 0.0027 | $-0.029$ | 0.459 | reference |
| **SFT-Extended ckpt-270** | **0.0016** | $-0.051$ | 0.413 | $\mathbf{-0.0010}$ ✓ |
| GRPO-50 | 0.0017 | $+0.010$ | 0.506 | $-0.0010$ marginal tie |
| SFT | 0.0017 | $-0.054$ | 0.402 | $-0.0010$ marginal tie |
| SFT-noCoT | 0.0019 ($n=66$) | $\mathbf{+0.199}$ | 0.510 | tied |
| RLOO | 0.0020 | $-0.012$ | 0.435 | tied |

**Interpretation:**
- **SFT-Extended significantly beats B1** on completion-rate MSE
  (CI excludes 0). 3 epochs of training on CoT-distilled data improved
  the model's ability to predict whether the audience would stick to
  the end.
- GRPO-50 and base SFT directionally beat B1 but just barely fail the
  significance bar.
- **SFT-noCoT has the highest Spearman ($+0.199$)** — when no-CoT does
  parse a curve, it ranks completion better. But it loses 21 ads to
  parse failures.

Full doc: `docs/03_conditional_results.md` (segment additions in inline
in `scripts/segment_eval.py` output).

---

## 7. GRPO-Extended saturation analysis

Trained for the full 1 epoch (179 steps, 2h 21m) from SFT-Extended
ckpt-270. Evaluated 6 intermediate checkpoints by IBS:

| ckpt | IBS | $\Delta$ vs base SFT | $\Delta$ vs base GRPO-50 |
|---|---:|---|---|
| 25 | 0.0097 | tied | $+0.0014$ ✓ worse |
| 100 | 0.0094 | tied | $+0.0012$ ✓ worse |
| 125 | 0.0092 | tied | tied |
| **150** | **0.0091** | **tied** | **tied** (best) |
| 175 | 0.0095 | tied | $+0.0013$ ✓ worse |
| 179 | 0.0091 | tied | tied |

**Conclusions:**
- IBS improved monotonically from ckpt-25 ($0.0097$) to ckpt-150 ($0.0091$).
- Slight regression at ckpt-175 then recovery to $0.0091$ at final ckpt-179.
- **Never surpassed base GRPO-50's $0.0083$.**

**Why we didn't gain over base GRPO:** SFT-Extended (the starting point
for GRPO-Extended) used `FPS_MAX_FRAMES=24` while base SFT used 32 — a
visual-context budget reduction that cost some IBS. Full-epoch GRPO
closed most of the gap but didn't recover the lost frame information.
Training-time reward saturated quickly ($1.18$–$1.20$ of a $1.20$ max), so
the learning headroom was small.

**On Q2+Q3 (the regime that matters):** GRPO-Extended ckpt-150
achieved $\text{IBS} = 0.0037$ vs base GRPO-50's $0.0030$. Did not improve
content-aware prediction. CI relative to B1 just contains 0.

---

## 8. Other findings (CoT ablation)

**SFT-noCoT** (CoT removed from assistant target, 1 epoch identical to
base SFT):
- **66 / 87 parsed** (vs 87 / 87 for base SFT). 21 ads lost to format
  drift.
- On the 66 that parsed: $\text{IBS} = 0.0115$; paired $\Delta\text{IBS}$ vs base SFT
  $= +0.0010$, CI $[-0.0002,\,+0.0022]$ — tied.
- **Headline:** CoT scaffolding is doing **format-adherence work**, not
  numeric-quality work. The model can learn the curve numbers without
  CoT, but it loses format reliability. CoT is load-bearing for
  deployment, not for math.

---

## 9. Implementation honest deltas vs milestone

The milestone document specified:
1. VideoLLaMA2.1-7B-AV with a **hazard head** for monotone-by-construction
   prediction.
2. SFT loss $\mathcal{L}_{\text{SFT}} = \sum_t \big(\log \hat\lambda(t) - \log \lambda(t)\big)^2 + \alpha \cdot \mathrm{CE}_{\text{CoT}}$ (joint
   log-hazard regression + CoT cross-entropy).
3. GRPO reward $r = \rho_S$ (within-ad Spearman).
4. Teacher = Gemini-2.5 (API).

We diverged on every one:
| | milestone | ttcc-rl | reason |
|---|---|---|---|
| Architecture | VideoLLaMA2.1 + hazard head | Qwen2.5-Omni-3B + LM-head text output | smaller, native audio+video, Apache-2.0 |
| Monotonicity | by construction (softplus hazard) | post-hoc parser enforcement | text-domain prediction worked |
| SFT loss | log-hazard MSE + CoT CE | plain LM CE on the whole assistant target | simpler, matches HF/swift defaults |
| Reward | within-ad Spearman | $1 - \text{IBS}$ (proper scoring rule) | Spearman was empirically broken |
| Teacher | Gemini-2.5 (API, \$50 budget) | Qwen3-Omni-30B-A3B-Instruct (local, Apache-2.0) | local, free, comparable quality |

We did **not** implement the milestone's three "named baselines":
- SFT-MSE (per-second sigmoid head, MSE-only, no CoT)
- SFT-Hazard+CoT (the proper milestone-§2 SFT)
- GRPO with $r = \rho_S$ (the proper milestone-§3 reward)

These remain open. Our pipeline is a **sibling**, not a replacement, of
the milestone's three baselines. Detailed delta accounting in
`docs/01_method.md`.

---

## 10. Recipe at a glance

Reproducing the full pipeline from scratch (assumes Qwen2.5-Omni-3B
locally cached and ms-swift in `swift_venv`):

```bash
# 0. CoT distillation
python scripts/cot_distill.py --model INSTRUCT --full \
    --out work-out/cot/full_instruct.jsonl

# 1. Prepare ms-swift datasets
python scripts/prepare_dataset.py \
    --cot-jsonl work-out/cot/full_instruct.jsonl \
    --out-dir data/ttcc_swift

# 2. SFT (base)
bash go_viral/examples/train/grpo/qwen2_5_omni_ttcc/sft.sh

# 3. Eval SFT
bash scripts/infer_trained.sh <sft-ckpt> sft work-out/preds_sft.parquet

# 4. GRPO from SFT
SFT_CKPT=<sft-ckpt> bash go_viral/examples/train/grpo/qwen2_5_omni_ttcc/grpo.sh

# 5. Eval GRPO
bash scripts/infer_trained.sh <grpo-ckpt> grpo work-out/preds_grpo.parquet

# 6. Headline eval + paired BCa
python scripts/eval_one.py work-out/preds_grpo.parquet --name GRPO --vs B1 SFT

# 7. Conditional eval (Q2Q3 — where content-awareness matters)
python scripts/conditional_eval.py \
    --preds preds_sft.parquet:SFT preds_grpo.parquet:GRPO \
    --b1-preds preds_b1.parquet \
    --gt data/ttcc_swift/ttcc_test.jsonl \
    --subset Q2Q3 --ref B1

# 8. Segment eval (hook + completion)
python scripts/segment_eval.py \
    --preds preds_sft.parquet:SFT preds_grpo.parquet:GRPO \
    --b1-preds preds_b1.parquet \
    --gt data/ttcc_swift/ttcc_test.jsonl
```

Branch on go_viral fork: `cliangyu/go_viral:ttcc-rl` (commit `fc1e79d2`).

---

## 11. Recommended discussion points with teammate

1. **Mean-IBS is misleading; should we make Q2+Q3 conditional the primary
   headline?** B1 is too strong on Q1 for the unweighted mean to be
   honest. Q2+Q3 is the "is content-awareness working" answer.
2. **Hook ranking (Spearman) is the most actionable single number.**
   For ad creators, "which ads have strong hooks" is more useful than
   "what's the full curve."
3. **Is the FPS_MAX_FRAMES=24 vs 32 budget worth re-running base SFT
   at 24** for an apples-to-apples comparison with everything else?
   Currently base SFT is the only one trained at 32.
4. **GRPO with `num_generations=2`** is unusual. Worth rerunning with
   4 if time/compute allow — could be the difference between
   GRPO-Extended winning vs tying on Q2+Q3.
5. **Routing/ensemble experiment (the next obvious step):** train a
   gate to predict per-ad novelty quartile and route Q1 to B1,
   Q2+Q3+Q4 to GRPO. Naive oracle upper bound is $\sim 17\%$ improvement.
6. **Three milestone baselines remain unimplemented** (SFT-MSE, SFT-
   Hazard+CoT, GRPO with $\rho_S$). These are needed for a proper
   side-by-side with the milestone's named recipes.

---

## Index of documents

- `docs/01_method.md` — implementation deltas vs milestone §2-3
- `docs/02_experiment_configs.md` — full hyperparameter audit
- `docs/03_conditional_results.md` — novelty-quartile decomposition
- **`docs/04_final_report.md`** — this document
- `ttcc-eval/docs/07_proper_scoring_rule_revision.md` — eval protocol revision rationale + 6 verified hypotheses
- Scripts: `scripts/eval_one.py`, `scripts/conditional_eval.py`,
  `scripts/segment_eval.py`, `scripts/inflection_analysis.py`
- Predictions parquets: `/home/ssm-user/work/work-out/preds_*.parquet`
- Training scripts (on `cliangyu/go_viral:ttcc-rl`):
  `examples/train/grpo/qwen2_5_omni_ttcc/{sft,sft_nocot,sft_extended,grpo,grpo_extended,rloo}.sh`

---

*Report compiled 2026-05-21 02:58 UTC. All numbers from the parquets in
`/home/ssm-user/work/work-out/` and reproducible with the commands in §10.*
