# Implementation deltas from milestone §2–3

This document is the honest accounting of where `ttcc-rl` diverges from the
spec in the CS224R 2026 project milestone. The eval-protocol revision is
documented separately in `ttcc-eval/docs/07_proper_scoring_rule_revision.md`.

## Architecture

| milestone §2 | ttcc-rl |
|---|---|
| VideoLLaMA2.1-7B-AV (forked from LMM-EVQA) | **Qwen2.5-Omni-3B** (different family, smaller; native audio+video; Apache-2.0) |
| Hazard head $\hat\lambda(t) = \mathrm{softplus}(W h_{\langle/\text{cot}\rangle})$, $\hat R(t) = \exp(-\sum_{s \le t} \hat\lambda(s))$; monotone by construction; single forward | **No hazard head**. The LM head emits the curve as text tokens (e.g. `{"R": [1.0, 0.85, ...]}`). Parser enforces monotone post-hoc. |
| CoT generated between `<cot>...</cot>` special tokens, separate from $R$ | CoT and $R$ are part of the **same assistant text** stream (`Content: ... \nDrops: ... \nReasoning: ... \nCurve: {"R": [...]}`). |

The hazard-head replacement is the largest structural deviation. We did not
demonstrate hazard-head SFT or compare to text-domain SFT; both are open
questions. Given the eval-protocol shift to IBS (which directly measures
prediction quality regardless of parameterization), text-domain SFT did
empirically close the IBS gap to B1, suggesting the hazard head is not a
hard prerequisite for the IBS-headline result — but milestone §4(1)
SFT-MSE and §4(2) SFT-Hazard+CoT remain unimplemented for direct comparison.

## SFT loss

| milestone §2 | ttcc-rl |
|---|---|
| $\mathcal{L}_{\text{SFT}} = \sum_t \big(\log \hat\lambda(t) - \log \lambda(t)\big)^2 + \alpha \sum_k -\log p_\theta(c_k \mid \cdot)$ (joint hazard MSE + CoT CE) | **$\mathcal{L}_{\text{SFT}} = \sum_\tau -\log p_\theta(\tau_{\text{token}} \mid \tau_{<})$** over the entire assistant span (Content/Drops/Reasoning/Curve), i.e. plain LM cross-entropy. No log-hazard regression term. |

This means the SFT signal is on token sequences (including numeric digit
tokens). There is no inductive bias for $0.86 \approx 0.85$. Empirically this
still produces calibration slope $\approx 1.0$ and $\text{IBS} \approx \text{IBS}_{B_1}$ — the model learns the
curve well enough through token CE alone. Whether a hazard regression loss
would do better is untested.

## GRPO reward

| milestone §3 | ttcc-rl |
|---|---|
| $r_i = \rho_S\!\big(\hat R_i(\cdot), R_i(\cdot)\big)$ — per-rollout within-ad Spearman | **$r_i = 1 - \text{IBS}_i$** where $\text{IBS}_i = \dfrac{1}{T_i + 1} \sum_{t=0}^{T_i} \big(\hat R_i(t) - R_i(t)\big)^2$ — proper scoring rule (Brier 1950; Graf 1999) |

Rationale and supporting literature in
`ttcc-eval/docs/07_proper_scoring_rule_revision.md`. Empirical justification:
H1 in that document — any monotone-decreasing prediction whatsoever gets
$\bar\rho_{\text{shape}} = +1.0$ under the existing pipeline, so within-ad Spearman
**cannot** distinguish a content-aware model from a content-blind one.

The companion format reward (`ttcc_format`, weight 0.2) gives a small
bonus when the completion contains a parseable R-list; this is from the
Qwen2.5-Omni grpo.sh example and is unchanged from milestone implicit
behavior.

## Training schedule

| milestone | ttcc-rl |
|---|---|
| Full epochs implied; 3 seeds per method | Partial epochs (SFT 1, GRPO 28%, RLOO 14%); 1 seed |

Saturation analysis (`scripts/viz/saturation.py`) shows SFT loss was still
descending at the cosine-LR-zero cutoff, GRPO reward was oscillating
high-90% with growing rollout `std` (healthy exploration, no $\sigma_g$ collapse).
Extending training to convergence is one of the open experiments
(`scripts/sft_extended.sh`, `grpo_extended.sh`).

## CoT distillation teacher

| milestone | ttcc-rl |
|---|---|
| Gemini-2.5 (API, ~$50 budget) | **Qwen3-Omni-30B-A3B-Instruct** (local, Apache-2.0) |

Teacher prompt is **outcome-conditioned rationale generation**: the teacher
is given the GT retention curve and asked to explain *why* the curve has
that shape. This grounds the reasoning in real per-second drop events
rather than hallucinated ones. Closer-cousin reference: Hsieh et al. 2023
"Distilling Step-by-Step" and Wang et al. 2023 SCOTT (CoT distillation
with answer in context). It is **not** Li et al. 2024 instruction
back-translation, which generates *instructions* not rationales.

## Missing baselines (milestone §4)

- **SFT-MSE** (per-second sigmoid head, MSE-only, no CoT): not implemented.
  This is the literature-style baseline (SnapUGC / VQualA winner recipe).
- **SFT-Hazard+CoT** (the proper milestone-§2 SFT): not implemented.
- **GRPO with $r = \rho_S$** (the proper milestone-§3 reward): not implemented.

These three remain as honest gaps. The IBS-based pipeline we built is a
*sibling*, not a replacement, of the milestone's three baselines.
