# Milestone strategy + progress evaluation
### Companion doc to `10_milestone_report.md`

Written 2026-05-23 while H2 (full-FT SFT on 717 ads) trains in background. Purpose: explain (a) why the milestone is structured the way it is, (b) where we are vs the project goal, (c) what experiments we have to report.

---

## Part 1 — Why the milestone reads the way it does

### The rubric, decoded

The CS224R milestone is a **1-page, lightly-graded progress check**. The rubric (§5 of the guidelines) lists three required answers:

1. What experiments have you conducted so far?
2. Are there hypothesis changes based on findings?
3. What concrete steps remain to completion?

Plus a "required experiment" — *at least one* attempt since proposal, can be a failure, figures encouraged.

The course-level rubric values "**new insights** that shed light on success or failure modes of an idea" and explicitly says novelty does **not** require SOTA performance. Negative results with analysis are valid contributions.

### What we lead with — and why

I chose to **lead with the H1 mode-collapse finding**, not with the system sweep, even though the sweep was the more time-intensive work. The reasoning:

| Candidate headline | Why considered | Why rejected as headline |
|---|---|---|
| **H1 mode collapse** | Concrete experiment, surprising finding, directly motivates RL, has a clean figure | (kept as headline) |
| Prior LoRA results (`04_final_report.md`) | Already-established numbers; B1 ties, GRPO ranking signal | These predate the proposal; not "since proposal" |
| System config sweep T1-T6 | 3.3× speedup, methodologically interesting | Infrastructure work — not an RL contribution. Belongs in "enabling work" mention, not headline. |
| Hardware ceiling discovery (FA3/4 on sm_120) | Documented + memory-saved | Same as above. |
| H2 full-FT SFT run | The "real" next training | Not finished by milestone deadline; only step-50+ data available. Belongs in "next steps" or "in progress." |

H1 wins because it's (a) **done**, (b) **surprising** — the obvious read "v38 overfit successfully" was wrong; the curve-space read shows mode collapse, (c) **course-relevant** — directly motivates RL as load-bearing, and (d) **has a figure** (the GT vs predicted curves plot showing visual mode collapse on val).

### Why hypothesis-update gets its own section

The original proposal framed SFT as the primary trainer with GRPO as polish. H1 evidence flips this. The reframed hypothesis — "SFT is fundamentally mode-collapsed; GRPO with curve-quality reward is load-bearing" — is the **single most important sentence in the milestone**. It justifies the entire downstream plan and connects directly to course material (the failure mode of SFT motivating RL). The rubric explicitly asks "are there any changes to the research hypothesis" — so this gets a dedicated heading.

### Why concrete next steps are numbered, not narrative

The graders are CAs grading dozens of milestones. They benefit from scannable structure:
1. H2 (in progress, ~6 h remaining)
2. H3 (GRPO from H2 best ckpt)
3. Paired BCa evaluation on held-out 87-ad test set
4. Fallback (aligner unfreeze) if GRPO doesn't break collapse

Each is **gated, measurable, and time-bounded** — not "we will do RL stuff."

### What I deliberately omitted

- **System-side details.** The T1-T6 sweep + dataloader bitter lesson + worker U-curve are documented at length in `docs/09` and in memory entries. The milestone footnotes them in one sentence (Phase 1 enabled the production-rate training) and moves on. The course is CS224R, not CS149.
- **Audit details.** The 5 fixed config issues (`docs/06_config_audit.md`) and recipe gaps (`docs/07`) are background that informed our settings; the milestone gives them one parenthetical reference, not a section.
- **Methodology citations.** Brier 1950, Graf 1999, Efron 1987 (BCa) — included only in the GRPO-reward deviation note. The detailed metric-protocol rationale (`docs/05_evaluation_protocol_v3.md`) is referenced for the final report, not pasted.
- **Prior LoRA tables.** The `04_final_report.md` headline IBS table (B1, SFT, GRPO-50, ...) doesn't appear in the milestone. We reference the GRPO-50 ρ = +0.22 hook-strength result once — that single number is the proof point that "RL distinguishes ads, SFT does not."

---

## Part 2 — How far are we from the goal?

### The original goal (proposal-era)

Build a video-LLM that predicts second-by-second retention curves $R(t)$ for short-form ads, trained via SFT then GRPO, evaluated against the train-mean baseline B1.

### Goal state checklist

| Component | Status | Evidence |
|---|---|---|
| End-to-end pipeline (data → SFT → GRPO → eval) | ✅ done at LoRA scale | `04_final_report.md` |
| Evaluation protocol (IBS + paired BCa) | ✅ done | `05_evaluation_protocol_v3.md`, `ttcc-eval/` repo |
| Methodology audit | ✅ done (5/22 morning) | `06_config_audit.md`, fixes in commits |
| Full-FT SFT at scale | 🟡 **in progress (H2, ~step 70/450 at writing)** | wandb run `sft_h2_20260523_023635` |
| Mode-collapse diagnosis | ✅ done | H1 results in `10_milestone_report.md` |
| System config tuned for production | ✅ done | `go_viral@dc6201f5` |
| GRPO from full-FT checkpoint | ❌ not done | H3 (next phase) |
| Headline test-set comparison | ❌ not done | requires H3 |

### What's actually missing

The remaining work is **mostly compute time**, not unsolved research questions:
- ~6.5 hours of H2 training
- ~4-6 hours of H3 GRPO
- ~1-2 hours of offline evaluation + writeup polish

If everything proceeds without crashes, we have all four headline pieces — full-FT SFT, full-FT GRPO, B1 baseline, paired BCa CIs — assembled by end of week.

### Risk areas

1. **H2 training divergence or NaN.** Currently looks healthy (loss 0.42 at step 70, smooth descent), but multi-hour runs can fail unexpectedly. Mitigation: 3 checkpoints saved + memory headroom of 40 GiB.
2. **H3 GRPO failure to break mode collapse.** This is the central research question. If H3 also shows R[1] correlation ≈ 0, the project has a meaningful negative result (per the rubric's "failure modes are valid") and we'd pursue the aligner-unfreeze ablation as a follow-up.
3. **Test-set leakage in checkpoint selection.** Without a CoT-formatted val set (we deferred building this for time; `docs/10` explains), we'll pick H2's "best" checkpoint via offline curve-space eval. We must be careful that selection happens on a held-out subset of the *train* set or on a generated val, not on the 87-ad test set.

---

## Part 3 — What experiments do we actually have to report?

In order of milestone-relevance (most → least):

### Tier 1 — must include in milestone

**H1: v38 full-FT capacity test + curve-space diagnosis.** The single experiment done since proposal that has a clean result and a surprising finding. Already in the milestone draft. Provides the figure (`runs/v38_inference/figures/ckpt80_compare.png`), the numbers (per-ad MSE, R[1] corr, std-across-ads vs B1), and the interpretation (mode collapse hidden behind token-CE convergence).

### Tier 2 — should mention briefly

**System config sweep T1-T6.** One sentence in "concrete next steps" — "the optimized system configuration documented in `docs/09` enabled training at 3.3× the naive baseline." Cited as the reason H2 fits in ~6.5 h rather than ~28 h. **Not a research contribution** — but it's the reason we can finish at all.

### Tier 3 — background only

**Prior LoRA results.** Cited once for context: "LoRA-GRPO-50 was the only method with non-trivial across-ad ranking signal (Spearman ρ = +0.22 on hook strength)." This single number corroborates the H1 finding that RL is the load-bearing component.

**Methodology deviations from proposal** (Qwen2.5-Omni-3B, $r = 1 - \text{IBS}$, text-domain curve emission). Mentioned in setup section because the rubric implicitly asks "what's different from your proposal." Doesn't need its own section.

### Tier 4 — don't include

- Hardware research (FA3/4 sm_120 ceiling)
- Audit doc details (the 5 fixed issues)
- Worker-count U-curve (T6a) — interesting but off-topic
- Recipe gaps (`07_recipe_gaps.md`)
- Conditional Q1/Q2+Q3/Q4 LoRA-era decomposition (save for final report)

---

## Recommendation

The milestone draft at `docs/10_milestone_report.md` already follows this rationale. To submit:

1. Verify the figure (`runs/v38_inference/figures/ckpt80_compare.png`) renders well at PDF scale — should be readable as ~3 × 4 inches.
2. Optionally add one sentence to "concrete steps #1" with the latest H2 progress (e.g., "as of submission, H2 is at step ~100/450 with loss ~0.4, no divergence").
3. Convert markdown → LaTeX via the course's milestone template (1 page, 11pt).
4. Submit to Gradescope.

If H2 finishes before submission, **swap "in progress" for "completed; best curve-space ckpt at step XX"** in concrete step 1 — that strengthens the report.
