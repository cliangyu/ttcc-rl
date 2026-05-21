# ttcc-rl

**Training pipeline** for the TTCC retention-curve project. SFT distillation + GRPO RL on top of Qwen2.5-Omni-3B, predicting per-second viewer retention $R(t)$ for short-form video ads.

## What this repo is and isn't

| repo | role | remote |
|---|---|---|
| **`ttcc-rl`** *(this one)* | training pipeline: CoT distillation, SFT, GRPO, inference glue | local-only (see `TOMORROW.md` for the push-to-remote decision) |
| **`ttcc-eval`** | evaluation protocol, metric implementations, predictions-parquet consumer | [`cliangyu/ttcc-eval`](https://github.com/cliangyu/ttcc-eval) |
| **`ttcc-inference`** | zero-shot inference + prompt iterations | [`cliangyu/ttcc-inference`](https://github.com/cliangyu/ttcc-inference) |
| **`go_viral`** | ms-swift fork with the training launcher scripts and reward plugins, on the `ttcc-rl` branch | [`cliangyu/go_viral:ttcc-rl`](https://github.com/cliangyu/go_viral/tree/ttcc-rl) |

## The math in one line

$$
\mathcal{L}_{\text{SFT}} = -\sum_\tau \log p_\theta(\tau_{\text{token}} \mid \tau_{<}) \qquad\Longrightarrow\qquad r_i = 1 - \text{IBS}_i, \quad \text{IBS}_i = \frac{1}{T_i + 1}\sum_{t=0}^{T_i} \big(\hat R_i(t) - R_i(t)\big)^2
$$

**SFT** teaches the model to imitate teacher-generated CoT rationales + numerical curves on the same assistant target. **GRPO** then improves the policy under a strictly proper scoring rule (Brier 1950, Graf 1999) — IBS = time-averaged squared error on probabilistic survival predictions; reduces to MSE in our no-censoring setting.

Full protocol description: [`ttcc-eval/docs/09_minimal_protocol.md`](https://github.com/cliangyu/ttcc-eval/blob/main/docs/09_minimal_protocol.md).

## Pipeline at a glance

```
TTCC parquet (raw retention curves)
        │
        ▼
[1] cot_distill.py        — Qwen3-Omni-30B-A3B-Instruct teacher writes CoT
        │                  conditioned on ground-truth curve (outcome-conditioned
        │                  rationale generation; not instruction back-translation)
        ▼
[2] prepare_dataset.py    — TTCC parquet × CoT JSONL → ms-swift JSONL
        │                  (assistant target = Content + Drops + Reasoning + Curve)
        ▼
[3] sft.sh                — LoRA SFT on Qwen2.5-Omni-3B (90 steps, FPS_MAX_FRAMES=32)
        │
        ▼
[4] grpo.sh               — GRPO from SFT, r = 1 − IBS + 0.2·format
        │                  (num_generations=2, β=0.04, vLLM colocate rollouts)
        ▼
[5] infer_trained.sh      — greedy inference on test set → predictions parquet
        │
        ▼
       ttcc-eval          — IBS, BS(3), BS_end, Spearman, paired BCa CI
```

## Directory layout

```
ttcc-rl/
├── README.md            ← you are here
├── TOMORROW.md          ← open decisions for next session
├── pyproject.toml       ← editable install of the ttcc_rl Python package
├── src/ttcc_rl/         ← canonical Python helpers
│   ├── parser.py        ← curve extraction from free-text model output
│   └── postprocess.py   ← swift infer JSONL → predictions parquet
├── scripts/             ← training entry points
│   ├── cot_distill.py   ← teacher CoT generation
│   ├── prepare_dataset.py        ← (symlink) TTCC parquet → ms-swift JSONL with CoT
│   ├── prepare_dataset_nocot.py  ← variant: assistant target = Curve only
│   ├── infer_trained.sh          ← ms-swift infer + postprocess + eval call
│   ├── eval_one.py               ← project-specific per-method headline + paired BCa
│   ├── test_hypotheses.py        ← ttcc-eval docs/07 hypothesis-verification harness
│   ├── grpo_signal.py            ← per-ad GRPO-vs-SFT decomposition (analysis)
│   ├── status.sh                 ← GPU / disk / log snapshot
│   └── viz/                      ← all the figures referenced in docs/*
├── go_viral_overlay/    ← symlinks into the live go_viral ttcc-rl branch
│   └── examples/train/grpo/qwen2_5_omni_ttcc/{sft,grpo,rloo}.sh + _common.sh
└── docs/                ← project-specific reports (eval protocol lives in ttcc-eval)
    ├── 01_method.md                ← implementation deltas vs milestone §2-3
    ├── 02_experiment_configs.md    ← every hyperparameter + assumption
    ├── 03_conditional_results.md   ← novelty-quartile decomposition (where methods earn keep)
    ├── 04_final_report.md          ← TL;DR + headline table for an external reader
    ├── 05_evaluation_protocol_v3.md ← rich layered v3 protocol with results
    └── overnight_runlog.md          ← narrative of the 2026-05-20 SFT+GRPO run
```

## Quickstart

```bash
# (0) CoT distillation
python scripts/cot_distill.py --model INSTRUCT --full --out work-out/cot/full.jsonl

# (1) ms-swift datasets
python scripts/prepare_dataset.py --cot-jsonl work-out/cot/full.jsonl --out-dir data/ttcc_swift

# (2) SFT seed
bash go_viral_overlay/examples/train/grpo/qwen2_5_omni_ttcc/sft.sh

# (3) GRPO from SFT
SFT_CKPT=<sft-ckpt> bash go_viral_overlay/examples/train/grpo/qwen2_5_omni_ttcc/grpo.sh

# (4) Infer + eval (minimal 6-number protocol)
bash scripts/infer_trained.sh <grpo-ckpt> grpo work-out/preds_grpo.parquet
python <ttcc-eval>/scripts/minimal_eval.py \
    --preds preds_b1.parquet:B1 preds_sft.parquet:SFT preds_grpo.parquet:GRPO \
    --gt data/ttcc_swift/ttcc_test.jsonl --ref B1
```

## Headline result (n = 87 test ads, 2026-05-20)

| method | $n_{\text{par}}$ | $\overline{\text{IBS}}$ | $\text{BS}(3)$ | $\rho_H$ | $\text{BS}_{\text{end}}$ | $\rho_C$ |
|---|---:|---:|---:|---:|---:|---:|
| B1 (train-mean) | 87/87 | 0.0083 | 0.0200 | n/a | 0.0027 | $-0.029$ |
| SFT (CoT, 1 epoch) | 87/87 | 0.0094 | 0.0258 | $+0.165$ | **0.0017** | $-0.054$ |
| **GRPO** (50 steps) | 87/87 | **0.0083** | 0.0213 | $\mathbf{+0.217}$ | **0.0017** | $+0.010$ |
| SFT-Extended (3 ep) | 87/87 | 0.0093 | 0.0260 | $+0.120$ | **0.0016** | $-0.051$ |
| GRPO-Extended (1 ep) | 87/87 | 0.0091 | 0.0264 | $+0.100$ | 0.0017 | $-0.035$ |
| SFT-noCoT | **66/87** | 0.0115 | 0.0346 | $-0.035$ | 0.0019 | $+0.199$ |

Paired BCa $\Delta\text{IBS}$ vs B1: every method tied on the full set. On Q2+Q3 (43 moderate-novelty ads), GRPO beats B1 by $\text{BSS} = +0.42$ with CI excluding 0 — see [`docs/03_conditional_results.md`](docs/03_conditional_results.md).

## Math foundations (citations)

| concept | source | role |
|---|---|---|
| Strictly proper scoring | Brier 1950 | shows squared error on probabilities is unbiased; can't be gamed by emitting any monotone curve |
| Survival-form Brier | Graf et al. 1999 | integrates Brier over time; reduces to MSE in no-censoring case |
| Calibration ↔ discrimination decomposition | Murphy 1973; Bröcker 2009 | $\text{IBS} = \text{REL} - \text{RES} + \text{UNC}$; separates "magnitude" from "ranking" errors |
| Climatology skill score | Mason 2004 | $\text{BSS} = 1 - \overline{\text{IBS}}_{\text{method}} / \overline{\text{IBS}}_{B_1}$; unit-free, $%$-improvement |
| Paired bootstrap CI | Efron 1987 | bias-corrected accelerated; necessary for $\Delta\text{IBS}$ significance at $n = 87$ |
| Hook rate ($R(3)$) | TikTok / Meta industry | actionable point metric for ad creators |
| Endpoint completion ($R(T_i)$) | same | the CPV / CPCV billing unit |

## Why we deviated from the milestone

The milestone proposed:
- Architecture: VideoLLaMA2.1-7B + hazard head ($\hat\lambda(t) = \text{softplus}(W h)$).
- SFT loss: log-hazard MSE + CoT cross-entropy.
- GRPO reward: $r = \rho_S$ (within-ad Spearman).
- Teacher: Gemini-2.5 (API).

We replaced these with: Qwen2.5-Omni-3B + LM head (text-domain prediction), plain LM cross-entropy on the assistant target, $r = 1 - \text{IBS}$ (because $\rho_S$ is broken — any monotone curve scores $\bar\rho \approx +1$, including the trivial train-mean baseline), and Qwen3-Omni-30B-A3B-Instruct teacher (Apache-2.0, local). Full delta accounting and the verified critique of $\rho_S$ are in [`docs/01_method.md`](docs/01_method.md) and [`ttcc-eval/docs/07_proper_scoring_rule_revision.md`](https://github.com/cliangyu/ttcc-eval/blob/main/docs/07_proper_scoring_rule_revision.md).

The milestone's three named baselines (SFT-MSE, SFT-Hazard+CoT, GRPO-$\rho_S$) remain unimplemented; see [`TOMORROW.md`](TOMORROW.md) for the decision to revisit them.

## Open decisions

See [`TOMORROW.md`](TOMORROW.md) for the explicit set of pending design questions raised by today's work — reward redesign, $\beta$, num_generations, recalibration experiment, routing/ensemble, and the ttcc-rl repo fate.
