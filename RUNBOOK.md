# TTCC runbook

How to actually run things in this project. Keep terse — verbose explanations belong in `docs/`.

## Repo layout

```
ttcc-rl/
├── docs/                          # method writeups + session journals
│   ├── 0[1-7]_*.md                # method, configs, results, audit
│   ├── 08_session_journal_*.md    # daily journal: 2026-05-22
│   ├── 09_session_journal_*.md    # daily journal: 2026-05-22 afternoon
│   └── overnight_runlog.md
├── scripts/
│   ├── bench/                     # config-sweep harness (T1a-T2b)
│   │   ├── bench_sft.sh           #   ← 5-step micro-run with mem polling
│   │   └── launch_T1b.sh          #   ← T1b config (Liger ON)
│   ├── infer/                     # vLLM inference scripts
│   │   ├── infer_v38_ckpt80.sh
│   │   ├── infer_v38_ckpt50.sh
│   │   └── h1_v38_ckpt50_run.sh
│   ├── analysis/                  # curve-space analysis scripts
│   │   ├── analyze_v38.py
│   │   ├── analyze_h1_ckpt_comparison.py
│   │   ├── plot_v38_gt_vs_pred.py
│   │   └── filter_long_rows.py
│   ├── prepare_dataset*.py        # data builders
│   ├── eval_one.py                # eval pipeline
│   └── ...
├── runs/                          # one subdir per experiment
│   ├── v38_inference/             # H1 outputs (ckpt-50 vs ckpt-80)
│   │   ├── input_{train,val}.jsonl
│   │   ├── ckpt{50,80}_preds_{train,val}.jsonl
│   │   └── figures/*.png
│   └── config_sweep/              # T1a-T2b benchmark logs
│       └── bench_B0.log
└── patches/                       # local diffs against /home/ubuntu/go_viral/swift
    └── swift_n_try_fetch_30.patch
```

## Environment assumptions

| What | Where | Notes |
|---|---|---|
| Python venv | `/opt/dlami/nvme/work/swift_venv` | owned by ssm-user; root or `sudo -u ssm-user` for write access |
| Model | `/home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B` | |
| Data (CoT) | `/home/ssm-user/work/data/ttcc_swift_v2cot/` | 717 train + 101 val |
| Data (noCoT) | `/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/` | filtered 116/20 (overfit test); originals at `*.original.jsonl` |
| Checkpoints | `/opt/dlami/nvme/ssm-out/` | timestamp-versioned by ms-swift |
| GPUs | 2× RTX PRO 6000 Blackwell (sm_120, ~95 GB each) | |
| flash-attn | 2.8.3, compiled-from-source for sm_120 | FA3 wheel from windreamer is sm_80+sm_90a only — DOES NOT WORK on sm_120 |

## Installation / first-time setup

### Flash Attention 2.8.3 (must compile from source on sm_120 + cu130 + torch 2.11)

No prebuilt wheel covers our combo (Blackwell sm_120 / CUDA 13.0 / torch 2.11.0 / Python 3.12). Dao-AILab official releases stop at torch 2.10 for cu13; community wheels (mjun0812) have cu130torch2.11 only for cp313 / aarch64; FA3 wheels (windreamer) are sm_80+sm_90a only; FA4 needs Hopper/B200 TMEM. **Must compile from source.**

The exact command we used (run as `ubuntu`, the venv-owning user):

```bash
MAX_JOBS=12 \
TORCH_CUDA_ARCH_LIST="12.0" \
CUDA_HOME=/usr/local/cuda-13.0 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
/opt/dlami/nvme/work/swift_venv/bin/pip install --no-build-isolation -v flash-attn==2.8.3
```

**Flag rationale:**

| Flag | Why |
|---|---|
| `MAX_JOBS=12` | flash-attn's setup.py formula is `min(cpu_count//2, free_mem_gb/9)` — on our box that's `min(24, 11) = 11`; 12 is safe. Each nvcc job uses ~3.5–5 GB RAM peak; don't exceed memory budget. |
| `TORCH_CUDA_ARCH_LIST="12.0"` | sm_120 only. Set to `"8.0;9.0;10.0;12.0"` for a fat multi-arch binary (~978 MB instead of ~244 MB). |
| `CUDA_HOME=/usr/local/cuda-13.0` | System default `nvcc` was cuda-13.2; force match with the torch wheel's build cuda. |
| `PATH=/usr/local/cuda-13.0/bin:$PATH` | Same reason — ensures nvcc/ptxas from cuda-13.0 are found first. |
| `FLASH_ATTENTION_FORCE_BUILD=TRUE` | Bypasses pip's wheel-cache lookup; without it pip may find an unusable cached wheel from another venv's torch version (the C++ ABI changes across torch minor versions). |
| `--no-build-isolation` | Uses the venv's existing torch instead of installing a fresh one in an isolated build env. |
| `-v` | Verbose; surfaces nvcc lines so you can see arch_list expansion. |

**Traps:**

- **Don't use `sudo -u ssm-user`.** The venv is `ubuntu`-owned, but adding sudo breaks env-var inheritance through ninja's subprocess tree → only 1 nvcc runs at a time → projected build time 175 min instead of 18.
- **Don't reuse cached wheels from other venvs.** A wheel compiled against torch 2.8 fails to import with `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...` because torch's C++ ABI changed across minor versions.
- **Don't try FA3 / FA4.** sm_120 (RTX 5090, RTX PRO 6000) physically lacks the TMEM silicon that FA4 needs and the TMA semantics FA3 needs. They will never work; see `docs/09_session_journal_20260522_afternoon.md` and the `ttcc-sm120-attention-kernel-landscape` memory entry.

**Observed:**
- Build time: ~18 min wall (with MAX_JOBS=12: 11 parallel nvcc + ~44 cicc workers; load avg 11-34 on 48 logical CPUs).
- Sub-linear speedup ≈9.3× over serial (init overhead + dep-graph serialization).
- Built `.so` size: 978 MB (multi-arch fat binary: sm_80, 90, 100, 120) — vs ~244 MB single-arch.
- Smoke test (seqlen=12288, bf16, head_dim=128, fwd+bwd, sm_120): pass, 0.66 GB peak GPU.

### Patch swift's n_try_fetch (default 10 → 30)

```bash
cd /home/ubuntu/go_viral
git apply /home/ubuntu/ttcc-rl/patches/swift_n_try_fetch_30.patch
# Verify:
grep "n_try_fetch: int = " swift/dataset/utils.py     # should say 30
```

Reason: at high `FPS_MAX_FRAMES` + tight `max_length`, the default 10-retry budget can exhaust on a small run of bad rows and abort training with a misleading "increase max_length" error. 30 gives resilience without masking real issues.

## Required env vars (every Qwen2.5-Omni-3B job)

```bash
export FPS_MAX_FRAMES=60          # SFT — fewer frames silently drops temporal coverage
# or FPS_MAX_FRAMES=24 for GRPO/inference (match audit)
export MAX_PIXELS=200704          # 448×448 input
export VIDEO_MAX_PIXELS=200704
export VIDEO_MAX_TOKEN_NUM=16384  # SFT
export ENABLE_AUDIO_OUTPUT=False  # disables Talker (~1.5 GB GPU saved)
```

## SFT (full-FT, audit-ambitious config)

```bash
# Recipe: zero3, no offload, vit_gradient_checkpointing, flash-attn-2, Talker off
# Defaults from sft_v2cot_full.sh: 10 epochs, lr=1e-5, eval+save every 50 steps
cd /home/ubuntu/go_viral/examples/train/grpo/qwen2_5_omni_ttcc
SFT_DATA=/home/ssm-user/work/data/ttcc_swift_v2cot/ttcc_train_sft.jsonl \
  OUT=/opt/dlami/nvme/ssm-out/$(date +%Y%m%d_%H%M%S)_sft_full \
  ./sft_v2cot_full.sh
```

## Inference (full-FT checkpoint via vLLM)

```bash
bash /home/ubuntu/ttcc-rl/scripts/infer/infer_v38_ckpt50.sh \
  <input.jsonl> <output.jsonl>
# Edit CKPT= at the top of the script for a different checkpoint
```

## Curve-space analysis (after inference)

```bash
cd /home/ubuntu/ttcc-rl
/opt/dlami/nvme/work/swift_venv/bin/python scripts/analysis/analyze_h1_ckpt_comparison.py
# Outputs: per-ad MSE, R[1] correlation, std-across-ads (mode-collapse proxy)
```

## Config-sweep micro-benchmark

```bash
# Run any single config in 5 steps (~10 min total)
LABEL=mytest LIGER=1 ZERO=zero3 VIT_CKPT=true BS=1 GA=8 \
  bash /home/ubuntu/ttcc-rl/scripts/bench/bench_sft.sh
# Output: bench log + memory.csv at /opt/dlami/nvme/ssm-out/bench_<LABEL>_<time>/
```

## What NOT to do

- **No `--attn_impl flash_attention_3`** — wheel is incompatible with sm_120 Blackwell.
- **No `truncation_strategy=left`** — chops vision tokens and corrupts training (loss diverges).
- **No `kill -0` cross-user** for liveness checks — returns EPERM, misreads live procs as dead.
- **No auto-relaunch watchdogs** for overnight runs — log-only watchers; humans decide restart.
- **No converting JSONL to parquet/arrow** — ms-swift consumes JSONL natively.
- **No deleting `*.original.jsonl`** from data dirs — they preserve pre-filter state.

## Quick-look bookmarks

| Topic | File |
|---|---|
| Method writeup | `docs/01_method.md` |
| Config audit (Β, num_generations, FPS) | `docs/06_config_audit.md` |
| Recipe gaps (val split, CoT audit) | `docs/07_recipe_gaps.md` |
| Today's session journal | `docs/09_session_journal_20260522_afternoon.md` |
| v38 run summary (capacity test) | `docs/08_session_journal_20260522.md` |
| Final report (LoRA-era results) | `docs/04_final_report.md` |
