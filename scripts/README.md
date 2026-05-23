# scripts/

Organized by purpose. Each subdir is one job category.

## Layout

| Dir | Purpose | Examples |
|---|---|---|
| `analysis/` | Curve-space metrics on inference outputs (MSE, R[1] corr, mode-collapse proxy). | `analyze_h1_ckpt_comparison.py`, `plot_v38_gt_vs_pred.py` |
| `bench/` | Training-config micro-benchmarks (5-step warm-up runs, mem polling). | `bench_sft.sh`, `launch_T1b.sh` |
| `data/` | Dataset builders + uploads (CoT distillation, filtering, HF push, token profiling). | `cot_distill_v3_gemini.py`, `filter_val.py`, `push_hf.sh` |
| `diagnostic/` | Variance / distribution diagnostics that informed the SFT/GRPO design. | `diag1_variance.py`, `diag5_compare.py` |
| `infer/` | vLLM-backed inference scripts (one per checkpoint). | `infer_v38_ckpt50.sh`, `h1_v38_ckpt50_run.sh` |
| `overnight/` | Long-run orchestration + log-only watchdogs. **No auto-relaunch** (see auto-recovery memory). | `overnight_orchestrator.sh`, `v3_watchdog.sh` |
| `utility/` | One-off / setup / bootstrap scripts; "did the job, kept for reference." | `bootstrap.sh`, `ckpt_mirror.sh`, `setup_cc.sh` |
| `viz/` | Plotting / figure generation (separate from `analysis/` to keep the pipeline scripts clean). | `qualitative.py`, `saturation.py`, `final.py` |

Top-level scripts at `scripts/*.{py,sh}` predate the categorization (`cot_distill.py`, `eval_one.py`, `prepare_dataset_nocot.py`, `status.sh`, etc.) and are the canonical entry points referenced by the README + docs. Don't move them without updating references.

## Naming convention

- Shell scripts: `verb_noun.sh` (e.g., `infer_v38_ckpt50.sh`).
- Python scripts: `verb_noun.py` (e.g., `analyze_h1_ckpt_comparison.py`).
- One-off / experiment-specific scripts include the experiment tag (`v38`, `T1b`, `h1`, etc.) so future-you can tie them back to the journal.

## Adding a new script

1. Pick the right subdir (see table above).
2. Use canonical paths in the script (`/home/ubuntu/ttcc-rl/runs/<exp>/...`) not `/tmp` — see [feedback-make-it-right-first-time](../../.claude/projects/-home-ubuntu/memory/feedback_make_it_right_first_time.md).
3. If it produces outputs, write them to `runs/<exp-name>/`, not `/tmp` or `/opt/dlami/nvme/ssm-out/`.
4. If it depends on a patch to ms-swift, add the patch to `patches/` and reference it in `RUNBOOK.md`.
