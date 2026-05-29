# CoT Data Generation Runbook (Gemini teacher)

Colleague-facing runbook for regenerating the **leak-free Chain-of-Thought (CoT)** training
data for the TTCC retention-curve project, using a **Gemini teacher on Google Vertex AI**.

This is the pipeline that produced the public dataset
[`liangyuch/ttcc-cot`](https://huggingface.co/datasets/liangyuch/ttcc-cot). Every step below is
the actual code in this repo — file paths are relative to the repo root unless absolute.

> **What "leak-free" means here:** the teacher writes a qualitative `Content / Drops / Reasoning`
> analysis and is **never shown** the ground-truth retention curve, R values, or any drop
> percentages. The ground-truth curve `R_true` is carried through the output JSONL for downstream
> SFT, but it never enters the teacher's prompt. This is what prevents the label-in-input leak.

---

## 0. Prerequisites (the two that actually gate a fresh machine)

1. **Google Vertex AI access.**
   - GCP project (the repo default is `vizzylabs-ai-prod`, location `global`).
   - A service-account JSON key, exported as `GOOGLE_APPLICATION_CREDENTIALS`
     (the launchers expect `/home/ssm-user/work/vizzy-sa.json`).
   - Vertex AI API enabled; the SA needs `Vertex AI User` (and `Storage Admin` for the
     oversize/GCS pass in step 3).
   - Override project/location with `--project <id> --location <loc>`.

2. **The source TTCC parquet** at `WORK/data/ttcc/data/train-*-of-*.parquet`
   (`WORK = /home/ssm-user/work` — hardcoded in the script; change it there if your layout
   differs). Required columns: `ad_id, duration, retention_curve, split, video_local_path`
   (the last carries the embedded MP4 bytes). The script extracts each MP4 to
   `WORK/data/videos/train/<ad_id>.mp4` on first run.
   - ⚠️ **This must be the same crawl the project trained on** (the `763…`-style `ad_id`s).
     The public `liangyuch/ttcc-v0_2_0` is a *different* crawl (different `ad_id`s) and will
     **not** reproduce the same CoT set. Get the original parquet from the project owner.

3. **Python env** (a venv at `WORK/venv` in the launchers) with:
   `google-genai`, `google-cloud-storage`, `pyarrow`, `numpy`, `huggingface_hub`.

---

## 1. The generator — `scripts/data/cot_distill_v3_gemini.py`

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/vizzy-sa.json

# Models:  --model flash  -> gemini-3.5-flash   (what the published ttcc-cot used)
#          --model pro     -> gemini-3.1-pro-preview
python scripts/data/cot_distill_v3_gemini.py --model flash --pilot 5 --concurrency 5 \
    --out work-out/cot/v3_flash_pilot.jsonl        # sanity: 5 ads, check creds + format

python scripts/data/cot_distill_v3_gemini.py --model flash --full --concurrency 4 \
    --out work-out/cot/v3_flash_full.jsonl         # full inline run
```

**Defaults (all overridable):** `--temperature 0.4 --top-p 0.95 --max-tokens 512 --seed 0`,
`--concurrency 4 --retries 3`. Thinking is disabled (`thinking_budget=0`) so the output is exactly
the three labeled lines. Resumable sharding: `--start-idx N --stride K --max-ads M`.

**How it selects ads (`build_manifests`):** reads every `train-*` parquet shard, keeps
`split == "train"`, computes the horizon `T` (T_MIN=5, T_MAX=60; rejects ads whose
`round(duration)` vs `len(curve)-1` mismatch by >1), normalizes the curve as `c[:T+1]/c[0]` with a
5e-3 monotonicity tolerance, extracts the MP4, and drops anything malformed.

**Inline cap:** videos ≤ **20 MB** are sent inline (`Part.from_bytes`). Larger ones are **skipped**
and logged `OVERSIZE` (~38 of ~717 ads) — handled in step 3.

**Output schema:** JSONL, one row per ad: `{ad_id, T, R_true, raw}` where `raw` is the 3-line CoT.

### The prompt (verbatim — this is the leak-free contract)

**System:**
> You are a careful analyst of short-form video ad engagement. You will watch and listen to a short
> ad. Your task is to predict, based ONLY on what you actually see and hear, which moments are most
> likely to lose viewer attention. You do NOT have access to the retention curve, drop percentages,
> or any audience-measurement numbers. Never invent or reference such numbers. Ground every claim in
> a concrete on-screen or audio event.

**User(T):** asks for **exactly three labeled lines** —
`Content:` (one sentence on what's shown), `Drops:` (2–3 specific seconds where attention lapses,
each tied to a concrete observed event, seconds in `[0, T]` only), `Reasoning:` (one sentence on
why). Strict rules: no R value / percentage / fraction / "drops to" phrasing; integer seconds only.

---

## 2. End-to-end pipeline (in order)

| # | Command | What it does |
|---|---------|--------------|
| 1 | `python scripts/data/cot_distill_v3_gemini.py --model flash --pilot 5 ...` | Pilot (5 ads) — verify creds + output format |
| 2 | `python scripts/data/cot_distill_v3_gemini.py --model flash --full ...` | Full inline run (≤20 MB videos) |
| 3 | `python scripts/data/cot_distill_oversize.py` | The >20 MB videos: stage to GCS bucket `ttcc-cot-staging-191aab`, call Gemini via `Part.from_uri(gs://...)`. ⚠️ this file hardcodes `MODEL=gemini-3.1-pro-preview` — edit it to `gemini-3.5-flash` if matching flash. |
| 4 | `bash scripts/data/merge_and_push.sh` | Merge inline + oversize into one canonically-ordered JSONL (`*_merged.jsonl`) |
| 5 | `bash scripts/data/qc_v3.sh` | QC via `scripts/qc_cot.py`: enforces the 3-line format, no R-leak, seconds within `[0,T]` → `*_qc_report.json` |
| 6 | `bash scripts/utility/regen_failures.sh` | Re-runs QC-failed ads with `seed=1` + a tightened prompt → `*_merged_clean.jsonl` |
| 7 | `bash scripts/data/finalize_and_push.sh` | Drops the few unfixable ads → `*_merged_final.jsonl`, runs final QC (expect 0 fatal) |
| 8 | `bash scripts/data/push_final.sh` | Pushes JSONL + README → `liangyuch/ttcc-cot` (HF token from `~/.cache/huggingface/token`) |
| 9 | `bash scripts/utility/cleanup_gcs.sh` | Deletes the GCS staging bucket |

Inspect-only helper: `bash scripts/data/check_hf.sh` lists the current `ttcc-cot` files.

---

## 3. Affiliated utility files (what each is for)

| File | Purpose |
|------|---------|
| `scripts/data/cot_distill_v3_gemini.py` | **The generator.** Vertex/Gemini, inline videos. |
| `scripts/data/cot_distill_oversize.py` | Same prompt/model, but for >20 MB videos via GCS `Part.from_uri`. |
| `scripts/utility/run_pilot_flash.sh` / `run_pilot_pro.sh` | One-line launchers for the 5-ad pilot (flash / pro). |
| `scripts/data/merge_and_push.sh` | Merge inline + oversize JSONLs in canonical order. |
| `scripts/data/qc_v3.sh` + `scripts/qc_cot.py` | QC report. **Note:** `qc_cot.py` is referenced by the launchers but is **not committed in this repo snapshot** — it lived in the box's `work/scripts/`. Recover it from the box, or reimplement the three checks above. |
| `scripts/utility/regen_failures.sh` | Re-generate the handful of QC failures (seed=1 + stricter prompt). |
| `scripts/data/finalize_and_push.sh` | Drop unfixable ads + final QC. |
| `scripts/data/push_hf.sh` / `push_final.sh` | Upload to `liangyuch/ttcc-cot`. |
| `scripts/data/check_hf.sh` | List current HF dataset files. |
| `scripts/utility/cleanup_gcs.sh` | Tear down the GCS staging bucket. |

---

## 4. Honest caveats (read before trusting byte-for-byte reproduction)

1. **Output schema vs the published dataset.** This generator writes `{ad_id, T, R_true, raw}`.
   The live `liangyuch/ttcc-cot` exposes `{ad_id, cot, error, model, prompt_version,
   input_tokens, output_tokens, thinking_tokens}` with `model=gemini-3.5-flash`,
   `prompt_version=v6`. So the published data is a **later "v6/flash" iteration** that renamed
   `raw`→`cot` and added per-row metadata. That thin schema-mapping step is **not captured in this
   (2026-05-23) repo snapshot** — the core Gemini generator here *is* the engine, but if you need
   the exact published columns you must add the rename + metadata step.

2. **Use `--model flash`** to match what's on HF. The repo's `*_pro_*` launchers and the final push
   commit used `pro` (gemini-3.1-pro-preview, "714 rows post-QC"); the *published* dataset is flash.

3. **`qc_cot.py` is not in this repo** (see the table). Without it, steps 5–7 can't run as-is.

4. **Source parquet provenance** (see §0.2): the public `ttcc-v0_2_0` has different `ad_id`s than
   the crawl this CoT was generated against — use the original project parquet.

---

*Generated 2026-05-29 from a code read of `cot_distill_v3_gemini.py` + the affiliated utilities.
Pipeline overview also in this repo's `README.md` (steps 0–6) and `RUNBOOK.md`.*
