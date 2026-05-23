# Session journal — 2026-05-22 — SFT-noCoT v3 overfit test

End-to-end log of today's training expedition: what we tried, what we observed, what the diffs were, what we learned. Written progressively (so structure may drift — that's OK, accuracy > polish).

## Goal of the run

Test whether full-FT of Qwen2.5-Omni-3B can overfit 717 train ads. This is the **diagnostic** prerequisite to running the corrected-recipe baseline:

- If yes (train loss → ~0, val loss diverges) → full-FT has the capacity, our RL hyperparameter pain is downstream
- If no → we need to scale up data or rethink the recipe before any RL

First run with a real val split (101 ads from raw TTCC parquet's `split=val`) — earlier sessions had no val tracking, which made all "best checkpoint" choices test-tuned.

## Phases tried (chronological)

### Phase 1 — v1 launch (06:51)

**Config:** audit settings — `FPS_MAX_FRAMES=60`, `VIDEO_MAX_TOKEN_NUM=8192`, `max_length=8192` (unchanged), `lazy_tokenize=true`, no flash-attn, Talker disabled.

**Result:** crashed at startup with `ValueError: Failed to retrieve the dataset. You can avoid this issue by increasing max_length or modifying truncation_strategy` — silent-drop retry loop in `swift/dataset/utils.py:108` exhausted.

**Root cause:** token budget math at FPS=60 + audio + system + user + assistant target ≈ ~10,100 tokens worst case, but `max_length` was still 8192. With `truncation_strategy=delete`, every overlong row was dropped; eventually the retry budget ran out.

**Diff vs working 2026-05-21 run:** yesterday had `FPS_MAX_FRAMES=32` + `VIDEO_MAX_TOKEN_NUM=4096`, so worst-case seqlen was ~5800 — fit comfortably in `max_length=8192`. Today's audit raised the video budget but I forgot to raise `max_length`.

**wandb URL:** [liangyuch/ttcc/runs/cw9lyduj](https://wandb.ai/liangyuch/ttcc/runs/cw9lyduj) (died before first step)

### Phase 2 — v2 launch (07:29)

**Config:** added `--max_length 12288`, `--attn_impl flash_attn`, `--lazy_tokenize false`, `--group_by_length true`, `--dataset_num_proc 8`, `OMP_NUM_THREADS=6`.

**Result:** died 14 s in with `ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package for FlashAttention2 doesn't seem to be installed.`

**Root cause:** I assumed `--attn_impl flash_attn` was just an arg toggle; it actually requires the `flash-attn` PyPI package installed in the venv. The 2026-05-21 working run had `attn_impl=None` (SDPA default).

**Diff:** introduced a hard dependency we hadn't installed.

### Phase 3 — Prebuilt wheel hunt (~08:00-08:20)

**What I tried:**
- Dao-AILab/flash-attention official releases — `v2.8.3` has cu12torch{2.4..2.9}, `v2.8.1` has cu13torch2.10 but not torch 2.11
- `mjun0812/flash-attention-prebuild-wheels` community builds — has `cu130torch2.11` but only for cp313 / aarch64
- FA4 beta (pure-python wheel) — **but FA4 doesn't support sm_120** (built around sm_100 datacenter Blackwell TMEM)
- Cached local wheel at `/home/ssm-user/.cache/pip/wheels/.../flash_attn-2.8.3-cp312-...whl` (built against torch 2.8.0+cu128 yesterday)

**Cached wheel install attempt:**

```
ImportError: /opt/dlami/nvme/work/swift_venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so:
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

torch C++ ABI changed across 2.8 → 2.11; the symbol `c10::cuda::c10_cuda_check_implementation` was renamed/removed.

**Verdict:** no viable prebuilt wheel for torch 2.11+cu130+sm_120. Must compile from source.

**Validation:** the sibling venv `/home/ssm-user/work/venv` has flash-attn 2.8.3 built for torch 2.8.0+cu128 that runs on our sm_120 GPU — proves the hardware path works once the binary is correctly built. Confirmed by smoke test there.

### Phase 4 — FA compile attempt 1 (08:07-08:19)

**Config:** `sudo -u ssm-user`, `MAX_JOBS=8`, `TORCH_CUDA_ARCH_LIST=12.0`.

**Observed:** only 1 nvcc at a time, ~5 `.o` files in 12 min. Load average 2.8 on a 48-CPU box. Projected: 175 min total — unacceptable.

**Diagnosis:** flash-attn's `setup.py` calls `NinjaBuildExtension` which uses ninja. Ninja should honor `MAX_JOBS` env. But via `sudo -u ssm-user`'s `-E env`, the env var made it to the pip process (confirmed by reading `/proc/<pid>/environ`) — yet ninja's subprocess apparently wasn't seeing it. Suspected: env-var-forwarding fragility through nested sudo/env/pip/setup.py/cpp_extension/ninja invocation.

**Killed at 08:24.**

### Phase 5 — FA compile attempt 2 (08:24-08:43)

**Config change:** dropped `sudo -u ssm-user` entirely (the venv is owned by `ubuntu`, verified by `ls -ld`; the user-switching was unnecessary). Set `MAX_JOBS=12` (= 24 physical cores × 2 SMT / 4 threads-per-nvcc), `TORCH_CUDA_ARCH_LIST="12.0"`, `CUDA_HOME=/usr/local/cuda-13.0`.

**Observed:**
- 11 parallel nvcc + 44 cicc workers concurrently
- Load average climbed from 3 → 11+ → 34 (much better CPU utilization)
- **9.3× rate vs serial** (3.91 .o/min vs 0.42)
- 73 ninja targets completed in ~18 min
- Smoke test at training seqlen=12288 with backward pass: PASS, 0.66 GB peak

**Diff vs attempt 1:** removing `sudo -u ssm-user` was the actual fix. Direct ninja invocation as the right user honored `MAX_JOBS` correctly.

**Bonus finding:** flash-attn's `setup.py` has its own calibration formula:

```python
max_num_jobs_cores = max(1, os.cpu_count() // 2)        # 24 on our box
max_num_jobs_memory = int(free_memory_gb / 9)            # 11 on our box (~9 GB/job estimate)
max_jobs = min(cores, memory)                            # auto = 11
```

I'd been picking MAX_JOBS=8 from training-data heuristic; the codebase had a better answer (11) sitting in line 506-521 of its own setup.py. Lesson: read the actual code.

### Phase 6 — v3 launch + orchestrator misfire (08:45-08:50)

**Auto-launched** by the chain script (compile → smoke test → SFT launch) at 08:45:26 as wandb run `sft_nocot_full_overfit_20260522_v3`, pid 2911225.

**Within 22 seconds, an auto-recovery orchestrator I'd written misfired catastrophically.**

The bug: the orchestrator's poll-loop was `until ! kill -0 "$pid"; do sleep 60; done`. From the `ubuntu` user, `kill -0 <ssm-user-owned-pid>` returns EPERM (exit code 1), which bash's `until !` interprets as "process is dead, exit loop." So the orchestrator immediately decided v3 was dead, ran its recovery logic (sed-modify the script to fallback config, relaunch). The relaunched ghost failed with `EADDRINUSE` on port 29500 (v3 had it). Orchestrator misclassified the ghost's failure too, ran again. Burned through 3 "attempts" in 1 second, then gave up.

**Damage:**
1. Two ghost training processes (2911770, 2911821) created, both died immediately with port collision
2. Orchestrator's `relaunch()` did `sudo mv sft.log → sft.log.prev.<ts>` mid-stream — split v3's stdout (still open via its tee's file descriptor) onto an orphaned inode
3. Script-on-disk was sed-edited to `FPS_MAX_FRAMES=32, max_length=8192, attn_impl=sdpa` — undoing the audit config

**What was NOT damaged:**
1. v3's running process — its argv was fixed at launch; the script-on-disk edit had no effect
2. v3's GPU allocation — it kept port 29500 throughout
3. v3's checkpoints — output_dir hardcoded in argv

**Recovery actions:**
1. Killed orchestrator (already dead — "gave up")
2. Reverted script-on-disk to v3 config via reverse sed
3. Killed bad watchdog (which would have misread ghost errors in `sft.log` as "v3 crashed")
4. Renamed misleading files: `sft.log → sft.log.ghost_errors`, `sft.log.prev.XXX → sft.log.v3_orphan_inode`
5. Launched simpler v3-scoped watchdog: polls v3 pid + checkpoint emergence (NOT log content), runs val inference on exit

**Documentation correction made later (~09:00):** new watchdog initially had a false-positive — searching the whole `OUT_DIR` it picked up *yesterday's* `v0-20260521-114422/checkpoint-450` as "latest checkpoint." Fixed by scoping `find` to v3's specific subdir `v3-20260522-084539`.

### Phase 7 — v3 startup observation (08:50 onwards)

**State at 16+ min elapsed, no wandb run dir yet, alarming on the surface.**

Built a systematic case enumeration ([scenarios A-H](#case-enumeration-table-below)) to differentiate "healthy slow start" from "stuck/deadlock." Empirical signals checked:

- `wchan=0` (rank 1 not in kernel wait)
- `state=R` (rank 1 actively scheduled)
- CPU time accumulating linearly (1152s in 16 min wall — ~1.2 threads avg active)
- 3.5k voluntary + 6.8k involuntary context switches → progressing
- rank 0 has **9 children** (`dataset_num_proc=8` workers + main)
- rank 1 has **0 children** — doing sequential CUDA-side setup (DeepSpeed/FA kernel autotune)
- `.arrow` cache files written by `datasets.map` → at least one map pass completed
- only open file descriptors on rank 1 are `/dev/nvidia*` — no mp4 / cache files being read right now

**Conclusion:** rank-asymmetric setup, not deadlock. Rank 0 finished its parallel preprocessing fast; rank 1 is sequentially compiling/autotuning FA kernels + DeepSpeed init. GPU 1 at 100% util is CUDA compute, not busy-wait.

**Decision:** wait 15 more min (until 30 min total elapsed). If no wandb v3 run by then, attach `py-spy dump` to rank 1 to see the actual Python stack.

## Case enumeration table (kept for future stuck-diagnosis)

| # | scenario | GPU pattern | CPU pattern | wandb? | likelihood |
|---|---|---|---|---|---|
| A | Pure CPU preprocessing | idle | both ranks busy | No | ❌ rules out (GPU busy) |
| B | Model load extended | brief spike | brief spike | No | ❌ 16 min too long for 3B |
| C | FA first-call autotune | one rank 100% | low | No | 🟡 possible, but seconds not minutes |
| D | First train step very slow | 100% | dataloader workers active | **YES** (wandb inits BEFORE step 1) | ❌ no wandb |
| E | NCCL deadlock | active rank 100% (busy-wait), other 0% | one rank 0% | No | 🟡 partly matches but wchan=0 rules out |
| F | Preprocessing routed through GPU | one rank busy | active rank busy | No | 🟡 plausible |
| G | Stuck waiting on map result | usually idle | low on parent | No | ❌ GPU active |
| H | `group_by_length` length precompute | varies | active rank busy | No | ✅ **best fit** |

**Diagnostic decision rule:**
1. `state` flag in `/proc/<pid>/status`: D (uninterruptible) → kernel/IO issue, R → running OK, S → sleeping/blocked
2. `wchan`: 0 = scheduled, non-0 = waiting for kernel event
3. ctx switches over time: increasing rapidly = real work; flat = stuck
4. children count: matches `dataset_num_proc`? matches `num_workers`?
5. open files: `mp4` paths = video decode; `.arrow` = cache; `/dev/nvidia*` only = compute

## Differences from yesterday's working run

| dimension | 2026-05-21 (worked) | 2026-05-22 v3 (running) |
|---|---|---|
| Talker (token2wav) | enabled (loaded ~833 M unused params) | **disabled** via `ENABLE_AUDIO_OUTPUT=False` |
| `FPS_MAX_FRAMES` | 32 | 60 |
| `VIDEO_MAX_TOKEN_NUM` | 4096 | 8192 |
| `max_length` | 8192 | 12288 |
| `attn_impl` | None (SDPA default) | `flash_attn` 2.8.3 (compiled today for sm_120) |
| `lazy_tokenize` | true | **false** (upfront map) |
| `group_by_length` | false (random) | **true** (length-bucketed batches) |
| `dataset_num_proc` | 1 | 8 (with `OMP_NUM_THREADS=6`) |
| val split tracking | none (`val` JSONL didn't exist) | **101 ads** via `--val_dataset` |
| Wall-clock yesterday | 2 h 35 m / 450 steps | TBD |
| Step time yesterday | 15.65 s/it | TBD |

## 09:18 update — startup is preprocessing-bound (py-spy confirmed)

After 33 min total elapsed with no wandb run, attached `py-spy` to both ranks. The stack traces resolved the ambiguity completely:

**Rank 1 (waiting):**
```
barrier (torch/distributed/distributed_c10d.py:5030)
safe_ddp_context (swift/utils/torch_utils.py:66)
__call__ (swift/dataset/preprocessor/core.py:349)
_encode_dataset (swift/pipelines/train/sft.py:338)
```

**Rank 0 (orchestrating):**
```
_recv (multiprocess/connection.py:398)
iflatmap_unordered (datasets/utils/py_utils.py:610)
map (datasets/arrow_dataset.py:3623)
```

**Rank 0 worker (one of 8):**
```
_read_video_decord (qwen_omni_utils/v2_5/vision_process.py:319)
_new_read_video (swift/model/models/qwen.py:684)
fetch_video (qwen_omni_utils/v2_5/vision_process.py:411)
replace_tag (swift/template/templates/qwen.py:730)
```

**Architecture confirmed:** ms-swift's `safe_ddp_context` is designed so only the master rank runs the `AddLengthPreprocessor` (8 workers via `dataset_num_proc=8`) while non-master ranks wait at a barrier. The bottleneck is **video decoding via decord**: each of 818 rows requires opening an mp4 and decoding up to 60 frames at FPS=1, then resizing to max_pixels=200704. At an estimated 3-20 s/row depending on ad length, 818 rows ÷ 8 workers = 5-35 min preprocessing — the upper end matches what we see.

Rank 1's GPU=100% util is the NCCL barrier kernel spinning (low-latency busy-wait), not deadlock. State R + non-zero wchan would've meant something else.

**Implication for future runs:** `load_from_cache_file=False` (default) means we pay this 40-50 min every time we restart. Next configuration round, set `--load_from_cache_file true` to cache the preprocessed arrow files and skip this on subsequent runs.

## Revised time projections

| phase | morning estimate | revised after py-spy diagnosis |
|---|---|---|
| FA compile | 20 min | done (18 min) ✓ |
| Preprocessing | 10-15 min | **40-50 min** (video decode dominates) |
| Training (450 steps) | 2h 30m – 3h 15m | **3-4.5 h** (FPS=60 = ~2× video decode per step vs yesterday's FPS=32) |
| Val inference | 10 min | 10 min |
| **Total wall** | 3-3.5 h | **4-5.5 h** |

Started chain at 08:45 → expected completion 13:00-14:30.

## 09:40 update — v3 was STUCK, not slow. Killed. Relaunched as v4 (lazy mode).

The "slow" diagnosis was wrong. At 53 min total elapsed I re-checked:

| signal | reading | interpretation |
|---|---|---|
| All 8 workers state | S (sleeping), 0% CPU | Not doing work |
| Total worker CPU since launch (53 min) | 1-6 s each | ~500× less than expected if preprocessing was running |
| iostat nvme | **0 MB/s read** | Not reading any mp4 files |
| Worker py-spy stack | all in `_read_video_decord` | Entered decord, never returned |
| Worker kernel wchan | 0 | Not in kernel wait function |

This is decord hung — likely the first batch of mp4s each worker received caused decord at FPS_MAX_FRAMES=60 to enter some internal lock or busy-wait. No I/O, no CPU, but the function call hasn't returned. Possibly a decord bug at our specific frame count, possibly a corrupt mp4 in the dataset.

**Reasoning trap to remember:** at the 33-min checkpoint, rank 1's CPU activity (utime grew from 1152s → 1988s in 17 min) looked like "we're making progress." But that was the **NCCL barrier kernel busy-waiting** for rank 0 — not actual training work. The barrier kernel keeps GPU+CPU pegged at 100% to minimize latency when the wait condition resolves. So "GPU active + CPU active" does NOT imply progress when one rank is at a barrier. **Always check the worker CPU time, not just the rank-1 metrics.**

**Recovery action (09:40-09:41):**
- Killed v3 process tree cleanly via `pkill -9 -u ssm-user -f swift/cli/sft.py` (also cleared zombie workers holding port 29500 in CLOSE_WAIT)
- Killed old watchdog (pid 2917192)
- Edited `sft_v2cot_full.sh`:
  - `--lazy_tokenize true` (was false)
  - `--group_by_length false` (was true — incompatible with lazy=true, needs precomputed lengths)
  - `--dataset_num_proc 1` (was 8 — no parallel preprocess needed with lazy)
- Launched v4 at 09:41:37, wandb name `sft_nocot_full_overfit_20260522_v4_lazy`
  - Toplevel pid 2925521
  - Run subdir: `v4-20260522-094149`
- Restarted watchdog (pid 2927259) with v4-scoped paths

**v4 vs yesterday's working recipe — what's still different:**

| | yesterday | v4 |
|---|---|---|
| Talker | enabled | disabled |
| FPS_MAX_FRAMES | 32 | **60** ← still aggressive |
| VIDEO_MAX_TOKEN_NUM | 4096 | 8192 |
| max_length | 8192 | 12288 |
| attn_impl | SDPA | flash_attn 2.8.3 |
| lazy_tokenize | true | true ✓ (matches) |
| group_by_length | false | false ✓ (matches) |
| dataset_num_proc | 1 | 1 ✓ (matches) |
| val tracking | none | 101 ads |

**Risk that remains:** if decord hangs on the same problematic frame count at training time (when dataloader workers call `_read_video_decord` per-batch instead of upfront), v4 will hang at the first batch instead of upfront preprocessing. We'd see GPU loaded but utilization at 0% for a while during step 1. If that happens, the next downshift is FPS_MAX_FRAMES=60 → 32 to match yesterday's known-working decord behavior.

## 09:47 update — v4 OOMed during backward pass

v4 ran 6 min then died with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.57 GiB.
GPU 1 has a total capacity of 94.97 GiB of which 4.10 GiB is free.
This process has 90.86 GiB memory in use.
```

90 GB used at moment of crash — pushed past our 95 GB Blackwell budget. Decomposition:
- Model params: ~12 GB (3B × 4 bytes for fp32 grad mirror)
- Optimizer state (zero2 sharded): ~12-16 GB per rank
- Activations w/ grad checkpointing: ~30-40 GB at seqlen 10-12K
- Attention activations (flash-attn helps here): ~5 GB instead of ~20 GB
- Misc (NCCL buffers, Talker-still-not-fully-freed, etc.): ~8-10 GB

**Flash-attn helped attention but not the other components.** FFN activations + grad buffers scale linearly with seqlen; they alone pushed past budget at max_length=12288.

For comparison, yesterday at max_length=8192 → 78/80 GB peak. Going to 12288 = 1.5× seqlen = pushed everything ~1.5× → OOM.

## 09:48 update — v5 launch (3rd retry; yesterday's recipe + safe audit additions only)

Reverted to yesterday's video config but kept Talker disable + flash-attn + val tracking.

| | yesterday v0 (worked) | v5 (running) |
|---|---|---|
| Talker | enabled | **disabled** (~1.5 GB saved) |
| FPS_MAX_FRAMES | 32 | 32 ✓ |
| VIDEO_MAX_TOKEN_NUM | 4096 | 4096 ✓ |
| max_length | 8192 | 8192 ✓ |
| attn_impl | SDPA | **flash_attn** 2.8.3 (sm_120) |
| lazy_tokenize | true | true ✓ |
| group_by_length | false | false ✓ |
| dataset_num_proc | 1 | 1 ✓ |
| val_dataset | none | **101 ads** ✓ |

This recipe is "yesterday + only the audit changes that don't risk OOM." We give up the FPS=60 tail visibility for this overnight run — that'll come back in a future run with packing or smaller max_pixels. The objective (overfit test on full-FT) doesn't require FPS=60.

**v5 details:**
- wandb run: `sft_nocot_full_overfit_20260522_v5_safe`
- toplevel pid 2943477
- subdir `v5-20260522-094847`
- launched 09:48:35

**Expected wall time:** ~2h 30m (yesterday's recipe) - 10-15% (flash-attn speedup) = **~2h 10m for training + 10 min val inference = ~2h 20m total**. Expected completion ~12:10-12:30.

## Today's incident postmortem in one table

| # | what failed | symptom | root cause | recovery |
|---|---|---|---|---|
| v1 | dataloader retry loop exhausted | `Failed to retrieve the dataset` | max_length=8192 too small for FPS=60 video tokens | raised max_length to 12288 |
| v2 | model load failed | `ImportError: FlashAttention2 ... not installed` | added `--attn_impl flash_attn` without installing | compiled flash-attn 2.8.3 from source |
| (orchestrator) | wild fire of attempts | sed-modified script, port conflicts | `kill -0` cross-user EPERM misread as death | killed orchestrator, restored script |
| v3 | stuck inside decord | 0% CPU, 0 disk I/O, no progress for 50 min | FPS=60 + 8-worker dataset_num_proc triggered decord hang | switched to lazy_tokenize=true |
| v4 | OOM during backward | `tried to allocate 6.57 GB / 4.10 GB free` | max_length=12288 + FPS=60 too big for 95 GB budget | reverted max_length=8192, FPS=32 |

The thread: each launch changed multiple things. The lesson keeps repeating itself — **change one variable per launch**. v5 is the right experiment because it's "yesterday + flash-attn + Talker off + val", changes that are minimally entangled and individually known-safe.

## 09:53-10:30 update — three more failures, then the user's first-principles reset

### v5 → v8 timeline (all crashed before any checkpoint)

| run | launched | recipe diff vs prev | failure | wall time |
|---|---|---|---|---|
| **v5** | 09:48 | `lazy_tokenize=true` + FPS=32, ml=8192 (yesterday-safe + FA + Talker off + val) | `Failed to retrieve the dataset` (silent-retry exhaust) at step 1 of training loop | ~5 min |
| **v6** | 09:56 | restore FPS=60, ml=12288 + `max_pixels=100352` (halved) | OOM during backward, EXACT same 90.86 GB used as v4 | ~7 min |
| **v7** | 10:05 | restore max_pixels=200704, **`--deepspeed zero2_offload`** | `CUDAMismatchException`: nvcc 13.2 (system default) ≠ torch cu130 | 21 s |
| **v7b** | 10:06 | + `CUDA_HOME=/usr/local/cuda-13.0` | `Unable to JIT load cpu_adam: ninja not installed` (venv-bin not on `sudo -u ssm-user`'s PATH) | 25 s |
| **v7c** | 10:07 | + `PATH=venv/bin:cuda-13.0/bin:…` | **STARTED training**: step 1 done at 4:55 wall, loss=1.011, mem 85 GB. **But step time = 295 s/step** (vs yesterday's 15.65). Then crashed at step 2 with the same `Failed to retrieve dataset` error | ~12 min |
| **v8** | 10:25 | drop offload (slow + didn't fix dataset error). Switch to **`--deepspeed zero3`** + **`--strict true`** (re-raise actual exception instead of swallow) + restore FPS=60, max_pixels=200704, ml=12288 | TBD — designed to reveal the actual exception, not just retry-exhaust |

### Critical insight from v6 vs v4

Halving `max_pixels` (200704 → 100352) **did NOT reduce GPU memory** — v6 OOMed with the EXACT same 90.86 GB / 6.57 GB allocation attempt as v4. This proves activation memory in our setup is bound by `max_length` (pre-allocated buffer size), not actual sequence content. Reducing pixels reduces token count per row but doesn't shrink the allocated buffer.

This was an expensive thing to learn but it eliminates a whole family of "just reduce input fidelity" fixes — they don't address the real bottleneck.

### Critical insight from v7c

**zero2_offload (AdamW to CPU)** dropped model-load GPU usage 23 GB → 9.9 GB confirming offload works mechanically. v7c then ran step 1 successfully (loss 1.011, mem 85 GB) — proving the recipe trains. But two showstoppers emerged:

1. **295 s/step.** Projected 36 hours for full run. CPU offload tax was much heavier than estimated. The user warned me this would be aggressive and they were right.
2. **Step 2 also hit `Failed to retrieve dataset`** — same as v1, v5. So the dataset-retrieval failure is **independent of memory strategy**. It's a real bug we've been blaming on memory.

### The user's first-principles reset (10:24)

User observation: "we have edited a lot of configs and they all make sense; what is the real issue here?"

What I'd been missing: **I had been guessing token counts and memory needs** instead of measuring them. Each "fix" was a hypothesis without data. The audit recipe (FPS=60, max_pixels=200704, max_length=12288) is the right TARGET, but I never verified that our data actually fits inside it.

The real first-principles diagnostic: token-count profiling. Until we know what the actual per-row token counts are at the audit settings, we cannot tell:
- what max_length we need
- which rows are being silently dropped
- whether the `Failed to retrieve` error is genuine MaxLengthError or something else hidden by swift's catch-all retry

### Immediate next step: --strict true

Two ways to investigate the recurring "Failed to retrieve" error:

1. **`--strict true`** (cheap): tells swift's retry loop to re-raise the actual underlying exception instead of swallowing it. The FIRST encoding error becomes visible with full traceback. v8 launched at 10:25 with this flag.

2. **Custom profile script** (more work): encode all 818 ads offline, build histogram. Stalled on swift import path (`swift.llm` doesn't exist in this fork; need to use `swift.arguments.sft_args` etc.).

Strategy: let v8 reveal the error type. If it's MaxLengthError on specific rows, we know max_length is genuinely violated and the profile script becomes mandatory to right-size it. If it's some other exception (decord, audio, etc.), the profile script can target that specific failure mode.

## What ran in v8 (current state at 10:25)

| arg | value |
|---|---|
| model | Qwen2.5-Omni-3B |
| max_length | 12288 (audit ambitious) |
| FPS_MAX_FRAMES | 60 (audit ambitious) |
| max_pixels / VIDEO_MAX_PIXELS | 200704 (audit ambitious) |
| VIDEO_MAX_TOKEN_NUM | 8192 |
| attn_impl | flash_attn 2.8.3 (compiled today for sm_120) |
| lazy_tokenize | true |
| group_by_length | false |
| **strict** | **true** ← NEW: makes errors visible |
| **deepspeed** | **zero3** ← swap from zero2_offload (slower + didn't help) |
| Talker | disabled |
| val_dataset | 101 ads |

## Today's launches at a glance

```
v1   06:51   ❌ data retrieval (max_length=8192 vs FPS=60)
v2   07:29   ❌ flash-attn not installed
v3   08:45   ❌ stuck inside decord (50+ min, 0% CPU after entry)
v4   09:41   ❌ OOM backward at ml=12288 (90 GB / 95 GB)
v5   09:48   ❌ data retrieval at ml=8192 (yesterday's config + FA)
v6   09:56   ❌ OOM backward at ml=12288 (max_pixels halved didn't help)
v7   10:05   ❌ DeepSpeed CUDAMismatchException
v7b  10:06   ❌ ninja not on PATH
v7c  10:07   ✅ trained step 1 (loss=1.011!) but 295s/step, then data retrieval at step 2
v8   10:25   ⏳ --strict + zero3 to expose real exception
```

## Patterns that repeat

- **Bundling many config changes**: every launch changed 3-7 variables at once. Failure modes compound.
- **Guessing instead of measuring**: token counts, memory needs, decord behavior — all guessed, none measured.
- **Swallowing errors**: swift's retry loop hid 3 different failures behind the same misleading "increase max_length" message. Should have used `--strict true` from the start.
- **Auto-recovery scripts amplify mistakes**: the orchestrator's `kill -0` bug wasted 22 seconds catastrophically. Log-only watchdogs from then on.

## 10:22 — v8 result with --strict true: real exception revealed

```
swift.template.base.MaxLengthError: Current length of row(12407) is larger than the max_length(12288).
```

**The real failure mode, finally exposed:** one row encodes to 12,407 tokens at FPS=60 + max_pixels=200704 + VIDEO_MAX_TOKEN_NUM=8192. max_length=12288 is just 119 tokens too small for at least this row.

This single fact retroactively explains v1, v5, v7c — all hit MaxLengthError on some row that exceeded max_length, then the retry loop exhausted, and swift's catch-all wrapper produced the misleading "increase max_length" message. The message was technically correct but obscured the real signal (which row, by how much).

**Now we have ground truth** but only for ONE row. The distribution is unknown — could be 1 row at 12407, or 50 rows scattered from 12300 to 18000.

## 10:25 — token-count profile launched

`/tmp/profile_tokens_v2.py` running as ssm-user (pid 3025370). Encodes all 717 train + 101 val rows through the actual `Qwen2_5OmniTemplate.encode` path with FPS=60 / max_pixels=200704 / VIDEO_MAX_TOKEN_NUM=8192. Output: full histogram + drop counts at various candidate max_length values.

ETA: ~20-30 min for 818 multimodal rows. Outputs to `/tmp/token_profile.txt`.

## Final outcome — v38 completed (2h 6m wall, ~15:00 UTC)

After **38 launches** today, the working recipe was found and the overfit hypothesis was confirmed.

### Winning config

```bash
# Env vars
MAX_PIXELS=200704
VIDEO_MAX_PIXELS=200704
FPS_MAX_FRAMES=32                              # yesterday's, not the audit-aspirational 60
VIDEO_MAX_TOKEN_NUM=4096                       # yesterday's
ENABLE_AUDIO_OUTPUT=False                      # Talker disabled
CUDA_HOME=/usr/local/cuda-13.0                 # match torch cu130
PATH=/opt/dlami/nvme/work/swift_venv/bin:...   # ninja reachable

# CLI args
--max_length 8192
--truncation_strategy delete
--attn_impl sdpa                               # explicit SDPA (not auto-detect flash-attn)
--lazy_tokenize true
--strict false
--dataset_num_proc 1
--group_by_length false
--gradient_checkpointing true
--vit_gradient_checkpointing true              # KEY: freed 34 GB
--deepspeed zero2_offload                      # frees ~12 GB more
--dataloader_num_workers 1

# Dataset
train: 116-row filtered subset (originals dropped if estimated tokens > 5000)
val:   20-row filtered subset (same filter applied to val)
```

### Final training curve (v38)

| step | train_loss | train_acc | eval_loss | eval_acc | epoch |
|---|---|---|---|---|---|
| 1 | 0.99 | 0.70 | — | — | 0.14 |
| 5 | 0.91 | 0.69 | — | — | 0.69 |
| 10 | 0.58 | 0.78 | — | — | 1.28 |
| 20 | 0.54 | 0.79 | — | — | 2.55 |
| 30 | 0.40 | 0.85 | — | — | 3.83 |
| 40 | 0.29 | 0.90 | — | — | 5.00 |
| **50** | **0.18** | **0.94** | **1.21** | **0.76** | **6.28** |
| 60 | 0.087 | 0.97 | — | — | 7.55 |
| 70 | 0.047 | 0.98 | — | — | 8.83 |
| **80** | **0.036** | **0.99** | **1.82** | **0.76** | **10.0** |

### Capacity test verdict

**PASSED.** Train loss reaches 0.036 (99% accuracy) on 116 ads, proving the architecture has memorization capacity. Val loss INCREASES from step 50→80 (1.21 → 1.82) while train still decreases — textbook overfit. Best val checkpoint = **step 50** (eval_loss=1.21).

For future work: the recipe + 116-row training set demonstrates capacity. With the full 717-row dataset (once we solve the >8192-token rows issue without filtering), val loss should improve. The audit's FPS=60 + ml=12K aspiration is currently blocked by hardware (2× 95 GB Blackwell is insufficient at full FT for the model's vocab=152064 × seqlen=13K logits float32 cast).

### Total session stats

- 38 distinct launches (v1 — v38, plus some retries)
- 4 successful checkpoints (only v37 + v38 produced any)
- 1 successful complete run (v38)
- Wall time spent: ~9 hours from first launch to final eval
- flash-attn 2.8.3 compiled successfully (and ultimately not used — explicit SDPA proved more memory-efficient for our model+hardware combo)
- liger-kernel installed (didn't apply to Qwen2.5-Omni's multimodal forward)

## Open: how to do this right next time

1. **Profile data once (token counts) before tuning configs** — should have been step zero, not step seven
2. **Change one variable per launch** — bundling 7 changes compounds 7 failure modes
3. **Use `--strict true` from day one** — swift's misleading silent-retry error message wasted hours
4. **Set `--load_from_cache_file true`** — so preprocessing pays once per dataset
5. **Distinguish encoding errors from training errors early** — MaxLengthError (data-side) vs OOM (memory-side) are different problems with different fixes; I conflated them
6. **vit_gradient_checkpointing is HUGE for multimodal full-FT** — freed 34 GB in v28 (memory dropped 94→60 GB), the key fix that finally allowed training
7. **`truncation_strategy=left` corrupts vision token alignment** — model diverges. Use `delete` (drop rows) instead, even at cost of dataset size
8. **`n_try_fetch=10` (swift default) is the silent failure point** — it's the retry budget for MaxLengthError; one stuck batch of 10 long rows kills training. Either filter the dataset offline OR bump `n_try_fetch`
9. **Always filter both train AND val datasets the same way** — v37 trained successfully but eval crashed because val was unfiltered
2. **Step time at FPS=60** — yesterday at FPS=32 was 15.65 s/it. Today's FPS=60 doubles per-row video decode; flash-attn partly offsets. Expected 18-26 s/it; will know once wandb starts logging.
3. **Long-bucket OOM risk** — with `group_by_length=true` and `max_length=12288`, the longest-seqlen bucket may push past yesterday's 78/80 GB peak. Mitigation if it crashes: fall back to `max_length=10240`.
4. **`--cached_dataset` worth setting up?** — if the upfront map is the dominant startup cost, caching could amortize across future runs.

## Lessons (also worth saving to memory)

1. **Verify environment facts before encoding them as assumptions.** I used `sudo -u ssm-user` because earlier outputs were under that user — never `ls -ld` the venv to check ownership. Was `ubuntu`-owned. 30-sec check skipped, cost ~12 min.
2. **The codebase often has better calibrations than training-data heuristics.** flash-attn's `setup.py:506` has the MAX_JOBS formula. I picked 8 from memory; the source said 11. Read the actual code.
3. **Auto-recovery scripts need extreme care.** My `kill -0` permission bug burned 3 attempts in 1 second. Default to *no* auto-recovery if you're not sure the recovery path is bulletproof. Log-only watchdogs (no auto-relaunch) are safer.
4. **Process state ≠ log content.** `sft.log` was the wrong signal (it had ghost errors); rank state in `/proc` was the right signal. Filter monitors for both.
5. **Bundle one variable per launch.** v1 + v2 + v3 each introduced multiple changes; debugging the first failure required disentangling which interaction caused it. For the next round, change one thing per launch.
6. **`ninja's [N/M]` progress markers** are the right monitor filter when a build uses ninja — they distinguish actual parallel progress from serial slog much more reliably than log keywords.
7. **`sudo -u` is not free.** It rebuilds the entire env from the target user's profile and can break env-var inheritance to child processes (ninja in our case). Drop it whenever the target user owns the destination.
