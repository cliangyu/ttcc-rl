#!/bin/bash
# v3 watchdog: watch the actual training process and checkpoint emergence.
# Does NOT depend on sft.log (which was orphaned by orchestrator's mv during ghost relaunches).
# Triggers: (a) checkpoint-450 appears (training complete) OR (b) v3 pid dies
# Then: run val inference on the latest checkpoint, write summary.

set -u
V3_PID=3060994           # v9 — audit recipe + max_length=14336 + zero3 + strict=false
SUMMARY=/tmp/overnight_summary.md
WATCHDOG_LOG=/tmp/v3_watchdog.log
OUT_DIR=/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full
V3_SUBDIR=/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full/v11-20260522-103529  # v9's actual subdir

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCHDOG_LOG"; }

# alive check using `sudo kill -0` — avoids the EPERM bug that killed orchestrator v1
is_alive() { sudo kill -0 "$1" 2>/dev/null; }

last_checkpoint() {
    # ONLY look in v3's specific subdir — not the whole OUT_DIR (yesterday's v0 has ckpt-450)
    sudo find "$V3_SUBDIR" -maxdepth 2 -name "checkpoint-*" -type d 2>/dev/null \
        | sed 's/.*checkpoint-//' | sort -n | tail -1
}

write_summary() {
    local state="$1"
    local latest_ckpt="$2"
    {
        echo "# Overnight TTCC SFT-noCoT v5 — overfit test (3rd retry; v3 stuck, v4 OOMed)"
        echo
        echo "Last updated: $(date '+%F %T')"
        echo
        echo "## TL;DR"
        echo "$state"
        echo
        echo "## Recipe (v5, actually running now — yesterday's working config + audit's safe additions)"
        echo "- **max_length=8192, FPS_MAX_FRAMES=32, VIDEO_MAX_TOKEN_NUM=4096** (yesterday's, known to fit at 78/80 GB)"
        echo "- attn_impl=flash_attn 2.8.3 (compiled today for sm_120/cu130/torch2.11)"
        echo "- lazy_tokenize=true, group_by_length=false, dataset_num_proc=1 (yesterday's)"
        echo "- ENABLE_AUDIO_OUTPUT=False (Talker disabled, ~1.5 GB GPU saved)"
        echo "- 717 train ads, **101 val ads** (first run with real val tracking), 10 epochs full FT"
        echo "- wandb run: \`sft_nocot_full_overfit_20260522_v5_safe\` in project \`ttcc\` (liangyuch)"
        echo "- Output dir: \`$V3_SUBDIR\` (v5)"
        echo
        echo "## Status of v4 training process"
        echo "- pid $V3_PID (top-level python): $(is_alive $V3_PID && echo ALIVE || echo DEAD)"
        echo "- last checkpoint on disk: ${latest_ckpt:-none}"
        echo
        echo "## Background incidents overnight (full disclosure)"
        echo
        echo "**Incident 1 (08:45) — auto-recovery orchestrator misfire.**"
        echo "Buggy \`kill -0\` cross-user check burned 3 attempts in 1 sec, sed-edited script to"
        echo "fallback. v3 itself unaffected. Reverted script, killed orchestrator."
        echo
        echo "**Incident 2 (08:45-09:40) — v3 stuck inside decord at FPS=60.**"
        echo "All 8 dataset_num_proc workers entered \`_read_video_decord\` and never returned."
        echo "0% CPU, 0 MB/s disk for 50+ min. Killed v3."
        echo
        echo "**Incident 3 (09:41-09:47) — v4 OOMed during backward pass at max_length=12288.**"
        echo "torch.OutOfMemoryError, GPU 1 used 90 GB / 95 GB, tried to allocate 6.57 GB more."
        echo "Flash-attn saved attention memory but FFN activations + grads at seqlen 10-12K still"
        echo "exceeded budget. Reverted to yesterday's max_length=8192 + FPS=32 (known to fit at"
        echo "78/80 GB). Launched v5."
        echo
        echo "See \`/home/ubuntu/ttcc-rl/docs/08_session_journal_20260522.md\` for full forensics."
        echo
        echo "## What to verify when you wake up"
        echo "1. wandb run \`sft_nocot_full_overfit_20260522_v3\` should show training loss curves"
        echo "2. Checkpoints should be in \`$OUT_DIR/v?-…/checkpoint-{50,100,150,…,450}\`"
        echo "3. If val inference completed, predictions at \`/tmp/preds_sft_nocot_v3_val.parquet\`"
        echo
        echo "## Watchdog log"
        echo '```'
        tail -25 "$WATCHDOG_LOG" 2>/dev/null
        echo '```'
        echo
        echo "## Misleading files (do not read for v3 truth)"
        echo "- \`$OUT_DIR/sft.log.ghost_errors\` — ghost relaunch EADDRINUSE traceback"
        echo "- \`$OUT_DIR/sft.log.v3_orphan_inode\` — also ghost errors (rotated mid-stream)"
        echo "- v3's actual log went to an orphan inode (no name in dir); wandb has the truth"
    } > "$SUMMARY"
}

log "v3 watchdog started. monitoring pid $V3_PID + checkpoint emergence in $OUT_DIR"
write_summary "watchdog active — v3 training in progress; checking every 2 min" ""

POLL_COUNT=0
while is_alive "$V3_PID"; do
    POLL_COUNT=$((POLL_COUNT + 1))
    if [ $((POLL_COUNT % 5)) -eq 0 ]; then
        LATEST=$(last_checkpoint)
        log "still alive after $((POLL_COUNT * 2)) min; latest ckpt step: ${LATEST:-none}"
        write_summary "running — last poll $(date '+%F %T'), pid alive, latest ckpt=${LATEST:-none}" "$LATEST"
    fi
    sleep 120
done

log "v3 pid $V3_PID exited at $(date '+%F %T')"
LATEST=$(last_checkpoint)
log "latest checkpoint step: ${LATEST:-none}"

# Determine outcome from checkpoint presence
if [ -n "$LATEST" ] && [ "$LATEST" -ge 450 ]; then
    OUTCOME="success — reached step $LATEST (>= 450 = end of 10 epochs)"
elif [ -n "$LATEST" ] && [ "$LATEST" -ge 50 ]; then
    OUTCOME="partial — got to step $LATEST but did not complete all 450 steps"
else
    OUTCOME="failed — no checkpoint reached (training crashed before step 50)"
fi
log "outcome: $OUTCOME"

# Run val inference on latest checkpoint if we have one
if [ -n "$LATEST" ]; then
    CKPT_PATH=$(sudo find "$V3_SUBDIR" -maxdepth 2 -name "checkpoint-${LATEST}" -type d 2>/dev/null | head -1)
    if [ -n "$CKPT_PATH" ]; then
        log "running val inference on $CKPT_PATH"
        cd /home/ubuntu/ttcc-rl
        sudo bash scripts/infer_trained.sh "$CKPT_PATH" sft_nocot_v3_ckpt${LATEST} \
            /tmp/preds_sft_nocot_v3_val.parquet \
            /home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_val.jsonl \
            > /tmp/val_inference.log 2>&1
        VAL_RC=$?
        log "val inference exit code: $VAL_RC"
        if [ $VAL_RC -eq 0 ]; then
            log "val predictions at /tmp/preds_sft_nocot_v3_val.parquet"
        fi
    fi
fi

write_summary "DONE: $OUTCOME" "$LATEST"
log "watchdog complete"
