#!/bin/bash
# Simple post-training watchdog. NO auto-relaunch (orchestrator v1 had a kill -0 perm bug
# that misclassified live procs as dead and burned all 3 attempts).
# Strategy: poll sft.log for completion or crash markers. On success, run val inference.

set -u
SFT_LOG=/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full/sft.log
SUMMARY=/tmp/overnight_summary.md
WATCHDOG_LOG=/tmp/watchdog.log
OUT_DIR=/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCHDOG_LOG"; }

write_summary() {
    local state="$1"
    {
        echo "# Overnight TTCC SFT-noCoT v3 (corrected recipe + flash-attn) — overfit test"
        echo
        echo "Last updated: $(date '+%F %T')"
        echo
        echo "## Current state"
        echo "$state"
        echo
        echo "## What ran"
        echo "- Recipe: \`max_length=12288\`, FPS_MAX_FRAMES=60, flash-attn 2.8.3 (compiled today for sm_120/cu130/torch2.11)"
        echo "- Train set: 717 ads (real train split, v2cot_nocot variant)"
        echo "- Val set: 101 ads (real val split — first time we're tracking val loss)"
        echo "- 10 epochs, full FT of LLM only (ViT + audio aligner frozen, Talker disabled)"
        echo "- wandb run: \`sft_nocot_full_overfit_20260522_v3\` in project \`ttcc\` (liangyuch)"
        echo
        echo "## What went wrong overnight (full disclosure)"
        echo "First overnight orchestrator had a permission bug — \`kill -0\` against ssm-user-owned"
        echo "training pid from ubuntu user returned EPERM, which the \`until ! kill -0\` loop"
        echo "misread as 'process died'. Orchestrator falsely declared 3 'failed' attempts in 1 second"
        echo "and gave up. **Importantly:**"
        echo "- The actual v3 training process kept running (its argv was set at launch and unaffected)"
        echo "- The orchestrator's sed-modifications to the script were reverted"
        echo "- A simpler log-only watchdog now drives post-training (no auto-relaunch)"
        echo
        echo "## How to verify the right thing ran"
        echo "Inspect the live training process argv (it should still show v3 config):"
        echo "\`\`\`"
        echo "ps -p 2911225 -o cmd"
        echo "\`\`\`"
        echo "Expect: \`--max_length 12288 --attn_impl flash_attn --lazy_tokenize false ...\`"
        echo
        echo "## Watchdog log"
        echo "\`\`\`"
        tail -30 "$WATCHDOG_LOG" 2>/dev/null
        echo "\`\`\`"
        echo
        echo "## Last 50 lines of sft.log"
        echo "\`\`\`"
        sudo tail -50 "$SFT_LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*[A-Za-z]//g'
        echo "\`\`\`"
    } > "$SUMMARY"
}

OUTCOME="unknown"
log "watchdog started. polling sft.log every 60s for completion or crash markers."
write_summary "watchdog active — training in progress"

while true; do
    # success: train_runtime appears at the very end of training
    if sudo grep -qE "train_runtime" "$SFT_LOG" 2>/dev/null; then
        log "TRAINING SUCCEEDED — train_runtime line detected"
        OUTCOME=success
        break
    fi
    # failure: ChildFailedError or sft.py FAILED
    if sudo grep -qE "ChildFailedError|cli/sft\.py FAILED|core dumped|exitcode: -[0-9]" "$SFT_LOG" 2>/dev/null; then
        log "training crashed — ChildFailedError or FAILED detected"
        OUTCOME=crash
        break
    fi
    # OOM specifically (often visible before the wrapper-level FAILED)
    if sudo grep -qE "CUDA out of memory|OutOfMemoryError" "$SFT_LOG" 2>/dev/null; then
        log "OOM detected"
        OUTCOME=oom
        # don't break yet — wait for the FAILED line to be sure
    fi

    # heartbeat summary every 5 polls (5 min)
    write_summary "running — last poll $(date '+%F %T'), no terminal markers yet"
    sleep 60
done

write_summary "training finished, outcome=$OUTCOME, running post-training steps if applicable"

if [ "$OUTCOME" = "success" ]; then
    LAST_CKPT=$(sudo find "$OUT_DIR" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    log "last checkpoint: $LAST_CKPT"
    if [ -n "$LAST_CKPT" ] && sudo test -d "$LAST_CKPT"; then
        log "running val inference on $LAST_CKPT"
        cd /home/ubuntu/ttcc-rl
        sudo bash scripts/infer_trained.sh "$LAST_CKPT" sft_nocot_v3 \
            /tmp/preds_sft_nocot_v3_val.parquet \
            /home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_val.jsonl \
            > /tmp/val_inference.log 2>&1
        VAL_RC=$?
        log "val inference exit code: $VAL_RC"
        if [ $VAL_RC -eq 0 ] && [ -f /tmp/preds_sft_nocot_v3_val.parquet ]; then
            log "val predictions written: /tmp/preds_sft_nocot_v3_val.parquet"
        else
            log "val inference issue — check /tmp/val_inference.log"
        fi
    else
        log "no checkpoint found at $OUT_DIR — cannot run val inference"
    fi
fi

write_summary "DONE: outcome=$OUTCOME. See /tmp/watchdog.log and sft.log."
log "watchdog done"
