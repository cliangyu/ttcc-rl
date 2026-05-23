#!/bin/bash
# Overnight orchestrator: poll training, recover from OOM/crash, do post-training inference.
# Triggered by fa_chain.sh after the SFT-noCoT v3 launch.
# Logs to /tmp/orchestrator.log; final summary in /tmp/overnight_summary.md.
#
# Pre-planned recovery strategy (ordered):
#   attempt 1 = the v3 launch (max_length=12288, FA on, FPS_MAX_FRAMES=60)
#   attempt 2 = max_length 12288 -> 10240 (relieves long-bucket OOM)
#   attempt 3 = fall back to yesterday's known-working: FPS_MAX_FRAMES=32, max_length=8192, no FA
#   give up after 3.

set -u
ORCH_LOG=/tmp/orchestrator.log
SUMMARY=/tmp/overnight_summary.md
SFT_LOG=/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full/sft.log
OUT_DIR=/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full
SCRIPT_DIR=/home/ubuntu/go_viral/examples/train/grpo/qwen2_5_omni_ttcc
TRAIN_SCRIPT="${SCRIPT_DIR}/sft_v2cot_full.sh"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ORCH_LOG"; }

write_summary() {
    {
        echo "# Overnight training run — TTCC SFT-noCoT full-FT overfit test"
        echo
        echo "Last updated: $(date '+%F %T')"
        echo
        cat <<EOF
## Goal
Test whether full-FT of Qwen2.5-Omni-3B can overfit on 717 train ads. Recipe: corrected per audit (max_length=12288, FPS_MAX_FRAMES=60, group_by_length, Talker disabled). Validation: 101 ads from real val split.

## Run state
$1

## What to read first
1. wandb runs in project \`ttcc\` (user liangyuch); look for runs named \`sft_nocot_full_overfit_20260522_*\`
2. SFT log: \`$SFT_LOG\` (always the latest attempt)
3. Orchestrator log: \`$ORCH_LOG\` (chronological per-attempt notes)
4. ms-swift output dirs: \`$OUT_DIR/v*/checkpoint-*\`

## Attempts log
EOF
        if [ -f /tmp/attempts.log ]; then cat /tmp/attempts.log; fi
        echo
        echo "## Last 40 lines of sft.log"
        echo '```'
        tail -40 "$SFT_LOG" 2>/dev/null
        echo '```'
    } > "$SUMMARY"
}

find_train_pid() {
    # the training python -m swift.cli.main sft (under torch.distributed.run)
    pgrep -f "swift.cli.main sft.*ttcc" | head -1
}

wait_for_train_to_exit() {
    local pid
    pid=$(find_train_pid)
    if [ -z "$pid" ]; then
        log "no training pid found yet, polling..."
        local tries=0
        while [ -z "$pid" ] && [ $tries -lt 60 ]; do
            sleep 30
            pid=$(find_train_pid)
            tries=$((tries + 1))
        done
    fi
    if [ -z "$pid" ]; then
        log "ERROR: training never started after 30 min wait. abort."
        return 1
    fi
    log "watching training pid $pid"
    until ! kill -0 "$pid" 2>/dev/null; do sleep 60; done
    log "training pid $pid exited"
    return 0
}

classify_failure() {
    # Returns: "success" | "oom" | "fa_init" | "data" | "other"
    if grep -qE "train_runtime.*epoch" "$SFT_LOG" 2>/dev/null; then
        echo "success"; return
    fi
    if grep -qE "CUDA out of memory|OutOfMemoryError|cublas.*alloc" "$SFT_LOG" 2>/dev/null; then
        echo "oom"; return
    fi
    if grep -qE "flash_attn|FlashAttention.*ImportError|no kernel image" "$SFT_LOG" 2>/dev/null; then
        echo "fa_init"; return
    fi
    if grep -qE "Failed to retrieve the dataset|FileNotFoundError|Permission denied.*data" "$SFT_LOG" 2>/dev/null; then
        echo "data"; return
    fi
    echo "other"
}

relaunch() {
    local wandb_name="$1"
    log "launching training run: $wandb_name"
    sudo mv "$SFT_LOG" "${SFT_LOG}.prev.$(date +%s)" 2>/dev/null || true
    cd "$SCRIPT_DIR"
    sudo -u ssm-user -E env \
        HOME=/home/ssm-user \
        WANDB_PROJECT=ttcc \
        WANDB_NAME="$wandb_name" \
        bash ./sft_nocot_v2cot_full.sh > "/tmp/sft_${wandb_name}.launcher.log" 2>&1 &
    local pid=$!
    disown "$pid"
    log "relaunched, pid $pid (wandb name: $wandb_name)"
    return 0
}

# --- main loop ---
write_summary "starting — waiting for v3 launch to actually start training"

ATTEMPT=1
TRAINING_OK=0
echo "Attempt 1: v3 default recipe (max_length=12288, FA, FPS=60)" >> /tmp/attempts.log

while [ "$TRAINING_OK" = "0" ] && [ "$ATTEMPT" -le 3 ]; do
    log "Attempt $ATTEMPT: waiting for training to exit"
    if ! wait_for_train_to_exit; then
        write_summary "abort: training never started for attempt $ATTEMPT"
        break
    fi
    CLASS=$(classify_failure)
    log "Attempt $ATTEMPT outcome: $CLASS"
    echo "  outcome: $CLASS" >> /tmp/attempts.log

    case "$CLASS" in
        success)
            TRAINING_OK=1
            log "training reported success"
            ;;
        oom)
            if [ "$ATTEMPT" -lt 3 ]; then
                ATTEMPT=$((ATTEMPT + 1))
                log "OOM detected. Retrying with max_length=10240"
                echo "Attempt $ATTEMPT: max_length=12288 -> 10240 (OOM recovery)" >> /tmp/attempts.log
                sed -i 's/--max_length 12288/--max_length 10240/' "$TRAIN_SCRIPT"
                relaunch "sft_nocot_full_overfit_20260522_v4_oom_recovery"
            else
                log "OOM after attempt 3, giving up"
                break
            fi
            ;;
        fa_init|other)
            if [ "$ATTEMPT" -lt 3 ]; then
                ATTEMPT=$((ATTEMPT + 1))
                log "Falling back to yesterday's known-working config (FPS=32, max_length=8192, no FA)"
                echo "Attempt $ATTEMPT: full fallback to known-working (FPS=32, ml=8192, no FA)" >> /tmp/attempts.log
                # restore yesterday's config
                sed -i 's/--max_length 12288/--max_length 8192/; s/--max_length 10240/--max_length 8192/' "$TRAIN_SCRIPT"
                sed -i 's/--attn_impl flash_attn \\/--attn_impl sdpa \\/' "$TRAIN_SCRIPT"
                sed -i 's/FPS_MAX_FRAMES=60/FPS_MAX_FRAMES=32/' "$TRAIN_SCRIPT"
                sed -i 's/VIDEO_MAX_TOKEN_NUM=8192/VIDEO_MAX_TOKEN_NUM=4096/' "$TRAIN_SCRIPT"
                relaunch "sft_nocot_full_overfit_20260522_v5_fallback"
            else
                log "non-recoverable error after attempt 3, giving up"
                break
            fi
            ;;
        data)
            log "data-side error — not auto-recoverable, aborting"
            break
            ;;
    esac
    write_summary "running attempt $ATTEMPT (last outcome: $CLASS)"
done

if [ "$TRAINING_OK" = "1" ]; then
    log "training succeeded; running val inference on the last checkpoint"
    LAST_CKPT=$(find "$OUT_DIR" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    log "last ckpt: $LAST_CKPT"
    if [ -n "$LAST_CKPT" ] && [ -d "$LAST_CKPT" ]; then
        cd /home/ubuntu/ttcc-rl
        sudo bash scripts/infer_trained.sh "$LAST_CKPT" sft_nocot_v3 \
            /tmp/preds_sft_nocot_v3_val.parquet \
            /home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_val.jsonl \
            > /tmp/val_inference.log 2>&1
        log "val inference done; predictions at /tmp/preds_sft_nocot_v3_val.parquet"
    else
        log "no checkpoint found, skipping inference"
    fi
fi

write_summary "$([ "$TRAINING_OK" = "1" ] && echo "TRAINING SUCCEEDED on attempt $ATTEMPT" || echo "TRAINING FAILED after attempt $ATTEMPT — see attempts log")"
log "orchestrator done"
