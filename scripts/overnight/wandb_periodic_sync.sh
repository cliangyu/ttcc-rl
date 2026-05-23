#!/bin/bash
# Re-sync GRPO + RLOO TB events to wandb every 5 min while the orchestrator runs.
# Exits when chain_v2cot_full prints ALL 4 EXPERIMENTS DONE.
set -uo pipefail
export WANDB_ENTITY=liangyuch
export WANDB_PROJECT=ttcc
LOG=/home/ssm-user/work/work-out/chain_v2cot_full.log
SYNC=/opt/dlami/nvme/work/swift_venv/bin/wandb

while true; do
    for d in \
        /opt/dlami/nvme/ssm-out/ttcc_grpo_v2cot_full/v0-*/runs/* \
        /opt/dlami/nvme/ssm-out/ttcc_rloo_v2cot_full/v0-*/runs/*; do
        [ -d "$d" ] || continue
        "${SYNC}" sync --entity liangyuch --project ttcc "$d" >/dev/null 2>&1 || true
    done
    # Exit when chain done or fatal
    if grep -qE "ALL 4 EXPERIMENTS DONE|FATAL:" "${LOG}" 2>/dev/null; then
        START=$(grep -n "=== RESUME2 chain" "${LOG}" | tail -1 | cut -d: -f1)
        if tail -n +"${START}" "${LOG}" | grep -qE "ALL 4 EXPERIMENTS DONE|FATAL:"; then
            echo "[$(date '+%F %T')] chain finished -- final sync + exit"
            break
        fi
    fi
    sleep 300
done
