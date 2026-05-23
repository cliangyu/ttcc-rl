#!/usr/bin/env bash
# T6a: dataloader num_workers sweep. Tests {2, 4, 8, 12} workers.
# All other knobs match T5 (the winning config so far).
# Goal: find the inflection point where step time stops decreasing.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/ubuntu/ttcc-rl/runs/config_sweep"

run_one() {
    local nw="$1"
    local label="T6a_w${nw}"
    echo "=== T6a sweep launching ${label} $(date) ==="
    sudo -u ssm-user -E HOME=/home/ssm-user \
        PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin \
        LABEL="${label}" LIGER=0 ZERO=zero3 VIT_CKPT=true BS=2 GA=4 COMPILE=1 GROUP_BY_LEN=false \
        NUM_WORKERS="${nw}" PERSIST_WORKERS=true PREFETCH=4 \
        bash "${SCRIPT_DIR}/bench_sft.sh" > "${LOG_DIR}/bench_${label}.log" 2>&1
    local rc=$?
    echo "=== ${label} done (exit ${rc}) $(date) ==="
    # Drain
    while pgrep -af "swift\.cli\.main sft" > /dev/null; do
        sleep 10
    done
    sleep 5
}

run_one 2
run_one 4
run_one 8
run_one 12

echo "=== T6a sweep complete $(date) ==="
