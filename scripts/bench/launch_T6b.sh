#!/usr/bin/env bash
# T6b: T5 winning config + DeepSpeed overlap_comm=true.
# Uses custom zero3_overlap.json (overlap_comm: true) instead of swift's stock zero3.json.
# Hypothesis: hide all-gather/reduce-scatter behind backward. Predicted +5-10% step time.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo -u ssm-user -E HOME=/home/ssm-user \
    PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin \
    LABEL=T6b LIGER=0 ZERO="${SCRIPT_DIR}/zero3_overlap.json" VIT_CKPT=true BS=2 GA=4 COMPILE=1 GROUP_BY_LEN=false \
    NUM_WORKERS=4 PERSIST_WORKERS=true PREFETCH=4 \
    bash "${SCRIPT_DIR}/bench_sft.sh" > /home/ubuntu/ttcc-rl/runs/config_sweep/bench_T6b.log 2>&1
