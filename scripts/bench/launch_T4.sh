#!/usr/bin/env bash
# T4: T3a winning config + bs=2 + ga=4 (effective batch unchanged) + group_by_length=true.
# Memory budget: T3a was 53.77 GiB; bs=2 expected +30 GiB → ~84 GiB peak, ~11 GiB headroom.
# group_by_length=true reduces shape diversity so dynamo can stabilize cache → less recompile churn.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo -u ssm-user -E HOME=/home/ssm-user \
    PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin \
    LABEL=T4 LIGER=0 ZERO=zero3 VIT_CKPT=true BS=2 GA=4 COMPILE=1 GROUP_BY_LEN=true \
    bash "${SCRIPT_DIR}/bench_sft.sh" > /home/ubuntu/ttcc-rl/runs/config_sweep/bench_T4.log 2>&1
