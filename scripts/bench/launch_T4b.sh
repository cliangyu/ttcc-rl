#!/usr/bin/env bash
# T4b: T3a + bs=2 + ga=4 ONLY (group_by_length dropped — incompatible with lazy_tokenize=true).
# Memory budget: T3a was 53.77 GiB; bs=2 expected +30 GiB → ~84 GiB peak.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo -u ssm-user -E HOME=/home/ssm-user \
    PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin \
    LABEL=T4b LIGER=0 ZERO=zero3 VIT_CKPT=true BS=2 GA=4 COMPILE=1 GROUP_BY_LEN=false \
    bash "${SCRIPT_DIR}/bench_sft.sh" > /home/ubuntu/ttcc-rl/runs/config_sweep/bench_T4b.log 2>&1
