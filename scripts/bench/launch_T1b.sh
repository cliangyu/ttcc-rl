#!/usr/bin/env bash
# T1b config-sweep launch: same as B0 baseline but with --use_liger_kernel true.
# See docs/09_session_journal_20260522_afternoon.md for the sweep design.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo -u ssm-user -E HOME=/home/ssm-user \
    PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin \
    LABEL=T1b LIGER=1 ZERO=zero3 VIT_CKPT=true BS=1 GA=8 \
    bash "${SCRIPT_DIR}/bench_sft.sh" > /home/ubuntu/ttcc-rl/runs/config_sweep/bench_T1b.log 2>&1
