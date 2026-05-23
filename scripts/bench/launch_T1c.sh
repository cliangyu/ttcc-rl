#!/usr/bin/env bash
# T1c: ZeRO-2 (no offload) instead of ZeRO-3. Liger off (T1b was a wash).
# Risk: +4.7 GiB peak (un-sharded params); B0 was 86 GiB → ~91 GiB. ~4 GiB headroom.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo -u ssm-user -E HOME=/home/ssm-user \
    PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin \
    LABEL=T1c LIGER=0 ZERO=zero2 VIT_CKPT=true BS=1 GA=8 \
    bash "${SCRIPT_DIR}/bench_sft.sh" > /home/ubuntu/ttcc-rl/runs/config_sweep/bench_T1c.log 2>&1
