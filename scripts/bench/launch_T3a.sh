#!/usr/bin/env bash
# T3a: torch.compile additive on top of baseline (or whichever config T1c proved best).
# Risk: compilation time can be 5-10 min on first step + dynamic shape recompiles.
# Falls back gracefully if compile fails (swift's _init_liger pattern catches Exception).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo -u ssm-user -E HOME=/home/ssm-user \
    PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin \
    LABEL=T3a LIGER=0 ZERO=zero3 VIT_CKPT=true BS=1 GA=8 COMPILE=1 \
    bash "${SCRIPT_DIR}/bench_sft.sh" > /home/ubuntu/ttcc-rl/runs/config_sweep/bench_T3a.log 2>&1
