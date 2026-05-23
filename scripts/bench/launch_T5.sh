#!/usr/bin/env bash
# T5: T4b winning config + the full dataloader fix.
#   num_workers=4         (2 per GPU; conservative start within "4-8 per GPU" recommendation)
#   persistent_workers=true (amortize spawn cost across epochs)
#   prefetch_factor=4     (research-backed default; ~32 batches in flight = 4 workers × 4 prefetch × bs 2)
# Predicted: step time -15-25% vs T4b if data loading was the hidden bottleneck.
# CPU RAM cost: ~1.6 GiB; negligible at 499 GiB available.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo -u ssm-user -E HOME=/home/ssm-user \
    PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin \
    LABEL=T5 LIGER=0 ZERO=zero3 VIT_CKPT=true BS=2 GA=4 COMPILE=1 GROUP_BY_LEN=false \
    NUM_WORKERS=4 PERSIST_WORKERS=true PREFETCH=4 \
    bash "${SCRIPT_DIR}/bench_sft.sh" > /home/ubuntu/ttcc-rl/runs/config_sweep/bench_T5.log 2>&1
