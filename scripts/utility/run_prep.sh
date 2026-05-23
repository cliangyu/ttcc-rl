#!/bin/bash
set -uo pipefail
source /opt/dlami/nvme/work/swift_venv/bin/activate
export HF_HOME=/home/ssm-user/work/hf-cache
cd /home/ssm-user/work
mkdir -p /home/ssm-user/work/data/ttcc_swift_v2cot
exec python /home/ubuntu/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/prepare_dataset.py \
  --cot-jsonl /home/ssm-user/work/work-out/cot/v2_full_instruct_merged.jsonl \
  --out-dir /home/ssm-user/work/data/ttcc_swift_v2cot \
  > /home/ssm-user/work/data/ttcc_swift_v2cot/prep.log 2>&1
