#!/bin/bash
# Usage: run_full.sh <gpu_idx> <start_idx>
set -uo pipefail
GPU=$1
START=$2
source /opt/dlami/nvme/work/venv/bin/activate
export HF_HOME=/home/ssm-user/work/hf-cache
cd /home/ssm-user/work
mkdir -p /home/ssm-user/work/work-out/cot
exec python scripts/cot_distill_v2.py \
  --model INSTRUCT \
  --full \
  --start-idx "${START}" \
  --stride 2 \
  --gpu "${GPU}" \
  --out "/home/ssm-user/work/work-out/cot/v2_full_gpu${GPU}.jsonl" \
  > "/home/ssm-user/work/work-out/cot/v2_full_gpu${GPU}.log" 2>&1
