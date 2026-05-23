#!/bin/bash
set -uo pipefail
source /opt/dlami/nvme/work/venv/bin/activate
export HF_HOME=/home/ssm-user/work/hf-cache
cd /home/ssm-user/work
mkdir -p /home/ssm-user/work/work-out/cot
exec python scripts/cot_distill_v2.py \
  --model INSTRUCT \
  --pilot 5 \
  --gpu 0 \
  --out /home/ssm-user/work/work-out/cot/v2_pilot.jsonl \
  > /home/ssm-user/work/work-out/cot/v2_pilot.log 2>&1
