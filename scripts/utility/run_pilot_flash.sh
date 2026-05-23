#!/bin/bash
set -e
mkdir -p /home/ssm-user/work/work-out/cot
source /home/ssm-user/work/venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=/home/ssm-user/work/vizzy-sa.json
cd /home/ssm-user/work
python /home/ssm-user/work/scripts/cot_distill_v3_gemini.py \
  --model flash --pilot 5 --concurrency 5 \
  --out /home/ssm-user/work/work-out/cot/v3_flash_pilot.jsonl \
  >/home/ssm-user/work/work-out/cot/v3_flash_pilot.log 2>&1
