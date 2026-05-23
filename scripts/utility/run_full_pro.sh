#!/bin/bash
set -e
source /home/ssm-user/work/venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=/home/ssm-user/work/vizzy-sa.json
cd /home/ssm-user/work
python /home/ssm-user/work/scripts/cot_distill_v3_gemini.py \
  --model pro --full --concurrency 32 \
  --out /home/ssm-user/work/work-out/cot/v3_pro_full.jsonl \
  >/home/ssm-user/work/work-out/cot/v3_pro_full.log 2>&1
