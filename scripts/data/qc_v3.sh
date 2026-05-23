#!/bin/bash
set -e
source /home/ssm-user/work/venv/bin/activate
cd /home/ssm-user/work
mkdir -p /home/ssm-user/work/work-out/cot
python /home/ssm-user/work/scripts/qc_cot.py \
  --in /home/ssm-user/work/work-out/cot/v3_pro_merged.jsonl \
  --report /home/ssm-user/work/work-out/cot/v3_pro_qc_report.json
echo "exit=$?"
