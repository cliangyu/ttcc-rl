#!/bin/bash
source /home/ssm-user/work/venv/bin/activate
cd /home/ssm-user/work
python /home/ssm-user/work/scripts/qc_cot.py \
  --in /home/ssm-user/work/work-out/cot/v3_pro_merged_clean.jsonl \
  --report /home/ssm-user/work/work-out/cot/v3_pro_qc_report_clean.json
