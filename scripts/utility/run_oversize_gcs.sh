#!/bin/bash
set -e
source /home/ssm-user/work/venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=/home/ssm-user/work/vizzy-sa.json
cd /home/ssm-user/work
python /tmp/cot_distill_oversize.py >/home/ssm-user/work/work-out/cot/v3_pro_oversize.log 2>&1
