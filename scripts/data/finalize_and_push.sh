#!/bin/bash
set -e
source /home/ssm-user/work/venv/bin/activate
cd /home/ssm-user/work

python <<'PY'
import json
from pathlib import Path
SRC = Path("/home/ssm-user/work/work-out/cot/v3_pro_merged_clean.jsonl")
DST = Path("/home/ssm-user/work/work-out/cot/v3_pro_merged_final.jsonl")
DROPS = {"7394659479395663873", "7586187965820076048", "7597011350359425040"}
n = 0
with DST.open("w") as fout:
    for line in SRC.open():
        r = json.loads(line)
        if r["ad_id"] in DROPS:
            continue
        fout.write(line)
        n += 1
print(f"final clean rows: {n} (dropped {len(DROPS)} from {n+len(DROPS)})")
PY

# Final QC: should be zero fatal failures now
python /home/ssm-user/work/scripts/qc_cot.py \
  --in /home/ssm-user/work/work-out/cot/v3_pro_merged_final.jsonl \
  --report /home/ssm-user/work/work-out/cot/v3_pro_qc_report_final.json
