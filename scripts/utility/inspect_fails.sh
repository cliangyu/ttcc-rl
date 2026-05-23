#!/bin/bash
source /home/ssm-user/work/venv/bin/activate
python <<'PY'
import json
from pathlib import Path
src = "/home/ssm-user/work/work-out/cot/v3_pro_merged.jsonl"
bad = {"7314162629433737217", "7552518277650645009", "7583757874478120961",
       "7527129753511706632", "7561807350107488273", "7394659479395663873",
       "7586187965820076048", "7481317091302129672", "7580334501797429256",
       "7597011350359425040"}
for line in open(src):
    r = json.loads(line)
    if r["ad_id"] in bad:
        print("=" * 80)
        print(f"ad={r['ad_id']} T={r['T']}")
        print(r["raw"])
        print()
PY
