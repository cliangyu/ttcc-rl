#!/bin/bash
set -e
source /home/ssm-user/work/venv/bin/activate
cd /home/ssm-user/work

python <<'PY'
import json
from pathlib import Path

INLINE = Path("/home/ssm-user/work/work-out/cot/v3_pro_full.jsonl")
OVER   = Path("/home/ssm-user/work/work-out/cot/v3_pro_oversize.jsonl")
OUT    = Path("/home/ssm-user/work/work-out/cot/v3_pro_merged.jsonl")

# Manifest order from v2 file so we ship the same canonical order
V2 = Path("/home/ssm-user/work/work-out/cot/v2_full_instruct_merged.jsonl")
order = [json.loads(l)["ad_id"] for l in V2.open()]

idx = {}
for src in [INLINE, OVER]:
    for line in src.open():
        rec = json.loads(line)
        idx[rec["ad_id"]] = rec

with OUT.open("w") as fout:
    n = 0
    missing = 0
    for ad_id in order:
        if ad_id in idx:
            fout.write(json.dumps(idx[ad_id]) + "\n")
            n += 1
        else:
            missing += 1
print(f"merged: {n}/{len(order)}  missing: {missing}")
print(f"  inline file: {sum(1 for _ in INLINE.open())} lines")
print(f"  oversize file: {sum(1 for _ in OVER.open())} lines")
print(f"  merged file:  {sum(1 for _ in OUT.open())} lines -> {OUT}")

# Sample first record so we can eyeball
first = json.loads(OUT.open().readline())
print("schema:", sorted(first.keys()))
print("first ad_id:", first["ad_id"], "T:", first["T"])
print("raw preview:", first["raw"][:200])
PY
