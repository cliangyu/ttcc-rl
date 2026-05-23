"""Build 8-ad overfit SFT dataset + matching eval JSONL."""
import json, random, re

random.seed(7)
SRC = "/home/ssm-user/work/data/ttcc_swift_v2cot/ttcc_train_sft.jsonl"
OUT = "/home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8.jsonl"
OUT_EVAL = "/home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8_eval.jsonl"

rows = []
with open(SRC) as f:
    for line in f:
        rows.append(json.loads(line))

mid = [r for r in rows if 10 <= r["T"] <= 40]
sample = random.sample(mid, 8)
for r in sample:
    m = re.search(r"/(\d+)\.mp4", r["videos"][0])
    r["ad_id"] = m.group(1) if m else "unknown"

with open(OUT, "w") as g:
    for r in sample:
        g.write(json.dumps(r) + "\n")

with open(OUT_EVAL, "w") as g:
    for r in sample:
        r2 = dict(r)
        r2["messages"] = [m for m in r["messages"] if m["role"] != "assistant"]
        g.write(json.dumps(r2) + "\n")

ts = [r["T"] for r in sample]
ids = [r["ad_id"] for r in sample]
print(f"wrote {len(sample)} ads, Ts={ts}, ad_ids={ids[:3]}...")
print(f"  SFT data: {OUT}")
print(f"  eval data: {OUT_EVAL}")
