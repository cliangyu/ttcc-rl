"""Diagnostic 1: per-second across-ad std of predictions vs truth.

Mode collapse looks like: model emits ~the same curve regardless of ad input,
so across-ad std is much smaller than truth's across-ad std.

Usage:
    python diag1_variance.py
Outputs to stdout. Reproduces: /home/ssm-user/work/diagnostics/logs/diag1.txt
"""
import os, json, sys
import numpy as np
import pyarrow.parquet as pq

WORK = "/home/ssm-user/work/work-out"
TEST = "/home/ssm-user/work/data/ttcc_swift/ttcc_test.jsonl"

truth = {}
with open(TEST) as f:
    for line in f:
        r = json.loads(line)
        truth[str(r["ad_id"])] = (int(r["T"]), list(r["R_true"]))


def across_ad_std(curves_by_ad, t):
    vals = []
    for c in curves_by_ad.values():
        if len(c) > t:
            vals.append(c[t])
    return np.std(vals) if vals else 0.0


truth_std = {t: across_ad_std({k: v[1] for k, v in truth.items()}, t)
             for t in [0, 1, 5, 10, 20, 30]}

models = [
    ("B1 train-mean",       "preds_b1.parquet"),
    ("v1 SFT (LoRA)",       "preds_sft.parquet"),
    ("v1 GRPO-179",         "preds_grpo_ext179.parquet"),
    ("v1 RLOO",             "preds_rloo.parquet"),
    ("v2 SFT-cot full-FT",  "preds_sft_v2cot_full.parquet"),
    ("v2 SFT-nocot full-FT","preds_sft_v2cot_nocot_full.parquet"),
]

print(f"{'model':28s}   " + "  ".join(f"std@t={t:<2d}" for t in [0,1,5,10,20,30]))
print(f"{'TRUTH':28s}   " + "  ".join(f"{truth_std[t]:.4f}  " for t in [0,1,5,10,20,30]))
print("-" * 90)
for label, fn in models:
    path = f"{WORK}/{fn}"
    if not os.path.exists(path):
        print(f"{label:28s}   MISSING"); continue
    t = pq.read_table(path).to_pandas()
    preds = {str(row["ad_id"]): list(row["R_hat"]) for _, row in t.iterrows()}
    print(f"{label:28s}   " + "  ".join(f"{across_ad_std(preds, tt):.4f}  " for tt in [0,1,5,10,20,30]))
