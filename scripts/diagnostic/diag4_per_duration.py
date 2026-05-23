"""Diagnostic 4: per-duration train-mean baseline.

Build a 4-bin train-mean (bins by ad horizon T), evaluate on test split,
compare to global B1 train-mean. Writes a parquet and prints metrics.
"""
import os, sys, json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

WORK = "/home/ssm-user/work/work-out"
TEST = "/home/ssm-user/work/data/ttcc_swift/ttcc_test.jsonl"
TRAIN = "/home/ssm-user/work/data/ttcc_swift_v2cot/ttcc_train_grpo.jsonl"

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

train = load_jsonl(TRAIN); test = load_jsonl(TEST)
BIN_EDGES = [5, 15, 30, 45, 61]

def bin_of(T):
    for i in range(len(BIN_EDGES)-1):
        if BIN_EDGES[i] <= T < BIN_EDGES[i+1]:
            return i
    return len(BIN_EDGES)-2

def pad(c, T):
    R = np.asarray(c, dtype=float)[:T+1]
    return np.concatenate([R, np.full(60+1-len(R), R[-1])])

bins_train = {i: [] for i in range(len(BIN_EDGES)-1)}
for r in train:
    bins_train[bin_of(int(r["T"]))].append(pad(r["R_true"], int(r["T"])))
print("per-bin counts:", {
    f"[{BIN_EDGES[i]},{BIN_EDGES[i+1]-1}]": len(bins_train[i])
    for i in range(len(BIN_EDGES)-1)})
bin_means = {i: np.mean(arr, axis=0) if arr else None for i, arr in bins_train.items()}

ad_ids, R_hats = [], []
for r in test:
    T = int(r["T"]); b = bin_of(T)
    if bin_means[b] is None: continue
    R_hat = bin_means[b][:T+1].copy(); R_hat[0] = 1.0
    for k in range(1, len(R_hat)):
        if R_hat[k] > R_hat[k-1]: R_hat[k] = R_hat[k-1]
    ad_ids.append(str(r["ad_id"])); R_hats.append(R_hat.tolist())

out_pq = f"{WORK}/B1_per_duration.parquet"
pq.write_table(pa.table({
    "ad_id": pa.array(ad_ids),
    "R_hat": pa.array(R_hats, type=pa.list_(pa.float64())),
    "method": pa.array(["B1_per_duration"] * len(ad_ids)),
    "seed":   pa.array([0] * len(ad_ids), type=pa.int64()),
}), out_pq)

sys.path.insert(0, "/home/ssm-user/work/ttcc-eval/src")
os.environ.setdefault("HF_HOME", "/home/ssm-user/work/hf-cache")
from ttcc_eval.eval import evaluate, paired_compare
rep = evaluate(out_pq, B=2000, seed=0)
m = rep.metrics
def fmt(d): return f"{d['point']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
print("=== B1_per_duration ===")
print(f"  IBS   = {fmt(m['ibs'])}")
print(f"  slope = {fmt(m['calibration_slope'])}")
print(f"  AUC   = {fmt(m['auc_spearman'])}")
cmp = paired_compare(f"{WORK}/B1_train_mean.parquet", out_pq, B=2000, seed=0)
d = cmp["ibs"]["diff"]
v = 'BEATS B1' if d['point']<0 and d['hi']<0 else ('LOSES vs B1' if d['point']>0 and d['lo']>0 else 'tie')
print(f"  dIBS vs B1 = {fmt(d)}  ({v})")
