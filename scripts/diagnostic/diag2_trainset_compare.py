"""Diagnostic 2 (analysis): compute IBS + cross-ad Spearman on the train slice.

Inputs:
    /home/ssm-user/work/data/ttcc_swift_v2cot/ttcc_train_eval50.jsonl  (50 train ads with truth)
    /tmp/diag2_infer.jsonl                                             (model predictions)

Compares train-set fidelity to test-set fidelity (see diagnostics/ANALYSIS.md).
"""
import json, re
import numpy as np
from scipy.stats import spearmanr, pearsonr

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
def parse_curve(text, T):
    cleaned = text.replace("```json", "").replace("```", "")
    nums = None
    start = cleaned.find("{")
    while start != -1 and nums is None:
        depth = 0
        for end in range(start, len(cleaned)):
            if cleaned[end] == "{":
                depth += 1
            elif cleaned[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:end+1])
                        if isinstance(obj, dict) and "R" in obj:
                            nums = [float(x) for x in obj["R"]]
                    except Exception:
                        pass
                    break
        start = cleaned.find("{", start+1)
    if nums is None:
        m = re.search(r'\bR\b\s*[:=]\s*\[', cleaned)
        if m:
            tail = cleaned[m.end():]
            eb = tail.find("]")
            extracted = [float(s) for s in NUM_RE.findall(tail if eb < 0 else tail[:eb])]
            if extracted:
                nums = extracted
    if nums is None:
        return None
    if len(nums) < T+1:
        nums = nums + [nums[-1]] * (T+1 - len(nums))
    elif len(nums) > T+1:
        nums = nums[:T+1]
    nums[0] = 1.0
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            nums[i] = nums[i-1]
        nums[i] = max(0.0, min(1.0, nums[i]))
    return nums


train_eval = [json.loads(l) for l in open("/home/ssm-user/work/data/ttcc_swift_v2cot/ttcc_train_eval50.jsonl")]
infer = [json.loads(l) for l in open("/tmp/diag2_infer.jsonl")]
print(f"train rows={len(train_eval)}, infer rows={len(infer)}")

n = min(len(train_eval), len(infer))
ibs_vals, pearson_vals = [], []
true_means, pred_means, preds = [], [], []
for tr, ir in zip(train_eval[:n], infer[:n]):
    T = int(tr["T"]); R_true = np.asarray(tr["R_true"], dtype=float)
    resp = ir.get("response") or ir.get("completion") or ir.get("output") or ""
    if not resp: continue
    R_hat = parse_curve(resp, T)
    if R_hat is None: continue
    R_hat = np.asarray(R_hat[:T+1])
    ibs_vals.append(float(((R_hat - R_true[:T+1])**2).mean()))
    true_means.append(R_true[1:T+1].mean())
    pred_means.append(R_hat[1:T+1].mean())
    if T >= 10:
        r, _ = pearsonr(R_true[1:T+1], R_hat[1:T+1])
        if not np.isnan(r): pearson_vals.append(r)
    preds.append(R_hat)

print(f"=== train-set 50-ad eval ===")
print(f"  parsed              : {len(ibs_vals)} / {n}")
print(f"  IBS (mean)          : {np.mean(ibs_vals):.4f}")
print(f"  cross-ad Spearman   : {spearmanr(true_means, pred_means)[0]:+.3f}")
print(f"  median per-ad Pearson: {np.median(pearson_vals):+.3f}")
vals_t10 = [c[10] for c in preds if len(c) > 10]
print(f"  across-ad std @ t=10: {np.std(vals_t10):.4f}  (truth ~0.081 on test)")
