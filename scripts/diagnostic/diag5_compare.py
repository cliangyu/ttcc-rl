"""Diagnostic 5: video-swap comparison.

Compares each prediction (made with swapped prompt+video pairs) against
two hypotheses:
  H_text : model emits the curve associated with the PROMPT (text memorization)
  H_video: model emits the curve associated with the VIDEO (video learning)

If predictions match H_text => model memorized text, ignored video.
If predictions match H_video => model used video signal.

Inputs:
    /home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8_swap.jsonl
    /tmp/diag5_swap_infer.jsonl
"""
import json, re
import numpy as np

EVAL = "/home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8_swap.jsonl"
INFER = "/tmp/diag5_swap_infer.jsonl"

evals = [json.loads(l) for l in open(EVAL)]
infers = [json.loads(l) for l in open(INFER)]

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
def parse_curve(text, T):
    cleaned = text.replace("```json","").replace("```","")
    nums = None
    start = cleaned.find("{")
    while start != -1 and nums is None:
        depth = 0
        for end in range(start, len(cleaned)):
            if cleaned[end]=="{": depth+=1
            elif cleaned[end]=="}":
                depth-=1
                if depth==0:
                    try:
                        obj = json.loads(cleaned[start:end+1])
                        if isinstance(obj, dict) and "R" in obj:
                            nums = [float(x) for x in obj["R"]]
                    except: pass
                    break
        start = cleaned.find("{", start+1)
    if nums is None: return None
    if len(nums)<T+1: nums = nums + [nums[-1]]*(T+1-len(nums))
    elif len(nums)>T+1: nums = nums[:T+1]
    nums[0] = 1.0
    for i in range(1, len(nums)):
        if nums[i]>nums[i-1]: nums[i] = nums[i-1]
        nums[i] = max(0.0, min(1.0, nums[i]))
    return np.asarray(nums)

print(f"{'prompt_ad':22s} {'video_ad':22s} {'T_p':>3s} {'T_v':>3s} {'IBS vs prompt':>15s} {'IBS vs video':>14s}  verdict")
print("-"*100)
ibs_text, ibs_video = [], []
for ev, inf in zip(evals, infers):
    T_p = int(ev["T_prompt"]); T_v = int(ev["T_video"])
    R_p = np.asarray(ev["R_true_prompt"], dtype=float)[:T_p+1]
    R_v = np.asarray(ev["R_true_video"], dtype=float)[:T_v+1]
    resp = inf.get("response") or ""
    R_hat = parse_curve(resp, T_p)
    if R_hat is None:
        print(f"  PARSE FAIL ad={ev['prompt_ad_id']}")
        continue
    # Compare to prompt's truth on T_p horizon
    err_p = float(((R_hat[:T_p+1] - R_p)**2).mean())
    # Compare to video's truth — align lengths
    L = min(len(R_hat), len(R_v))
    err_v = float(((R_hat[:L] - R_v[:L])**2).mean())
    ibs_text.append(err_p); ibs_video.append(err_v)
    verdict = "TEXT-mem" if err_p < err_v - 0.001 else ("VIDEO-track" if err_v < err_p - 0.001 else "ambiguous")
    print(f"{ev['prompt_ad_id']:22s} {ev['video_ad_id']:22s} {T_p:3d} {T_v:3d} {err_p:15.5f} {err_v:14.5f}  {verdict}")

print()
print(f"mean IBS vs PROMPT truth: {np.mean(ibs_text):.5f}")
print(f"mean IBS vs VIDEO truth : {np.mean(ibs_video):.5f}")
print(f"(reference: identity match on 8 same-ad ads = 0.00000)")

if np.mean(ibs_text) < np.mean(ibs_video) - 0.001:
    print()
    print("CONCLUSION: model is TEXT-MEMORIZING. The video signal is being ignored.")
elif np.mean(ibs_video) < np.mean(ibs_text) - 0.001:
    print()
    print("CONCLUSION: model TRACKS THE VIDEO. Predictions follow the swapped video, not the original prompt.")
else:
    print()
    print("CONCLUSION: ambiguous — predictions track neither cleanly. Could be a mixed regime.")
