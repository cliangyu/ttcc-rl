"""Build a video-swap eval JSONL from the 8-ad overfit eval.

For each row i, replace the video+audio paths with ad ((i+3) mod 8)'s files.
Keep everything else (system, user message, T, R_true, ad_id of the PROMPT).

We also record both:
  - prompt_ad_id   (ad whose prompt is used; truth target if model is text-memorizing)
  - video_ad_id    (ad whose video is fed; truth target if model uses video)
  - R_true_prompt  (original target)
  - R_true_video   (true curve for the SWAPPED video)
"""
from __future__ import annotations
import json, re
from pathlib import Path

SRC = "/home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8_eval.jsonl"
TRAIN_SRC = "/home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8.jsonl"
OUT = "/home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8_swap.jsonl"

rows = [json.loads(l) for l in open(SRC)]
# We need R_true for each ad. The eval JSONL already has it (from the original sample).
# Build a map ad_id -> R_true from the training file (which has assistant + R_true).
truth_map = {}
T_map = {}
for r in [json.loads(l) for l in open(TRAIN_SRC)]:
    truth_map[r["ad_id"]] = list(r["R_true"])
    T_map[r["ad_id"]] = int(r["T"])

n = len(rows)
SHIFT = 3  # cycle by 3 to ensure every row gets a different ad's video
assert n == 8

out_rows = []
for i, r in enumerate(rows):
    j = (i + SHIFT) % n
    swap_ad = rows[j]["ad_id"]
    swap_video_path = rows[j]["videos"][0]
    swap_audio_path = rows[j]["audios"][0]

    new = dict(r)
    new["videos"] = [swap_video_path]
    new["audios"] = [swap_audio_path]
    # Tracking metadata
    new["prompt_ad_id"] = r["ad_id"]
    new["video_ad_id"] = swap_ad
    new["R_true_prompt"] = list(r["R_true"])  # original target
    new["R_true_video"] = truth_map[swap_ad]
    new["T_prompt"] = int(r["T"])
    new["T_video"] = T_map[swap_ad]
    out_rows.append(new)

with open(OUT, "w") as g:
    for r in out_rows:
        g.write(json.dumps(r) + "\n")

print(f"wrote {len(out_rows)} swapped rows to {OUT}")
for r in out_rows:
    print(f"  prompt={r['prompt_ad_id']} (T_p={r['T_prompt']}) <- video={r['video_ad_id']} (T_v={r['T_video']})")
