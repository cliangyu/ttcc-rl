"""SFT dataset variant: assistant target = JSON curve only, no CoT.

Used by exp (2) in the self-eval: tests whether the teacher's reasoning
text is load-bearing for SFT IBS, or whether the JSON curve alone is enough.

Output: ``data/ttcc_swift_nocot/ttcc_train_sft.jsonl`` whose rows have
assistant content = ``Curve: {"R": [1.0, ..., R(T)]}`` with no preceding
Content/Drops/Reasoning lines.

Test JSONL is unchanged (we still want the model's eval-time output to be
parseable, and the user prompt is identical).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WORK = Path("/home/ssm-user/work")
T_MIN, T_MAX = 5, 60

SYSTEM_PROMPT = (
    "You are an expert in short-form video advertising. You forecast "
    "second-by-second audience retention curves. R(t) is the fraction of "
    "viewers still watching at second t, with R(0) = 1 by definition. "
    "R(t) is monotone non-increasing. Use the video and audio content to "
    "estimate where viewers drop off."
)


def user_text(T: int) -> str:
    return (
        f"This ad is {T} seconds long. Watch and listen to it, then output a "
        f"JSON object on a single line of the form:\n"
        f"{{\"R\": [1.0, R(1), R(2), ..., R({T})]}}\n"
        f"Rules: exactly {T+1} numbers in R, R(0) = 1.0, every value in [0, 1], "
        f"monotone non-increasing."
    )


def horizon(d, L):
    if d is None or not np.isfinite(d):
        return None
    T_dur = round(float(d))
    if T_dur < T_MIN:
        return None
    T_curve = L - 1
    if min(T_dur, T_MAX) - T_curve > 1:
        return None
    T = min(T_dur, T_MAX, T_curve)
    return T if T >= T_MIN else None


def normalize(curve, T):
    c = np.asarray(curve, dtype=np.float64)
    if c[0] <= 0:
        return None
    c = c[: T + 1] / c[0]
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            if c[i] - c[i - 1] > 5e-3:
                return None
            c[i] = c[i - 1]
    return np.clip(c, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    DATA = WORK / "data/ttcc"
    VIDEOS = WORK / "data/videos/train"
    VIDEOS.mkdir(parents=True, exist_ok=True)

    train_rows = []
    for shard in sorted((DATA / "data").glob("train-*-of-*.parquet")):
        t = pq.read_table(
            shard,
            columns=["ad_id", "duration", "retention_curve", "split", "video_local_path"],
        ).to_pandas()
        t = t[t["split"] == "train"]
        for _, row in t.iterrows():
            raw = row["retention_curve"]
            if raw is None or len(raw) == 0:
                continue
            c = np.asarray(raw, dtype=np.float64)
            if not np.all(np.isfinite(c)) or c[0] <= 0:
                continue
            T = horizon(row["duration"], len(c))
            if T is None:
                continue
            curve = normalize(raw, T)
            if curve is None:
                continue
            ad_id = str(row["ad_id"])
            v = row["video_local_path"]
            if v is None or v.get("bytes") is None:
                continue
            mp4 = VIDEOS / f"{ad_id}.mp4"
            if not mp4.exists():
                mp4.write_bytes(bytes(v["bytes"]))
            train_rows.append({"ad_id": ad_id, "T": T, "R": curve.tolist(), "mp4": str(mp4)})

    print(f"train rows: {len(train_rows)}")

    out_sft = args.out_dir / "ttcc_train_sft.jsonl"
    with open(out_sft, "w") as f:
        for r in train_rows:
            R_str = "[" + ", ".join(f"{x:.4f}" for x in r["R"]) + "]"
            assistant_text = f"Curve: {{\"R\": {R_str}}}"
            f.write(json.dumps({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_text(r["T"])},
                    {"role": "assistant", "content": assistant_text},
                ],
                "videos": [r["mp4"]],
                "audios": [r["mp4"]],
                "T": r["T"],
                "R_true": r["R"],
            }) + "\n")
    print(f"wrote {len(train_rows)} rows → {out_sft}")
    print()
    print("Example assistant target (first row):")
    print(f"  {json.loads(open(out_sft).readline())['messages'][-1]['content'][:120]}...")


if __name__ == "__main__":
    main()
