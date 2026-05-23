#!/bin/bash
set -e
source /home/ssm-user/work/venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=/home/ssm-user/work/vizzy-sa.json
cd /home/ssm-user/work

python <<'PY'
"""Regenerate the 10 QC-failed ads with seed=1 and a tightened user prompt
that adds an extra reminder about T-bound and on-screen percentages."""
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from google.cloud import storage

PROJECT = "vizzylabs-ai-prod"
LOCATION = "global"
MODEL = "gemini-3.1-pro-preview"
BUCKET = "ttcc-cot-staging-191aab"
PREFIX = "v3-pro/oversize/"
INLINE_LIMIT_BYTES = 20 * 1024 * 1024

FAIL_IDS = [
    "7314162629433737217", "7552518277650645009", "7583757874478120961",
    "7527129753511706632", "7561807350107488273", "7394659479395663873",
    "7586187965820076048", "7481317091302129672", "7580334501797429256",
    "7597011350359425040",
]

SYSTEM_PROMPT = (
    "You are a careful analyst of short-form video ad engagement. You will "
    "watch and listen to a short ad. Your task is to predict, based ONLY on "
    "what you actually see and hear, which moments are most likely to lose "
    "viewer attention. You do NOT have access to the retention curve, drop "
    "percentages, or any audience-measurement numbers. Never invent or "
    "reference such numbers. Ground every claim in a concrete on-screen or "
    "audio event."
)

def build_user_instruction(T: int) -> str:
    # NOTE: identical to v2 plus two extra reminders about the failures we saw.
    return (
        f"This ad is {T} seconds long. Watch and listen to it from t=0 to "
        f"t={T} seconds.\n\n"
        f"Write a SHORT analysis with EXACTLY three labeled lines:\n"
        f"Content: <one sentence describing what's actually shown -- "
        f"product/scene, opening hook, pacing>.\n"
        f"Drops: <one or two sentences naming 2-3 SPECIFIC seconds where "
        f"viewer attention is most likely to lapse, each tied to a concrete "
        f"event you observed (e.g. 'static dialogue starts at t=4', "
        f"'unrelated b-roll cut at t=12', 'product disclaimer at t=20'). "
        f"Cite seconds in [0, {T}] only.>\n"
        f"Reasoning: <one sentence on why this combination of events would "
        f"shape audience drop-off>.\n\n"
        f"Strict rules:\n"
        f"- Do NOT mention any retention curve, R value, percentage, "
        f"fraction, or numeric drop magnitude. This applies even when the ad "
        f"contains on-screen percentages or sale text -- describe such "
        f"overlays in words (e.g. 'a large sale-discount overlay') without "
        f"quoting the digits.\n"
        f"- Every t=N you cite MUST satisfy 0 <= N <= {T}. Use integer "
        f"seconds only; never mm:ss notation.\n"
        f"- Do NOT write phrases like 'retention drops to', 'falls to 0.2', "
        f"'a 50% drop'.\n"
        f"- Output exactly the three labeled lines and nothing else."
    )


# Load existing rows so we can replace selectively
SRC = Path("/home/ssm-user/work/work-out/cot/v3_pro_merged.jsonl")
DST = Path("/home/ssm-user/work/work-out/cot/v3_pro_merged_clean.jsonl")
rows = [json.loads(l) for l in SRC.open()]
by_id = {r["ad_id"]: r for r in rows}

# Pull mp4 + T metadata for each failure
manifests = []
for ad_id in FAIL_IDS:
    if ad_id not in by_id:
        print(f"WARN: {ad_id} not in source")
        continue
    r = by_id[ad_id]
    mp4 = f"/home/ssm-user/work/data/videos/train/{ad_id}.mp4"
    sz = os.path.getsize(mp4)
    manifests.append({"ad_id": ad_id, "T": r["T"], "R": r["R_true"], "mp4": mp4, "size": sz})

# Stage oversize ones into GCS (they should already be there from earlier run)
storage_client = storage.Client(project=PROJECT)
bucket = storage_client.bucket(BUCKET)
for m in manifests:
    if m["size"] > INLINE_LIMIT_BYTES:
        blob = bucket.blob(f"{PREFIX}{m['ad_id']}.mp4")
        if not blob.exists():
            blob.upload_from_filename(m["mp4"])
        m["gcs_uri"] = f"gs://{BUCKET}/{PREFIX}{m['ad_id']}.mp4"

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

def gen(m):
    T = m["T"]
    if "gcs_uri" in m:
        part = types.Part.from_uri(file_uri=m["gcs_uri"], mime_type="video/mp4")
    else:
        part = types.Part.from_bytes(data=open(m["mp4"],"rb").read(), mime_type="video/mp4")
    t0 = time.time()
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[part, build_user_instruction(T)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.4, top_p=0.95,
                    max_output_tokens=512, seed=1,  # different seed!
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = resp.text
            if not text:
                continue
            return m["ad_id"], text.strip(), time.time()-t0
        except Exception as e:
            time.sleep(2.0*(attempt+1))
    return m["ad_id"], None, time.time()-t0

updated = {}
with ThreadPoolExecutor(max_workers=10) as ex:
    for fut in as_completed([ex.submit(gen, m) for m in manifests]):
        ad_id, text, dt = fut.result()
        if text:
            updated[ad_id] = text
            print(f"  REGEN {ad_id}: dt={dt:.1f}s len={len(text)}")
        else:
            print(f"  FAIL  {ad_id}: dt={dt:.1f}s")

# Replace failing rows; preserve order
with DST.open("w") as fout:
    for r in rows:
        if r["ad_id"] in updated:
            r = dict(r)
            r["raw"] = updated[r["ad_id"]]
        fout.write(json.dumps(r) + "\n")
print(f"wrote: {DST}")
PY
