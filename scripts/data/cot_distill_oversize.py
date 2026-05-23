"""Oversize-only CoT distillation: GCS-staged videos for the 38 ads >20MB.

Mirrors cot_distill_v3_gemini.py exactly — same SYSTEM_PROMPT, same user instruction,
same model (gemini-3.1-pro-preview on Vertex 'global'), thinking_budget=0,
same temperature/top_p/seed/max_tokens. Only difference: uses Part.from_uri(gs://...).
"""
from __future__ import annotations
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from google import genai
from google.genai import types
from google.cloud import storage

WORK = Path("/home/ssm-user/work")
PROJECT = "vizzylabs-ai-prod"
LOCATION = "global"
MODEL = "gemini-3.1-pro-preview"
BUCKET = "ttcc-cot-staging-191aab"
PREFIX = "v3-pro/oversize/"
OUT = WORK / "work-out/cot/v3_pro_oversize.jsonl"
DONE_FILE = WORK / "work-out/cot/v3_pro_full.jsonl"
INLINE_LIMIT_BYTES = 20 * 1024 * 1024
CONCURRENCY = 16
TEMPERATURE = 0.4
TOP_P = 0.95
MAX_TOKENS = 512
SEED = 0
RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("oversize")


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
        f"fraction, or numeric drop magnitude.\n"
        f"- Do NOT write phrases like 'retention drops to', 'falls to 0.2', "
        f"'a 50% drop'.\n"
        f"- Output exactly the three labeled lines and nothing else."
    )


def build_manifests():
    """Byte-identical to cot_distill_v2/v3 build_manifests."""
    T_MIN, T_MAX = 5, 60

    def horizon(d, L):
        Td = round(float(d))
        if Td < T_MIN:
            return None
        Tc = L - 1
        if min(Td, T_MAX) - Tc > 1:
            return None
        T = min(Td, T_MAX, Tc)
        return T if T >= T_MIN else None

    VIDEOS = WORK / "data/videos/train"
    manifests = []
    DATA = WORK / "data/ttcc"
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
            c = c[: T + 1] / c[0]
            ok = True
            for i in range(1, len(c)):
                if c[i] > c[i - 1]:
                    if c[i] - c[i - 1] > 5e-3:
                        ok = False
                        break
                    c[i] = c[i - 1]
            if not ok:
                continue
            v = row["video_local_path"]
            if v is None or v.get("bytes") is None:
                continue
            ad_id = str(row["ad_id"])
            mp4 = VIDEOS / f"{ad_id}.mp4"
            manifests.append({"ad_id": ad_id, "T": T, "R": np.clip(c, 0, 1).tolist(), "mp4": str(mp4)})
    return manifests


def ensure_bucket(storage_client):
    try:
        b = storage_client.get_bucket(BUCKET)
        log.info(f"bucket gs://{BUCKET} exists")
    except Exception:
        b = storage_client.create_bucket(BUCKET, location="us-central1")
        log.info(f"created bucket gs://{BUCKET}")
    return b


def upload_one(bucket, mp4_path, ad_id):
    blob = bucket.blob(f"{PREFIX}{ad_id}.mp4")
    if blob.exists():
        return f"gs://{BUCKET}/{PREFIX}{ad_id}.mp4", "cached"
    blob.upload_from_filename(mp4_path)
    return f"gs://{BUCKET}/{PREFIX}{ad_id}.mp4", "uploaded"


_lock = threading.Lock()


def call_gemini(client, gcs_uri, T):
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_uri(file_uri=gcs_uri, mime_type="video/mp4"),
            build_user_instruction(T),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE, top_p=TOP_P,
            max_output_tokens=MAX_TOKENS, seed=SEED,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text, (resp.candidates[0].finish_reason if resp.candidates else None)


def process(client, m, fout):
    ad_id, T, gcs_uri = m["ad_id"], m["T"], m["gcs_uri"]
    t0 = time.time()
    last = None
    for attempt in range(RETRIES):
        try:
            text, finish = call_gemini(client, gcs_uri, T)
            if not text:
                last = f"empty text finish={finish}"
                time.sleep(2.0 * (attempt + 1))
                continue
            rec = {"ad_id": ad_id, "T": T, "R_true": m["R"], "raw": text.strip()}
            with _lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            return ad_id, "OK", time.time() - t0
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:160]}"
            time.sleep(2.0 * (attempt + 1))
    return ad_id, f"FAIL: {last}", time.time() - t0


def main():
    # Determine which ads are still missing
    done_ids = set()
    if DONE_FILE.exists():
        for l in DONE_FILE.read_text().splitlines():
            done_ids.add(json.loads(l)["ad_id"])
    manifests = build_manifests()
    oversize = [m for m in manifests
                if os.path.getsize(m["mp4"]) > INLINE_LIMIT_BYTES
                and m["ad_id"] not in done_ids]
    log.info(f"missing oversize: {len(oversize)}")

    storage_client = storage.Client(project=PROJECT)
    bucket = ensure_bucket(storage_client)

    log.info(f"uploading {len(oversize)} videos to gs://{BUCKET}/{PREFIX} ...")
    t_up = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(upload_one, bucket, m["mp4"], m["ad_id"]): m for m in oversize}
        for f in as_completed(futs):
            m = futs[f]
            uri, status = f.result()
            m["gcs_uri"] = uri
            log.info(f"  {m['ad_id']}: {status} -> {uri}")
    log.info(f"upload done in {time.time()-t_up:.0f}s")

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    log.info(f"running inference at concurrency={CONCURRENCY}")
    t_inf = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    with open(OUT, "w") as fout:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = {ex.submit(process, client, m, fout): m for m in oversize}
            done_n = 0
            for fut in as_completed(futs):
                ad_id, status, dt = fut.result()
                done_n += 1
                if status == "OK":
                    ok += 1
                    log.info(f"[{done_n}/{len(oversize)}] {ad_id} OK dt={dt:.1f}s")
                else:
                    fail += 1
                    log.warning(f"[{done_n}/{len(oversize)}] {ad_id} {status}")
    log.info(f"inference done in {time.time()-t_inf:.0f}s: ok={ok} fail={fail}")
    log.info(f"output: {OUT}")


if __name__ == "__main__":
    main()
