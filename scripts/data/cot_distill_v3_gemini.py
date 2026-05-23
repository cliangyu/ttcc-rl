"""Causal CoT distillation (v3): Gemini teacher via Vertex AI.

Diff vs v2 (cot_distill_v2.py):
  - Teacher is gemini-3.1-pro-preview OR gemini-3.5-flash on Vertex AI 'global'.
  - Video is sent inline (Part.from_bytes) — videos > 20MB are skipped and logged
    for a separate GCS pass (~38 of 717 ads on this dataset).
  - Concurrency via threads (SDK is sync). Default 4 workers.
  - Thinking disabled (thinking_budget=0) so output is the 3 labeled lines only.

Output schema identical to v2: JSONL with {ad_id, T, R_true, raw}. R_true is
ground-truth carry-through, never shown to the teacher.

Prompt SYSTEM_PROMPT and build_user_instruction are byte-for-byte the v2 versions.

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json \
    python cot_distill_v3_gemini.py --model pro --pilot 5 \
        --out /home/ssm-user/work/work-out/cot/v3_pro_pilot.jsonl
    GOOGLE_APPLICATION_CREDENTIALS=... \
    python cot_distill_v3_gemini.py --model pro --full --concurrency 4 \
        --out /home/ssm-user/work/work-out/cot/v3_pro_full.jsonl
"""
from __future__ import annotations
import argparse
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

WORK = Path("/home/ssm-user/work")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("cot_distill_v3")

MODEL_IDS = {
    "pro": "gemini-3.1-pro-preview",
    "flash": "gemini-3.5-flash",
}

INLINE_LIMIT_BYTES = 20 * 1024 * 1024  # Vertex inline cap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_IDS), required=True)
    p.add_argument("--pilot", type=int, default=0)
    p.add_argument("--full", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-ads", type=int, default=None)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--project", default="vizzylabs-ai-prod")
    p.add_argument("--location", default="global")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


def build_manifests(start_idx: int, stride: int, max_ads: int | None):
    """Byte-identical to v2 build_manifests — keep in sync."""
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
    VIDEOS.mkdir(parents=True, exist_ok=True)

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
            if not mp4.exists():
                mp4.write_bytes(bytes(v["bytes"]))
            manifests.append(
                {"ad_id": ad_id, "T": T, "R": np.clip(c, 0, 1).tolist(), "mp4": str(mp4)}
            )
    manifests = manifests[start_idx::stride]
    if max_ads:
        manifests = manifests[:max_ads]
    return manifests


# Prompts: byte-identical to cot_distill_v2.py SYSTEM_PROMPT + build_user_instruction
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


def call_gemini(client, model_id, video_bytes, T, temperature, top_p, max_tokens, seed):
    from google.genai import types
    sys_inst = SYSTEM_PROMPT
    user_inst = build_user_instruction(T)
    resp = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
            user_inst,
        ],
        config=types.GenerateContentConfig(
            system_instruction=sys_inst,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            seed=seed,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = resp.text
    finish = None
    if resp.candidates:
        finish = getattr(resp.candidates[0], "finish_reason", None)
    return text, finish, resp.usage_metadata


_write_lock = threading.Lock()


def process_one(client, model_id, m, args, fout):
    ad_id, T, mp4 = m["ad_id"], m["T"], m["mp4"]
    t0 = time.time()
    sz = os.path.getsize(mp4)
    if sz > INLINE_LIMIT_BYTES:
        return ad_id, T, "OVERSIZE", sz, None
    with open(mp4, "rb") as f:
        vbytes = f.read()
    last_err = None
    for attempt in range(args.retries):
        try:
            text, finish, usage = call_gemini(
                client, model_id, vbytes, T,
                args.temperature, args.top_p, args.max_tokens, args.seed,
            )
            if not text:
                last_err = f"empty text finish={finish}"
                time.sleep(1.5 * (attempt + 1))
                continue
            rec = {"ad_id": ad_id, "T": T, "R_true": m["R"], "raw": text.strip()}
            with _write_lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            return ad_id, T, "OK", sz, time.time() - t0
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2.0 * (attempt + 1))
    return ad_id, T, f"FAIL: {last_err}", sz, time.time() - t0


def main():
    args = parse_args()
    from google import genai
    model_id = MODEL_IDS[args.model]
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    log.info(f"vertex client ready: project={args.project} loc={args.location} model={model_id}")

    if args.pilot > 0:
        manifests = build_manifests(start_idx=0, stride=1, max_ads=args.pilot)
    elif args.full:
        manifests = build_manifests(
            start_idx=args.start_idx, stride=args.stride, max_ads=args.max_ads
        )
    else:
        log.error("specify --pilot N or --full")
        sys.exit(1)
    log.info(f"to process: {len(manifests)} ads, concurrency={args.concurrency}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, ok, oversize, fail = 0, 0, 0, 0
    t_start = time.time()
    with open(args.out, "w") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {
                ex.submit(process_one, client, model_id, m, args, fout): m
                for m in manifests
            }
            for fut in as_completed(futs):
                ad_id, T, status, sz, dt = fut.result()
                done += 1
                tag = "OK"
                if status == "OK":
                    ok += 1
                elif status == "OVERSIZE":
                    oversize += 1
                    tag = "OVERSIZE"
                else:
                    fail += 1
                    tag = "FAIL"
                log.info(
                    f"[{done}/{len(manifests)}] ad={ad_id} T={T} {tag} "
                    f"size={sz/1e6:.1f}MB dt={dt:.1f}s "
                    f"({status if tag != 'OK' else 'ok'})"
                )
    elapsed = time.time() - t_start
    log.info(
        f"DONE in {elapsed:.0f}s: ok={ok} oversize={oversize} fail={fail} "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
