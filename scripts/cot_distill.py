"""CoT distillation: outcome-conditioned rationale generation for TTCC train split.

Teacher: Qwen3-Omni-30B-A3B-Thinking (or -Instruct, talker disabled either way).
For each train ad: feed (video, audio, GT retention curve) -> teacher writes a
Content/Drops/Reasoning analysis explaining why the curve looks as it does.

Output: JSONL with {ad_id, T, R_true, content, drops, reasoning} per ad.

Usage:
  python cot_distill.py --model THINKING --pilot 5
  python cot_distill.py --model INSTRUCT --pilot 5
  python cot_distill.py --model THINKING --full
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("HF_HOME", "/home/ssm-user/work/hf-cache")

WORK = Path("/home/ssm-user/work")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("cot_distill")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["THINKING", "INSTRUCT"], default="THINKING")
    p.add_argument("--pilot", type=int, default=0, help="if >0, only run on first N train ads")
    p.add_argument("--full", action="store_true", help="run on all clean train ads")
    p.add_argument("--out", type=Path, required=True, help="output JSONL path")
    p.add_argument("--max-ads", type=int, default=None)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--stride", type=int, default=1, help="shard stride (for 2-GPU split: use 2 + start-idx 0/1)")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


MODEL_DIRS = {
    "THINKING": "/home/ssm-user/work/hf-cache/Qwen3-Omni-30B-A3B-Thinking",
    "INSTRUCT": "/opt/dlami/nvme/hf-cache/Qwen3-Omni-30B-A3B-Instruct",
}


def build_manifests(start_idx: int, stride: int, max_ads: int | None):
    T_MIN, T_MAX = 5, 60
    def horizon(d, L):
        Td = round(float(d))
        if Td < T_MIN: return None
        Tc = L - 1
        if min(Td, T_MAX) - Tc > 1: return None
        T = min(Td, T_MAX, Tc)
        return T if T >= T_MIN else None

    VIDEOS = WORK / "data/videos/train"
    VIDEOS.mkdir(parents=True, exist_ok=True)

    manifests = []
    DATA = WORK / "data/ttcc"
    for shard in sorted((DATA / "data").glob("train-*-of-*.parquet")):
        t = pq.read_table(shard, columns=["ad_id","duration","retention_curve","split","video_local_path"]).to_pandas()
        t = t[t["split"] == "train"]
        for _, row in t.iterrows():
            raw = row["retention_curve"]
            if raw is None or len(raw) == 0: continue
            c = np.asarray(raw, dtype=np.float64)
            if not np.all(np.isfinite(c)) or c[0] <= 0: continue
            T = horizon(row["duration"], len(c))
            if T is None: continue
            c = c[:T+1] / c[0]
            ok = True
            for i in range(1, len(c)):
                if c[i] > c[i-1]:
                    if c[i] - c[i-1] > 5e-3: ok=False; break
                    c[i] = c[i-1]
            if not ok: continue
            v = row["video_local_path"]
            if v is None or v.get("bytes") is None: continue
            ad_id = str(row["ad_id"])
            mp4 = VIDEOS / f"{ad_id}.mp4"
            if not mp4.exists():
                mp4.write_bytes(bytes(v["bytes"]))
            manifests.append({"ad_id": ad_id, "T": T, "R": np.clip(c, 0, 1).tolist(), "mp4": str(mp4)})
    # Apply shard / max
    manifests = manifests[start_idx::stride]
    if max_ads:
        manifests = manifests[:max_ads]
    return manifests


SYSTEM_PROMPT = (
    "You are a careful analyst of short-form video ad engagement. Look at the "
    "video and audio, then explain WHY viewer retention drops where it does. "
    "Be specific about scene and audio events at the named seconds. Do not "
    "guess; ground every claim in something you actually see or hear."
)


def build_user_instruction(T: int, R: list[float]) -> str:
    R_str = "[" + ", ".join(f"{x:.2f}" for x in R) + "]"
    # Detect biggest single-step drop to anchor the reasoning
    diffs = -np.diff(R)
    if len(diffs) == 0:
        big_drop_hint = ""
    else:
        biggest = int(np.argmax(diffs))
        big_drop_hint = (
            f" The single biggest drop is at t={biggest}->t={biggest+1} "
            f"(R falls {R[biggest]:.2f} -> {R[biggest+1]:.2f})."
        )
    return (
        f"This ad is {T} seconds long. Its TRUE second-by-second retention curve is:\n"
        f"R = {R_str}\n"
        f"R(0) = 1.0 by definition. R(t) is the fraction of viewers still watching at second t.\n"
        f"{big_drop_hint}\n\n"
        f"Write a SHORT analysis with EXACTLY three labeled lines:\n"
        f"Content: <one sentence describing what's shown -- product/scene, opening hook, pacing>.\n"
        f"Drops: <one or two sentences naming SPECIFIC seconds where retention falls fastest, "
        f"with reasons tied to what happens on screen or in the audio>.\n"
        f"Reasoning: <one sentence summarizing WHY this overall shape makes sense given the content "
        f"(e.g. 'strong hook hooks viewers but slow middle loses them')>.\n"
        f"Do NOT output the curve again, do NOT output JSON, just the three lines."
    )


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sys.path.insert(0, "/home/ssm-user/work/ttcc-inference/src")
    from vllm import LLM, SamplingParams
    from qwen_omni_utils import process_mm_info

    model_dir = MODEL_DIRS[args.model]
    log.info(f"loading engine from {model_dir} ...")
    t0 = time.time()
    # Disable talker via not loading it: pass use_audio_in_video to False; for Instruct,
    # we rely on enforce_eager + a thinker-only sub-config if exposed. The Thinking variant
    # natively excludes the talker.
    engine = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        seed=0,
        limit_mm_per_prompt={"video": 1, "audio": 1},
        trust_remote_code=True,
        enforce_eager=True,  # safer on first run
    )
    log.info(f"engine ready in {time.time()-t0:.1f}s")
    # vLLM's wrapped tokenizer may not expose .chat_template — load a fresh
    # AutoTokenizer from disk and use that for templating.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        ctjson = Path(model_dir) / "chat_template.json"
        if ctjson.exists():
            ctdata = json.loads(ctjson.read_text())
            tok.chat_template = ctdata.get("chat_template", "")
            log.info(f"loaded chat_template ({len(tok.chat_template)} chars) from chat_template.json")

    # Decide manifests
    if args.pilot > 0:
        manifests = build_manifests(start_idx=0, stride=1, max_ads=args.pilot)
    elif args.full:
        manifests = build_manifests(start_idx=args.start_idx, stride=args.stride, max_ads=args.max_ads)
    else:
        log.error("specify --pilot N or --full")
        sys.exit(1)
    log.info(f"to process: {len(manifests)} ads")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as fout:
        for i, m in enumerate(manifests):
            t0 = time.time()
            T, R, video_path = m["T"], m["R"], m["mp4"]
            user_text = build_user_instruction(T, R)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": video_path, "audio_end": float(T)},
                    {"type": "video", "video": video_path, "video_start": 0.0, "video_end": float(T),
                     "fps": 1.0, "max_frames": 64, "max_pixels": 420*420},
                    {"type": "text", "text": user_text},
                ]},
            ]
            try:
                audios, _, videos = process_mm_info(messages, use_audio_in_video=False)
                prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                mm = {}
                if videos: mm["video"] = videos[0] if len(videos)==1 else videos
                if audios: mm["audio"] = audios[0] if len(audios)==1 else audios
                req = {"prompt": prompt_text, "multi_modal_data": mm,
                       "mm_processor_kwargs": {"use_audio_in_video": False}}
                sp = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=2048, seed=0)
                out = engine.generate([req], sp)[0].outputs[0].text
                rec = {"ad_id": m["ad_id"], "T": T, "R_true": R, "raw": out.strip()}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                written += 1
                log.info(f"[{i+1}/{len(manifests)}] ad={m['ad_id']} T={T} ok in {time.time()-t0:.1f}s")
            except Exception as e:
                log.warning(f"ad={m['ad_id']} FAILED: {type(e).__name__}: {e}")
    log.info(f"DONE: wrote {written}/{len(manifests)} CoTs to {args.out}")


if __name__ == "__main__":
    main()
