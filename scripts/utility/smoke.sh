#!/usr/bin/env bash
set -euxo pipefail
exec > /opt/dlami/nvme/smoke.log 2>&1
echo "[smoke] $(date) start"
PY=/opt/dlami/nvme/work/venv/bin/python
export VLLM_USE_V1=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/opt/dlami/nvme/hf-cache
mkdir -p "$HF_HOME"
$PY <<PYEOF
import os, time
print("[smoke] importing vllm...")
t0 = time.time()
from vllm import LLM, SamplingParams
print(f"[smoke] vllm imported in {time.time()-t0:.1f}s")
t0 = time.time()
print("[smoke] loading Qwen2.5-Omni-3B...")
engine = LLM(
    model="Qwen/Qwen2.5-Omni-3B",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.6,
    enforce_eager=True,
    trust_remote_code=True,
    limit_mm_per_prompt={"video": 1, "audio": 1},
)
print(f"[smoke] model loaded in {time.time()-t0:.1f}s")
print("[smoke] running text-only generation...")
out = engine.generate(["Hello! Reply in one short sentence."], SamplingParams(temperature=0, max_tokens=20))
print("[smoke] output:", repr(out[0].outputs[0].text))
print("[smoke] DONE")
PYEOF
