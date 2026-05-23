#!/usr/bin/env bash
# Inference using v38's full-FT checkpoint (not a LoRA adapter)
# Usage: bash infer_v38.sh <jsonl_input> <out_jsonl>
set -euo pipefail

INPUT_JSONL="${1:?input jsonl required}"
OUT_JSONL="${2:?output jsonl required}"
CKPT=/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full/v35-20260522-150156/checkpoint-80
VENV=/opt/dlami/nvme/work/swift_venv

# Match v38's training settings for inference (max_pixels=200704, FPS=1, FPS_MAX=32)
export ENABLE_AUDIO_OUTPUT="False"
CUDA_VISIBLE_DEVICES=0,1 \
FPS=1.0 \
FPS_MAX_FRAMES=32 \
MAX_PIXELS=200704 \
VIDEO_MAX_PIXELS=200704 \
VIDEO_MAX_TOKEN_NUM=4096 \
"${VENV}/bin/python" -m swift.cli.main infer \
    --model "${CKPT}" \
    --infer_backend vllm \
    --val_dataset "${INPUT_JSONL}" \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --top_p 1.0 \
    --result_path "${OUT_JSONL}" \
    --max_pixels 200704

echo "infer done: ${OUT_JSONL}"
