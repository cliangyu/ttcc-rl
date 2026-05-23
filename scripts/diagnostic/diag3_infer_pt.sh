#!/bin/bash
# Retry inference for diag 3 using PyTorch backend (vllm crashed previously).
set -uo pipefail
VENV=/opt/dlami/nvme/work/swift_venv
CKPT=/opt/dlami/nvme/ssm-out/diag3_overfit8/v0-20260521-215700/checkpoint-800
EVAL_JSONL=/home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8_eval.jsonl
OUT_DIR=/opt/dlami/nvme/ssm-out/diag3_overfit8
RESULT=/tmp/diag3_infer_pt.jsonl

export WANDB_DISABLED=true
CUDA_VISIBLE_DEVICES=1 \
  FPS_MAX_FRAMES=32 FPS=1.0 \
  MAX_PIXELS=200704 VIDEO_MAX_PIXELS=200704 \
  VIDEO_MAX_TOKEN_NUM=8192 \
  "${VENV}/bin/python" -m swift.cli.main infer \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --adapters "${CKPT}" \
    --infer_backend pt \
    --val_dataset "${EVAL_JSONL}" \
    --max_new_tokens 1024 --temperature 0.0 --top_p 1.0 \
    --result_path "${RESULT}" --max_pixels 200704 \
  > "${OUT_DIR}/infer_pt.log" 2>&1

cp "${RESULT}" "${OUT_DIR}/infer_results_pt.jsonl"
echo "INFER_DONE_PT"
