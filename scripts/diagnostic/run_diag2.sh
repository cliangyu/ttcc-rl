#!/bin/bash
# Diagnostic 2: run SFT-cot v2 full-FT inference on a 50-ad TRAIN slice.
set -uo pipefail

WORK=/home/ssm-user/work
VENV=/opt/dlami/nvme/work/swift_venv
OUT_DIR=/opt/dlami/nvme/ssm-out/diag2
TMP_OUT=/tmp/diag2_infer.jsonl
CKPT="${WORK}/work-out/ttcc_sft_v2cot_full/v0-20260521-081339/checkpoint-450"
TRAIN_EVAL="${WORK}/data/ttcc_swift_v2cot/ttcc_train_eval50.jsonl"

mkdir -p "${OUT_DIR}"
export WANDB_DISABLED=true

CUDA_VISIBLE_DEVICES=0 \
  FPS_MAX_FRAMES=32 FPS=1.0 \
  MAX_PIXELS=200704 VIDEO_MAX_PIXELS=200704 \
  VIDEO_MAX_TOKEN_NUM=8192 \
  "${VENV}/bin/python" -m swift.cli.main infer \
    --model "${CKPT}" \
    --infer_backend vllm \
    --val_dataset "${TRAIN_EVAL}" \
    --max_new_tokens 1024 --temperature 0.0 --top_p 1.0 \
    --result_path "${TMP_OUT}" --max_pixels 200704 \
  > "${OUT_DIR}/infer.log" 2>&1

cp "${TMP_OUT}" "${OUT_DIR}/infer_results.jsonl"
echo "DIAG2_DONE"
