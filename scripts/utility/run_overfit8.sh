#!/bin/bash
# 8-ad overfit experiment: can the SFT pipeline memorize a tiny dataset?
# Base model + LoRA (faster) + high LR + many epochs, GPU 1 only.
set -uo pipefail

WORK=/home/ssm-user/work
VENV=/opt/dlami/nvme/work/swift_venv
OUT=/opt/dlami/nvme/ssm-out/diag3_overfit8
mkdir -p "${OUT}"

export PYTHONPATH=/home/ubuntu/go_viral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true

MAX_PIXELS=200704 \
VIDEO_MAX_PIXELS=200704 \
FPS_MAX_FRAMES=32 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=1 \
"${VENV}/bin/python" -m swift.cli.main sft \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --tuner_type lora \
    --lora_rank 32 --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit true --freeze_aligner true \
    --torch_dtype bfloat16 \
    --dataset /home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8.jsonl \
    --max_length 8192 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --learning_rate 5e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 5 \
    --save_steps 100 --save_total_limit 1 \
    --output_dir "${OUT}" \
    --dataloader_num_workers 1 \
    > "${OUT}/sft.log" 2>&1
echo "TRAIN_DONE"

# Pick checkpoint
CKPT=$(ls -d "${OUT}"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "ckpt: ${CKPT}"

# Inference on those same 8 ads
TMP_JSONL=/tmp/diag3_infer.jsonl
CUDA_VISIBLE_DEVICES=1 \
  FPS_MAX_FRAMES=32 FPS=1.0 \
  MAX_PIXELS=200704 VIDEO_MAX_PIXELS=200704 \
  VIDEO_MAX_TOKEN_NUM=8192 \
  "${VENV}/bin/python" -m swift.cli.main infer \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --adapters "${CKPT}" \
    --infer_backend vllm \
    --val_dataset /home/ssm-user/work/data/ttcc_swift_v2cot/overfit_8_eval.jsonl \
    --max_new_tokens 1024 --temperature 0.0 --top_p 1.0 \
    --result_path "${TMP_JSONL}" --max_pixels 200704 \
  > "${OUT}/infer.log" 2>&1
echo "INFER_DONE"
cp "${TMP_JSONL}" "${OUT}/infer_results.jsonl"
