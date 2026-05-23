#!/usr/bin/env bash
# H2: full-FT SFT on 717 train ads with the T5/T6-locked audit-ambitious recipe.
# No val tracking (no v2cot val available yet; offline curve-space eval after).
#
# Estimated wall: ~7-8 h (steady-state ~63 s/step, ~448 update steps for 10 epochs).
# Output: timestamped ckpts in /opt/dlami/nvme/ssm-out/sft_h2_<timestamp>/
# We'll select the best ckpt offline using curve-space metrics on test set.
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/opt/dlami/nvme/ssm-out/sft_h2_${STAMP}"
mkdir -p "${OUT}"

WORK="/home/ssm-user/work"
VENV="/opt/dlami/nvme/work/swift_venv"
SFT_DATA="${WORK}/data/ttcc_swift_v2cot/ttcc_train_sft.jsonl"

export PYTHONPATH="/home/ubuntu/go_viral:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export ENABLE_AUDIO_OUTPUT="False"
export WANDB_ENTITY="${WANDB_ENTITY:-liangyuch}"
export WANDB_PROJECT="${WANDB_PROJECT:-ttcc}"
export WANDB_NAME="sft_h2_${STAMP}"

EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-5}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_LIMIT="${SAVE_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"

echo "[$(date '+%F %T')] H2 START: ${WANDB_NAME}" | tee "${OUT}/h2.log"
echo "[$(date '+%F %T')] OUT: ${OUT}" | tee -a "${OUT}/h2.log"
echo "[$(date '+%F %T')] SFT_DATA: ${SFT_DATA} (717 ads, CoT)" | tee -a "${OUT}/h2.log"
echo "[$(date '+%F %T')] VAL_DATA: (none — no v2cot val; offline eval after)" | tee -a "${OUT}/h2.log"

MAX_PIXELS=200704 \
VIDEO_MAX_PIXELS=200704 \
FPS_MAX_FRAMES=60 \
VIDEO_MAX_TOKEN_NUM=16384 \
OMP_NUM_THREADS=6 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
"${VENV}/bin/python" -m swift.cli.main sft \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --dataset "${SFT_DATA}" \
    --max_length 24576 \
    --truncation_strategy delete \
    --lazy_tokenize true \
    --strict false \
    --dataset_num_proc 1 \
    --group_by_length false \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --torch_compile true \
    --learning_rate "${LR}" \
    --warmup_ratio 0.05 \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_LIMIT}" \
    --output_dir "${OUT}" \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 4 \
    --report_to tensorboard wandb \
    2>&1 | tee -a "${OUT}/h2.log"

echo "[$(date '+%F %T')] H2 END: ${WANDB_NAME}" | tee -a "${OUT}/h2.log"
