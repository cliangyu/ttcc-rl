#!/usr/bin/env bash
# Benchmark harness: launches sft_v2cot_full.sh as a 5-step micro-run
# and captures step time + peak GPU memory.
#
# Usage:
#   LIGER=1 ZERO=zero3 VIT_CKPT=true BS=1 GA=8 LABEL=T1b bash bench_sft.sh
#
# Env knobs (all optional, sensible defaults):
#   LIGER=0|1            --use_liger_kernel
#   ZERO=zero3|zero2     --deepspeed
#   VIT_CKPT=true|false  --vit_gradient_checkpointing
#   BS=1|2               per_device_train_batch_size
#   GA=8|4               gradient_accumulation_steps (BS*GA*2 GPUs = effective batch)
#   COMPILE=0|1          --torch_compile (optional)
#   GROUP_BY_LEN=true|false  --group_by_length (T4 enables it to tame torch.compile recompile churn)
#   NUM_WORKERS=N        --dataloader_num_workers (default 1)
#   PERSIST_WORKERS=true|false  --dataloader_persistent_workers (default false)
#   PREFETCH=N           --dataloader_prefetch_factor (default 2 from HF when workers>0)
#   LABEL=T1b            tag for the output dir / log file

set -euo pipefail
LABEL="${LABEL:-bench}"
LIGER="${LIGER:-0}"
ZERO="${ZERO:-zero3}"
VIT_CKPT="${VIT_CKPT:-true}"
BS="${BS:-1}"
GA="${GA:-8}"
COMPILE="${COMPILE:-0}"
GROUP_BY_LEN="${GROUP_BY_LEN:-false}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PERSIST_WORKERS="${PERSIST_WORKERS:-false}"
PREFETCH="${PREFETCH:-}"

WORK="${WORK:-/home/ssm-user/work}"
VENV="${VENV:-/opt/dlami/nvme/work/swift_venv}"
SFT_DATA="${SFT_DATA:-${WORK}/data/ttcc_swift_v2cot/ttcc_train_sft.jsonl}"
OUT="/opt/dlami/nvme/ssm-out/bench_${LABEL}_$(date +%H%M%S)"
mkdir -p "${OUT}"

export PYTHONPATH="/home/ubuntu/go_viral:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export ENABLE_AUDIO_OUTPUT="False"
# Disable wandb logging for benchmark
export WANDB_MODE=offline

LIGER_FLAG=()
if [[ "${LIGER}" == "1" ]]; then
    LIGER_FLAG=(--use_liger_kernel true)
fi
COMPILE_FLAG=()
if [[ "${COMPILE}" == "1" ]]; then
    COMPILE_FLAG=(--torch_compile true)
fi

# Background memory poller (1 Hz) — captures peak GPU mem during the run
MEMLOG="${OUT}/memory.csv"
(while true; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk -v ts="$(date +%s)" '{print ts","$0}' >> "${MEMLOG}"
    sleep 1
done) &
MEM_PID=$!

# Trap to clean up the memory poller
trap "kill ${MEM_PID} 2>/dev/null; true" EXIT

PREFETCH_FLAG=()
if [[ -n "${PREFETCH}" ]]; then
    PREFETCH_FLAG=(--dataloader_prefetch_factor "${PREFETCH}")
fi

echo "=== BENCH ${LABEL} START $(date) ==="
echo "  LIGER=${LIGER}  ZERO=${ZERO}  VIT_CKPT=${VIT_CKPT}  BS=${BS}  GA=${GA}  COMPILE=${COMPILE}  GROUP_BY_LEN=${GROUP_BY_LEN}"
echo "  NUM_WORKERS=${NUM_WORKERS}  PERSIST_WORKERS=${PERSIST_WORKERS}  PREFETCH=${PREFETCH:-default}"
echo "  output: ${OUT}"

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
    --group_by_length "${GROUP_BY_LEN}" \
    --max_steps 6 \
    --per_device_train_batch_size "${BS}" \
    --gradient_accumulation_steps "${GA}" \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing "${VIT_CKPT}" \
    --learning_rate 1e-5 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --save_steps 99999 \
    --eval_steps 99999 \
    --output_dir "${OUT}" \
    --deepspeed "${ZERO}" \
    --dataloader_num_workers "${NUM_WORKERS}" \
    --dataloader_persistent_workers "${PERSIST_WORKERS}" \
    "${PREFETCH_FLAG[@]}" \
    --report_to none \
    "${LIGER_FLAG[@]}" \
    "${COMPILE_FLAG[@]}" \
    2>&1 | tee "${OUT}/bench.log"

echo "=== BENCH ${LABEL} END $(date) ==="

# Kill memory poller, summarize results
kill ${MEM_PID} 2>/dev/null || true
sleep 1

# Extract step times from log
echo ""
echo "=== STEP TIMES (from log) ==="
grep -oE "'loss':[^,]+, 'grad_norm':[^,]+.*'epoch':[^,]+" "${OUT}/bench.log" || true
# Look for swift's step time markers
grep -E "step\s+[0-9]+/|Train step|tok/s|samples/s|loss.*epoch" "${OUT}/bench.log" | tail -10

echo ""
echo "=== PEAK MEMORY (per GPU) ==="
awk -F',' '
NR>0 {
    ts=$1; gpu=$2+0; mem=$3+0
    if (mem > peak[gpu]) peak[gpu] = mem
}
END {
    for (g in peak) printf "  GPU %d peak: %d MiB (%.1f GB)\n", g, peak[g], peak[g]/1024
}' "${MEMLOG}" | sort
