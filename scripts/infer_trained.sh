#!/usr/bin/env bash
# Run inference with a trained LoRA adapter on the TTCC test split, emit
# a predictions parquet, and evaluate under the docs/07 protocol.
#
# Usage:
#   bash infer_trained.sh <adapter_path> <method_tag> <out_parquet>
set -euo pipefail

ADAPTER="${1:?adapter path required}"
METHOD="${2:?method tag required}"
OUT_PARQ="${3:?out parquet path required}"

WORK="${WORK:-/home/ssm-user/work}"
VENV="${VENV:-/opt/dlami/nvme/work/swift_venv}"
TTCC_RL="${TTCC_RL:-/home/ubuntu/ttcc-rl}"
TMP_JSONL="$(mktemp /tmp/ttcc_infer_XXXX.jsonl)"

CUDA_VISIBLE_DEVICES=0,1 FPS_MAX_FRAMES=24 FPS=1.0 VIDEO_MAX_TOKEN_NUM=4096 \
"${VENV}/bin/python" -m swift.cli.main infer \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --adapters "${ADAPTER}" \
    --infer_backend vllm \
    --val_dataset "${WORK}/data/ttcc_swift/ttcc_test.jsonl" \
    --max_new_tokens 1024 --temperature 0.0 --top_p 1.0 \
    --result_path "${TMP_JSONL}" \
    --max_pixels 49152

# JSONL → parquet via canonical postprocess module
PYTHONPATH="${TTCC_RL}/src:${PYTHONPATH:-}" "${VENV}/bin/python" -m ttcc_rl.postprocess \
    "${TMP_JSONL}" \
    "${WORK}/data/ttcc_swift/ttcc_test.jsonl" \
    "${OUT_PARQ}" \
    "${METHOD}"

rm -f "${TMP_JSONL}"

# Evaluate under revised protocol
HF_HOME="${HF_HOME:-${WORK}/hf-cache}" \
PYTHONPATH="${TTCC_RL}/src:${PYTHONPATH:-}" \
    "${VENV}/bin/python" "${TTCC_RL}/scripts/eval_one.py" \
        "${OUT_PARQ}" --name "${METHOD}" --vs B1 SFT GRPO \
        --report "${OUT_PARQ%.parquet}_report.json"
