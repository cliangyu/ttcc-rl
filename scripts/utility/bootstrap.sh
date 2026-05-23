#!/usr/bin/env bash
set -euxo pipefail
exec > /opt/dlami/nvme/bootstrap.log 2>&1
echo "[bootstrap] $(date) start"

WORK=/opt/dlami/nvme/work
mkdir -p "$WORK"
cd "$WORK"

# Clone inference repo (private; HTTPS works for public read after the repo is public, but ours is private — use ttcc-inference HTTPS with token via env, OR scp from local later).
# Repo is private. For now, clone via ttcc-inference fork that we will scp the code into. As a fallback, just write the code from heredocs.
# Simplest path: write a minimal inference driver inline. We DO have the parquet schema cached locally; just paste code.

# Create venv
python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install -q --upgrade pip wheel

echo "[bootstrap] installing vllm + deps..."
pip install -q "vllm==0.10.1.1"
pip install -q "qwen-omni-utils[decord]" "accelerate>=1.0" "huggingface_hub>=0.24" "hf_transfer>=0.1.6" "pyarrow>=15" "pandas>=2.1" "numpy>=1.26" "pydantic>=2.5"

echo "[bootstrap] verifying torch + cuda..."
python -c "import torch; print(torch=, torch.__version__); print(cuda_available=, torch.cuda.is_available()); print(device_count=, torch.cuda.device_count()); print(cap=, torch.cuda.get_device_capability(0)); print(name=, torch.cuda.get_device_name(0))"

echo "[bootstrap] verifying vllm..."
python -c "import vllm; print(vllm=, vllm.__version__)"
python -c "import qwen_omni_utils; print(qwen-omni-utils ok)"

echo "[bootstrap] $(date) DONE"
