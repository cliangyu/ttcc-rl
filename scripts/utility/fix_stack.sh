#!/usr/bin/env bash
set -euxo pipefail
exec > /opt/dlami/nvme/fix_stack.log 2>&1
echo "[fix] $(date) start"
PIP=/opt/dlami/nvme/work/venv/bin/pip
PY=/opt/dlami/nvme/work/venv/bin/python

# Uninstall everything pip-managed in the torch/vllm/xformers stack
$PIP uninstall -y -q vllm xformers torch torchvision torchaudio || true

# Reinstall with cu128 extra-index (Unsloth Blackwell recipe). Let vLLM pull a matching torch.
$PIP install -U -q vllm --extra-index-url https://download.pytorch.org/whl/cu128

# Verify
$PY <<PYEOF
import torch
print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
print("cap=", torch.cuda.get_device_capability(0))
x = torch.randn(4, 4, device="cuda")
y = x @ x.t()
print("matmul ok:", float(y.sum()))
import vllm
print("vllm=", vllm.__version__)
import vllm._C  # the ABI symbol resolves only if ABI matches
print("vllm._C ok")
PYEOF

echo "[fix] $(date) DONE"
