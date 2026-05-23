#!/usr/bin/env bash
set -euxo pipefail
exec > /opt/dlami/nvme/upgrade_torch.log 2>&1
echo "[upgrade] $(date) start"
PIP=/opt/dlami/nvme/work/venv/bin/pip
PY=/opt/dlami/nvme/work/venv/bin/python
# Uninstall current torch stack
$PIP uninstall -y -q torch torchvision torchaudio
# Install torch with Blackwell (sm_120) support — cu128 channel
$PIP install -q --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
$PY -c "
import torch
print(\"torch=\", torch.__version__)
print(\"cuda=\", torch.cuda.is_available())
x = torch.randn(4, 4, device=\"cuda\")
y = x @ x.t()
print(\"matmul ok:\", y.shape, \"sum=\", y.sum().item())
"
# Reinstall vLLM to pick up new torch (also need transformers + xformers to recompile)
$PIP install -q --upgrade --force-reinstall --no-deps vllm==0.10.1.1 || $PIP install -q --upgrade vllm
$PY -c "import vllm; print(\"vllm=\", vllm.__version__)"
echo "[upgrade] $(date) DONE"
