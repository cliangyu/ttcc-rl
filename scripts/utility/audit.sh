#!/bin/bash
source /home/ssm-user/work/venv/bin/activate
echo "=== HF README ==="
python <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("liangyuch/ttcc-cot", "README.md", repo_type="dataset")
print(open(p).read())
PY
echo
echo "=== QC script head ==="
head -120 /home/ssm-user/work/scripts/qc_cot.py
