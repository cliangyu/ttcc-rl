#!/bin/bash
source /home/ssm-user/work/venv/bin/activate
python <<'PY'
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.dataset_info("liangyuch/ttcc-cot")
    print("dataset exists. siblings:")
    for s in info.siblings[:30]:
        print(" ", s.rfilename)
except Exception as e:
    print("info err:", e)
PY
