#!/usr/bin/env bash
set -euo pipefail
exec 2>&1
echo "=== H1 inference start: $(date) ==="
echo "=== ckpt-50 on TRAIN (116 ads) ==="
bash /tmp/infer_v38_ckpt50.sh /tmp/v38_input_train.jsonl /tmp/v38_ckpt50_preds_train.jsonl
echo "=== ckpt-50 on VAL (20 ads) ==="
bash /tmp/infer_v38_ckpt50.sh /tmp/v38_input_val.jsonl /tmp/v38_ckpt50_preds_val.jsonl
echo "=== H1 inference done: $(date) ==="
