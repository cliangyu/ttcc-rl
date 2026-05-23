#!/usr/bin/env python
"""Profile actual encoded token counts for all TTCC ads at audit settings.

Goal: answer "what max_length do we actually need, and how many rows would drop at each candidate?"

Settings (mirror sft_v2cot_full.sh exactly):
  MAX_PIXELS=200704, VIDEO_MAX_PIXELS=200704
  FPS_MAX_FRAMES=60, FPS=1.0
  VIDEO_MAX_TOKEN_NUM=8192
  ENABLE_AUDIO_OUTPUT=False

Output: histogram + drop-count at various max_length to /tmp/token_profile.txt
"""
import os, sys, json, time, traceback
from pathlib import Path

# Set env vars BEFORE importing swift so the processor picks them up
os.environ['MAX_PIXELS'] = '200704'
os.environ['VIDEO_MAX_PIXELS'] = '200704'
os.environ['FPS_MAX_FRAMES'] = '60'
os.environ['FPS'] = '1.0'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '8192'
os.environ['ENABLE_AUDIO_OUTPUT'] = 'False'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'

sys.path.insert(0, '/home/ubuntu/go_viral')

import torch
from swift.llm import (
    SftArguments, get_model_tokenizer, prepare_template, load_dataset
)

print(f"[{time.strftime('%H:%M:%S')}] loading model processor (no weights — quick)...")
# Minimal arg setup
args = SftArguments(
    model='/home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B',
    template='qwen2_5_omni',
    dataset=['/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.jsonl',
             '/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_val.jsonl'],
    max_length=999999,    # use giant max so NOTHING is filtered — we want raw token counts
    truncation_strategy='delete',
    lazy_tokenize=False,
    tuner_type='full',
    freeze_vit=True,
    freeze_aligner=True,
    torch_dtype='bfloat16',
    output_dir='/tmp/profile_tokens_out',
)
print(f"[{time.strftime('%H:%M:%S')}] args ok; loading processor only (skip model)...")

# Load processor + tokenizer ONLY (no model weights — much faster)
from swift.model.register import get_model_tokenizer_func
processor = None
try:
    # try the easy path — load only processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"[{time.strftime('%H:%M:%S')}] processor loaded via AutoProcessor")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] AutoProcessor failed: {e}; falling back to full load")
    args.model_kwargs = {'device_map': 'cpu', 'dtype': torch.bfloat16}
    model, processor = args.get_model_processor()
    print(f"[{time.strftime('%H:%M:%S')}] full load done")

# Build the template
print(f"[{time.strftime('%H:%M:%S')}] building template...")
template = prepare_template(args, processor=processor)
print(f"[{time.strftime('%H:%M:%S')}] template built: {type(template).__name__}")

# Load JSONL rows directly
def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

train_rows = read_jsonl('/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.jsonl')
val_rows   = read_jsonl('/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_val.jsonl')
print(f"[{time.strftime('%H:%M:%S')}] loaded train={len(train_rows)}, val={len(val_rows)}")

def encode_row(row):
    """Encode a row through swift's template and return total token count."""
    encoded = template.encode(row, return_length=True)
    return encoded.get('length', len(encoded['input_ids']))

# Profile
results = []
err_count = 0
err_examples = []
t0 = time.time()

for split_name, rows in [('train', train_rows), ('val', val_rows)]:
    for i, row in enumerate(rows):
        try:
            length = encode_row(row)
            results.append((split_name, row.get('ad_id', f'row{i}'), length))
        except Exception as e:
            err_count += 1
            if len(err_examples) < 3:
                err_examples.append((split_name, row.get('ad_id', '?'), str(e)[:200]))
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[{time.strftime('%H:%M:%S')}] {split_name} {i+1}/{len(rows)}  rate={rate:.1f}/s")

elapsed_total = time.time() - t0
print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {elapsed_total/60:.1f} min")
print(f"  successful encodes: {len(results)}")
print(f"  failed encodes: {err_count}")
if err_examples:
    print("  failure examples:")
    for split, aid, err in err_examples:
        print(f"    [{split}/{aid}] {err}")

# Histogram + summary
lengths = sorted(l for (_, _, l) in results)
n = len(lengths)
if n == 0:
    print("NO successful encodes — abort"); sys.exit(1)

def pct(p):
    return lengths[min(int(p * n), n-1)]

print(f"\n=== TOKEN-LENGTH DISTRIBUTION (n={n}) ===")
print(f"  min:    {lengths[0]}")
print(f"  p10:    {pct(0.10)}")
print(f"  p25:    {pct(0.25)}")
print(f"  p50:    {pct(0.50)}")
print(f"  p75:    {pct(0.75)}")
print(f"  p90:    {pct(0.90)}")
print(f"  p95:    {pct(0.95)}")
print(f"  p99:    {pct(0.99)}")
print(f"  max:    {lengths[-1]}")
print(f"  mean:   {sum(lengths)/n:.0f}")

print(f"\n=== DROP COUNT AT VARIOUS max_length ===")
print(f"  max_length  | drop count | drop %")
print(f"  ----------- | ---------- | -------")
for ml in [4096, 6144, 8192, 9216, 10240, 12288, 14336, 16384, 20480, 24576]:
    drops = sum(1 for l in lengths if l > ml)
    pct_drop = 100.0 * drops / n
    print(f"  {ml:11d} | {drops:10d} | {pct_drop:5.1f}%")

# Save detailed results
with open('/tmp/token_profile.txt', 'w') as f:
    f.write(f"# Token length profile — {time.strftime('%F %T')}\n")
    f.write(f"# Settings: FPS_MAX_FRAMES=60, MAX_PIXELS=200704, VIDEO_MAX_TOKEN_NUM=8192\n")
    f.write(f"# n_train={len(train_rows)}, n_val={len(val_rows)}, errors={err_count}\n\n")
    f.write("split\tad_id\tlength\n")
    for s, aid, l in results:
        f.write(f"{s}\t{aid}\t{l}\n")
print(f"\nDetail saved to /tmp/token_profile.txt")
