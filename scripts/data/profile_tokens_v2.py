#!/usr/bin/env python
"""Profile actual encoded token counts for all TTCC ads at audit settings.

Uses swift's actual SftArguments.get_template + processor — same encode path as training.
Avoids loading model weights (processor only).
"""
import os, sys, json, time, traceback
sys.path.insert(0, '/home/ubuntu/go_viral')

# Mirror sft_v2cot_full.sh env exactly
os.environ['MAX_PIXELS'] = '200704'
os.environ['VIDEO_MAX_PIXELS'] = '200704'
os.environ['FPS_MAX_FRAMES'] = '60'
os.environ['FPS'] = '1.0'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '8192'
os.environ['ENABLE_AUDIO_OUTPUT'] = 'False'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'

import torch
from swift.arguments.sft_args import SftArguments

# Construct args same as training command would (but with giant max_length to disable filtering)
print(f"[{time.strftime('%H:%M:%S')}] building SftArguments...")
args = SftArguments(
    model='/home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B',
    template='qwen2_5_omni',
    dataset=['/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.jsonl'],
    val_dataset=['/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_val.jsonl'],
    max_length=999999,           # huge so nothing is filtered
    truncation_strategy='delete',
    lazy_tokenize=True,
    tuner_type='full',
    freeze_vit=True,
    freeze_aligner=True,
    torch_dtype='bfloat16',
    output_dir='/tmp/profile_tokens_out',
    attn_impl=None,              # don't need FA for encoding
    deepspeed=None,
)

print(f"[{time.strftime('%H:%M:%S')}] loading model+processor on CPU (needed for swift template; model is offloaded)...")
args.model_kwargs = {'device_map': 'cpu', 'dtype': torch.bfloat16}

# swift's get_template needs processor.model_info which only gets attached during get_model_processor
# Load model on CPU (slow but works; we won't actually use it for forward, just for processor.model_info)
model, processor = args.get_model_processor()
print(f"[{time.strftime('%H:%M:%S')}] processor + model_info ready")

print(f"[{time.strftime('%H:%M:%S')}] building template via args.get_template(processor)...")
template = args.get_template(processor)
print(f"[{time.strftime('%H:%M:%S')}] template ready: {type(template).__name__}")
print(f"  template.max_length = {template.max_length}")

# Load JSONL rows directly
def read_jsonl(path):
    return [json.loads(line) for line in open(path)]

train_rows = read_jsonl('/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.jsonl')
val_rows = read_jsonl('/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_val.jsonl')
print(f"[{time.strftime('%H:%M:%S')}] loaded train={len(train_rows)}, val={len(val_rows)}")

# Set template into training mode so it produces full encoded outputs
template.set_mode('train')

results = []
err_count = 0
err_examples = []
t0 = time.time()

for split_name, rows in [('train', train_rows), ('val', val_rows)]:
    for i, row in enumerate(rows):
        try:
            encoded = template.encode(row, return_length=True)
            length = encoded.get('length', len(encoded.get('input_ids', [])))
            results.append((split_name, row.get('ad_id', f'row{i}'), length))
        except Exception as e:
            err_count += 1
            if len(err_examples) < 3:
                err_examples.append((split_name, row.get('ad_id', '?'), f'{type(e).__name__}: {str(e)[:150]}'))
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            print(f"[{time.strftime('%H:%M:%S')}] {split_name} {i+1}/{len(rows)}  rate={rate:.1f}/s  elapsed={elapsed:.0f}s")

elapsed_total = time.time() - t0
print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {elapsed_total/60:.1f} min")
print(f"  successful: {len(results)}")
print(f"  failed: {err_count}")
for s, aid, e in err_examples:
    print(f"    [{s}/{aid}] {e}")

if not results:
    sys.exit(1)

lengths = sorted(l for (_, _, l) in results)
n = len(lengths)
def pct(p): return lengths[min(int(p * n), n-1)]

print(f"\n=== TOKEN-LENGTH DISTRIBUTION (n={n}) ===")
print(f"  min:  {lengths[0]}")
print(f"  p10:  {pct(0.10)}")
print(f"  p25:  {pct(0.25)}")
print(f"  p50:  {pct(0.50)}")
print(f"  p75:  {pct(0.75)}")
print(f"  p90:  {pct(0.90)}")
print(f"  p95:  {pct(0.95)}")
print(f"  p99:  {pct(0.99)}")
print(f"  max:  {lengths[-1]}")
print(f"  mean: {sum(lengths)/n:.0f}")

print(f"\n=== DROP COUNT AT max_length ===")
print(f"  max_length  | drop | drop %")
for ml in [8192, 9216, 10240, 11264, 12288, 13312, 14336, 16384, 20480, 24576, 32768]:
    drops = sum(1 for l in lengths if l > ml)
    print(f"  {ml:11d} | {drops:4d} | {100*drops/n:5.1f}%")

# Save raw + the longest 20 rows for inspection
with open('/tmp/token_profile.txt', 'w') as f:
    f.write(f"# Token length profile — {time.strftime('%F %T')}\n")
    f.write(f"# FPS_MAX_FRAMES=60, MAX_PIXELS=200704, VIDEO_MAX_TOKEN_NUM=8192\n")
    f.write(f"# n_train={len(train_rows)}, n_val={len(val_rows)}, errors={err_count}\n")
    f.write(f"# distribution: min={lengths[0]}, p50={pct(0.5)}, p99={pct(0.99)}, max={lengths[-1]}\n\n")
    f.write("split\tad_id\tlength\n")
    for s, aid, l in sorted(results, key=lambda r: -r[2]):
        f.write(f"{s}\t{aid}\t{l}\n")

print(f"\nDetail at /tmp/token_profile.txt (sorted longest → shortest)")
