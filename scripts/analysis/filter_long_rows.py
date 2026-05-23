"""Quick token-length estimate per row (no model load).
Token model:
  video_tokens = min(FPS_MAX_FRAMES, ad_duration_sec) * 256  (at max_pixels=200704)
  audio_tokens = ad_duration_sec * 50  (Whisper-style)
  text_tokens  = sum(len(content)) / 3.5  chars-per-token avg
  total = video + audio + text + ~200 (boundary/special tokens)
Filter rows where total > THRESHOLD."""
import json, sys, os
sys.path.insert(0, '/home/ubuntu/go_viral')

INPUT = '/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.jsonl'
OUTPUT = '/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.filtered.jsonl'
THRESHOLD = 5000    # leave 692-token buffer under ml=8192

FPS_MAX = 32
PER_FRAME = 256
AUDIO_PER_SEC = 50  # estimate
CHARS_PER_TOKEN = 3.5

kept = []
dropped = []
for i, line in enumerate(open(INPUT)):
    row = json.loads(line)
    T = row.get('T', 60)
    video = min(FPS_MAX, T) * PER_FRAME
    audio = int(T * AUDIO_PER_SEC)
    text_chars = 0
    for m in row.get('messages', []):
        c = m.get('content', '')
        if isinstance(c, str):
            text_chars += len(c)
    text = int(text_chars / CHARS_PER_TOKEN)
    total = video + audio + text + 200
    if total <= THRESHOLD:
        kept.append((row, total, T))
    else:
        dropped.append((row.get('ad_id', f'row{i}'), total, T))

print(f"input rows: {i+1}")
print(f"kept: {len(kept)}  ({100*len(kept)/(i+1):.1f}%)")
print(f"dropped: {len(dropped)}")
if dropped:
    print(f"  longest dropped: top 5")
    for aid, total, T in sorted(dropped, key=lambda x: -x[1])[:5]:
        print(f"    ad_id={aid}  est_total={total}  T={T}")
print(f"\ndist of kept totals (estimated):")
import statistics
totals = [t for _,t,_ in kept]
print(f"  min={min(totals)} p50={sorted(totals)[len(totals)//2]} p90={sorted(totals)[int(len(totals)*0.9)]} max={max(totals)}")
with open(OUTPUT, 'w') as f:
    for row, _, _ in kept:
        f.write(json.dumps(row) + '\n')
print(f"\nfiltered written to {OUTPUT}")
