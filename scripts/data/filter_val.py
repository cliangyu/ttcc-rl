import json, sys
sys.path.insert(0, '/home/ubuntu/go_viral')
THRESHOLD = 5000
FPS_MAX = 32
PER_FRAME = 256
AUDIO_PER_SEC = 50
CHARS_PER_TOKEN = 3.5
INPUT = '/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_val.jsonl'
kept = []; dropped = []
for line in open(INPUT):
    row = json.loads(line)
    T = row.get('T', 60)
    video = min(FPS_MAX, T) * PER_FRAME
    audio = int(T * AUDIO_PER_SEC)
    text_chars = sum(len(m.get('content','')) for m in row.get('messages',[]) if isinstance(m.get('content',''),str))
    text = int(text_chars / CHARS_PER_TOKEN)
    total = video + audio + text + 200
    if total <= THRESHOLD: kept.append(row)
    else: dropped.append(row)
print(f"val: kept {len(kept)} of {len(kept)+len(dropped)}")
with open(INPUT.replace('.jsonl','.filtered.jsonl'),'w') as f:
    for r in kept: f.write(json.dumps(r)+'\n')
