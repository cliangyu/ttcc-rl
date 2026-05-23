"""Quantitative read of v38 train vs val: shape vs magnitude error,
mode collapse check, T-conditioning check."""
import json, re, numpy as np

def parse_curve(resp):
    if not resp: return None
    m = re.search(r'Curve:\s*(\{.*?\})', resp, re.DOTALL) or re.search(r'\{"R":\s*\[[^\]]+\]\}', resp)
    if not m: return None
    try:
        return json.loads(m.group(1) if m.lastindex else m.group(0))['R']
    except: return None

def load(preds, inputs):
    inp = {}
    for line in open(inputs):
        r = json.loads(line)
        aid = r.get('ad_id') or (r['videos'][0].split('/')[-1].replace('.mp4','') if r.get('videos') else None)
        inp[aid] = (r['T'], np.array(r['R_true']))
    pairs = []
    for line in open(preds):
        r = json.loads(line)
        aid = r.get('ad_id') or (r['videos'][0].split('/')[-1].replace('.mp4','') if r.get('videos') else None)
        rp = parse_curve(r.get('response',''))
        if rp and aid in inp:
            T, Rt = inp[aid]
            # align to T+1 length
            n = min(len(Rt), len(rp))
            pairs.append((aid, T, Rt[:n], np.array(rp[:n])))
    return pairs

train = load('/home/ubuntu/ttcc-rl/runs/v38_inference/ckpt80_preds_train.jsonl', '/home/ubuntu/ttcc-rl/runs/v38_inference/input_train.jsonl')
val   = load('/home/ubuntu/ttcc-rl/runs/v38_inference/ckpt80_preds_val.jsonl',   '/home/ubuntu/ttcc-rl/runs/v38_inference/input_val.jsonl')

def metrics(pairs, name):
    mse_per_ad = [np.mean((p - t)**2) for _, _, t, p in pairs]
    mae_per_ad = [np.mean(np.abs(p - t)) for _, _, t, p in pairs]
    # IBS-like: (p - t)^2 averaged over time
    print(f'\n[{name}] n={len(pairs)}')
    print(f'  per-ad MSE:  mean={np.mean(mse_per_ad):.4f}  p50={np.median(mse_per_ad):.4f}  max={np.max(mse_per_ad):.4f}')
    print(f'  per-ad MAE:  mean={np.mean(mae_per_ad):.4f}  p50={np.median(mae_per_ad):.4f}  max={np.max(mae_per_ad):.4f}')

    # Is the prediction's R[1] (second-1 drop) capturing the GT's R[1]?
    r1_t = [t[1] for _,_,t,p in pairs if len(t) > 1]
    r1_p = [p[1] for _,_,t,p in pairs if len(p) > 1]
    if r1_t:
        corr_r1 = np.corrcoef(r1_t, r1_p)[0,1]
        print(f'  R[1] (sec-1 retention) corr GT vs pred: {corr_r1:.3f}  ← shape signal')

    # Average prediction shape (is it mode-collapsed?)
    # Resample all preds to same length 30 for averaging
    pad_to = 30
    resamp = []
    for _,_,_,p in pairs:
        if len(p) >= 2:
            # Linear resample to pad_to
            x_old = np.linspace(0, 1, len(p))
            x_new = np.linspace(0, 1, pad_to)
            resamp.append(np.interp(x_new, x_old, p))
    if resamp:
        arr = np.array(resamp)
        std_across_ads = arr.std(axis=0).mean()
        print(f'  std of preds across ads (avg over time): {std_across_ads:.4f}  ← LOW means mode-collapse')

    # T-conditioning: does pred length match GT T?
    T_match = sum(1 for _,T,t,p in pairs if abs(len(p) - (T+1)) <= 1)
    print(f'  pred length matches T+1 (±1): {T_match}/{len(pairs)} ({100*T_match/len(pairs):.0f}%)')

metrics(train, 'TRAIN')
metrics(val,   'VAL  ')

# Compare: are val predictions collapsed to a single shape (mode collapse)?
def avg_pred(pairs):
    pad_to = 30
    resamp = []
    for _,_,_,p in pairs:
        if len(p) >= 2:
            x_old = np.linspace(0, 1, len(p))
            x_new = np.linspace(0, 1, pad_to)
            resamp.append(np.interp(x_new, x_old, p))
    return np.array(resamp) if resamp else None

ar_train_pred = avg_pred(train)
ar_val_pred   = avg_pred(val)
print(f'\nAvg pred at relative time positions [0, 0.25, 0.5, 0.75, 1.0]:')
def at(arr, frac): return arr.mean(axis=0)[int(frac * (arr.shape[1]-1))]
for arr, name in [(ar_train_pred, 'train'), (ar_val_pred, 'val')]:
    if arr is None: continue
    print(f'  {name}: {at(arr,0):.3f}, {at(arr,0.25):.3f}, {at(arr,0.5):.3f}, {at(arr,0.75):.3f}, {at(arr,1.0):.3f}')
