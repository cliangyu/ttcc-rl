"""H1: compare v38 ckpt-50 vs ckpt-80 in curve space (MSE, R[1] corr, mode-collapse std)."""
import json, re, numpy as np

def parse_curve(resp):
    if not resp: return None
    m = re.search(r'Curve:\s*(\{.*?\})', resp, re.DOTALL) or re.search(r'\{"R":\s*\[[^\]]+\]\}', resp)
    if not m: return None
    try:
        return json.loads(m.group(1) if m.lastindex else m.group(0))['R']
    except Exception: return None

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
            n = min(len(Rt), len(rp))
            pairs.append((aid, T, Rt[:n], np.array(rp[:n])))
    return pairs

def metrics(pairs, name):
    if not pairs:
        print(f'[{name}] EMPTY')
        return
    mse_per_ad = [np.mean((p - t)**2) for _, _, t, p in pairs]
    r1_t = [t[1] for _,_,t,p in pairs if len(t) > 1 and len(p) > 1]
    r1_p = [p[1] for _,_,t,p in pairs if len(t) > 1 and len(p) > 1]
    corr_r1 = np.corrcoef(r1_t, r1_p)[0,1] if r1_t else float('nan')
    # Mode collapse std
    pad_to = 30
    resamp = []
    for _,_,_,p in pairs:
        if len(p) >= 2:
            resamp.append(np.interp(np.linspace(0,1,pad_to), np.linspace(0,1,len(p)), p))
    std_across = np.array(resamp).std(axis=0).mean() if resamp else float('nan')
    # B1-like comparison: per-ad MSE vs train-mean
    print(f'\n[{name}] n={len(pairs)}')
    print(f'  per-ad MSE: mean={np.mean(mse_per_ad):.4f}  p50={np.median(mse_per_ad):.4f}  max={np.max(mse_per_ad):.4f}')
    print(f'  R[1] corr GT-vs-pred: {corr_r1:+.3f}')
    print(f'  std across ads (mode-collapse proxy): {std_across:.4f}')
    return dict(n=len(pairs), mse_mean=np.mean(mse_per_ad), mse_med=np.median(mse_per_ad),
                r1_corr=corr_r1, std=std_across)

print('### CKPT-50 (best CE eval) ###')
c50_train = metrics(load('/home/ubuntu/ttcc-rl/runs/v38_inference/ckpt50_preds_train.jsonl', '/home/ubuntu/ttcc-rl/runs/v38_inference/input_train.jsonl'), 'TRAIN-c50')
c50_val   = metrics(load('/home/ubuntu/ttcc-rl/runs/v38_inference/ckpt50_preds_val.jsonl',   '/home/ubuntu/ttcc-rl/runs/v38_inference/input_val.jsonl'),   'VAL  -c50')

print('\n### CKPT-80 (final, overfit) ###')
c80_train = metrics(load('/home/ubuntu/ttcc-rl/runs/v38_inference/ckpt80_preds_train.jsonl', '/home/ubuntu/ttcc-rl/runs/v38_inference/input_train.jsonl'), 'TRAIN-c80')
c80_val   = metrics(load('/home/ubuntu/ttcc-rl/runs/v38_inference/ckpt80_preds_val.jsonl',   '/home/ubuntu/ttcc-rl/runs/v38_inference/input_val.jsonl'),   'VAL  -c80')

print('\n### SIDE-BY-SIDE (val) ###')
print(f'{"metric":<25} {"ckpt-50":>12} {"ckpt-80":>12}   delta')
def cmp(k, label, fmt='{:.4f}'):
    a, b = c50_val[k], c80_val[k]
    d = a - b
    print(f'  {label:<23} {fmt.format(a):>12} {fmt.format(b):>12}   {d:+.4f}')
cmp('mse_mean', 'per-ad MSE (mean)')
cmp('mse_med',  'per-ad MSE (median)')
cmp('r1_corr',  'R[1] corr')
cmp('std',      'std across ads')

print('\n### SIDE-BY-SIDE (train) ###')
print(f'{"metric":<25} {"ckpt-50":>12} {"ckpt-80":>12}   delta')
def cmp_tr(k, label, fmt='{:.4f}'):
    a, b = c50_train[k], c80_train[k]
    d = a - b
    print(f'  {label:<23} {fmt.format(a):>12} {fmt.format(b):>12}   {d:+.4f}')
cmp_tr('mse_mean', 'per-ad MSE (mean)')
cmp_tr('mse_med',  'per-ad MSE (median)')
cmp_tr('r1_corr',  'R[1] corr')
cmp_tr('std',      'std across ads')
