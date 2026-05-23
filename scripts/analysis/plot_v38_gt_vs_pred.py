#!/usr/bin/env python
"""Plot v38's predicted retention curves vs ground truth, for train + val.
Reads two inference output JSONLs and the input JSONLs, parses 'Curve: {...}'
JSON from the model response, plots GT vs predicted lines.
"""
import json, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_curve_from_response(resp):
    """Extract R list from 'Curve: {"R": [1.0, 0.33, ...]}' style response."""
    if not resp:
        return None
    m = re.search(r'Curve:\s*(\{.*?\})', resp, re.DOTALL)
    if not m:
        m = re.search(r'\{"R":\s*\[[^\]]+\]\}', resp)
        if not m:
            return None
        try:
            return json.loads(m.group(0))['R']
        except Exception:
            return None
    try:
        return json.loads(m.group(1))['R']
    except Exception:
        return None

def load_pairs(preds_jsonl, input_jsonl):
    """Load (ad_id, T, R_true, R_pred) tuples by ad_id alignment."""
    inputs = {}
    for line in open(input_jsonl):
        row = json.loads(line)
        # ad_id might be derived; v38 dataset has 'ad_id' or just 'messages'
        aid = row.get('ad_id', None)
        # Fall back: use the mp4 stem as ID
        if aid is None and 'videos' in row and row['videos']:
            aid = row['videos'][0].split('/')[-1].replace('.mp4','')
        inputs[aid] = (row['T'], row['R_true'])

    preds = {}
    for line in open(preds_jsonl):
        row = json.loads(line)
        aid = row.get('ad_id', None)
        if aid is None and 'videos' in row and row['videos']:
            aid = row['videos'][0].split('/')[-1].replace('.mp4','')
        resp = row.get('response', '')
        r_pred = parse_curve_from_response(resp)
        if r_pred is not None:
            preds[aid] = r_pred

    pairs = []
    for aid, (T, R_true) in inputs.items():
        if aid in preds:
            pairs.append((aid, T, R_true, preds[aid]))
    return pairs

train_pairs = load_pairs('/home/ubuntu/ttcc-rl/runs/v38_inference/ckpt80_preds_train.jsonl', '/home/ubuntu/ttcc-rl/runs/v38_inference/input_train.jsonl')
val_pairs   = load_pairs('/home/ubuntu/ttcc-rl/runs/v38_inference/ckpt80_preds_val.jsonl',   '/home/ubuntu/ttcc-rl/runs/v38_inference/input_val.jsonl')

print(f"train pairs (loaded): {len(train_pairs)}")
print(f"val pairs (loaded):   {len(val_pairs)}")

# Plot N ads from each side
def plot_grid(pairs, title, fname, n_show=12):
    pairs = pairs[:n_show]
    cols = 4
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*2.8), sharey=True)
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    for ax, (aid, T, R_true, R_pred) in zip(axes, pairs):
        t = np.arange(len(R_true))
        ax.plot(t, R_true, 'o-', color='C0', markersize=3, linewidth=1.5, label='GT')
        # truncate or pad pred to T+1
        t_pred = np.arange(min(len(R_pred), len(R_true)))
        ax.plot(t_pred, R_pred[:len(t_pred)], '^--', color='C3', markersize=3, linewidth=1.2, label='pred')
        ax.set_title(f'{aid[:14]}  T={T}s', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, T)
        ax.grid(alpha=0.25, linestyle=':')
        ax.tick_params(labelsize=7)
    for ax in axes[len(pairs):]:
        ax.axis('off')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print(f"saved {fname}")
    plt.close(fig)

plot_grid(train_pairs, 'v38 — TRAIN (overfit target, should be near-perfect)', '/home/ubuntu/ttcc-rl/runs/v38_inference/figures/ckpt80_train.png')
plot_grid(val_pairs,   'v38 — VAL (held-out, should diverge)', '/home/ubuntu/ttcc-rl/runs/v38_inference/figures/ckpt80_val.png')

# Combined comparison: 6 train + 6 val
fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharey=True)
selected_train = train_pairs[:6]
selected_val   = val_pairs[:6]
for i, (aid, T, R_true, R_pred) in enumerate(selected_train + selected_val):
    ax = axes[i // 4, i % 4]
    t = np.arange(len(R_true))
    ax.plot(t, R_true, 'o-', color='C0', markersize=4, linewidth=2, label='GT')
    t_pred = np.arange(min(len(R_pred), len(R_true)))
    ax.plot(t_pred, R_pred[:len(t_pred)], '^--', color='C3', markersize=4, linewidth=1.5, label='pred')
    split = 'TRAIN' if i < 6 else 'VAL'
    color = 'green' if i < 6 else 'red'
    ax.set_title(f'[{split}]  {aid[:14]}  T={T}s', fontsize=9, color=color)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(1, T))
    ax.grid(alpha=0.25, linestyle=':')
    ax.tick_params(labelsize=8)
    if i == 0: ax.legend(fontsize=9)
fig.suptitle('v38 overfit: TRAIN (top 2 rows, green) vs VAL (bottom row, red) — predicted retention vs ground truth', fontsize=12)
fig.tight_layout()
fig.savefig('/home/ubuntu/ttcc-rl/runs/v38_inference/figures/ckpt80_compare.png', dpi=120)
print('saved /home/ubuntu/ttcc-rl/runs/v38_inference/figures/ckpt80_compare.png')
