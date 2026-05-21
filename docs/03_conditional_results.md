# Novelty-conditional IBS — where content-awareness actually matters

The headline "mean IBS over 87 ads" hides what's really going on. Decomposing
the test set by **per-ad B1-deviation** (how far the truth deviates from the
train-mean curve) reveals three regimes:

- **Q1** — ads whose curves are closest to the train mean. B1 trivially nails
  these; trained methods over-think.
- **Q2+Q3** — moderately-novel ads. **This is where content-awareness pays.**
- **Q4** — extremely deviant ads. Nobody predicts the direction of deviation
  correctly; everyone falls back to producing something near the mean.

Per-ad novelty
$$
\nu_i = \frac{1}{T_i + 1} \sum_{t=0}^{T_i} \big(R_{B_1}(t) - R_i(t)\big)^2.
$$
Quartiles taken on the test set's empirical distribution.

## Headline ($n = 87$, mean IBS, paired BCa vs B1)

| subset | $n$ | B1 IBS | SFT | GRPO | RLOO | SFT-ext270 |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 87 | 0.0083 | 0.0094 ($\Delta=+0.0011$, tied) | 0.0083 ($\Delta=-0.0001$, tied) | 0.0091 (tied) | 0.0093 (tied) |
| **Q1 (closest to B1)** | 22 | **0.0012** | 0.0045 ($\Delta=+0.0034$ ✓ worse) | 0.0032 ($\Delta=+0.0020$ ✓ worse) | 0.0043 (✓ worse) | 0.0053 (✓ worse) |
| **Q2+Q3 (middle)** | 43 | 0.0052 | 0.0035 ($\Delta=\mathbf{-0.0016}$ ✓) | **0.0030** ($\Delta=\mathbf{-0.0022}$ ✓) | 0.0033 (tied) | 0.0038 (tied) |
| **Q4 (farthest)** | 22 | 0.0217 | 0.0257 (tied) | 0.0236 (tied) | 0.0251 (tied) | 0.0242 (tied) |

✓ = paired BCa 95% CI excludes 0 ($B = 10{,}000$).

## Reading the table

- **Q1: methods lose to B1.** When an ad's curve is already close to the
  training mean, predicting that mean is the right move. The trained models
  insist on being content-specific and pay a penalty. SFT is $3.8\times$ worse than
  B1 on Q1 — a big cost.
- **Q2+Q3: methods beat B1.** Both SFT and GRPO have BCa CIs that exclude 0.
  GRPO has the largest effect ($\Delta = -0.0022$ vs B1). This is the regime where
  watching the video and emitting a content-specific curve actually pays off.
- **Q4: nobody wins.** The very-novel ads are too hard — methods try to
  predict the deviation but get the direction wrong, while B1 plays it safe.
- **All-ads mean hides the story.** When you average across regimes, the
  Q1 losses cancel the Q2+Q3 wins, producing the "tied with B1" headline
  that suggests methods aren't doing anything.

## GRPO vs SFT on Q2+Q3

Paired BCa: $\Delta_{\text{GRPO} - \text{SFT}} = -0.0005$, CI $[-0.0017,\,+0.0002]$ (tied; CI just barely
contains 0). On the full 87-ad test, base GRPO-50 cleanly beat SFT
($\Delta = -0.0011$, CI excludes 0). On Q2+Q3 specifically the directional advantage
holds but isn't significant at $n = 43$.

## Practical takeaway

This shape suggests a two-stage hybrid would dominate:

1. Train a *gate* that classifies ads by predicted novelty (e.g. logistic
   regression on the SFT's confidence + simple video features).
2. Predict B1 for Q1-classified ads; predict GRPO output for Q2+Q3+Q4.

Naive upper bound: if we knew the true quartile and routed perfectly, the
hybrid IBS would be
$$
\overline{\text{IBS}}_{\text{hybrid}} = \frac{n_{Q_1}}{n}\,\text{IBS}_{B_1, Q_1} + \frac{n_{Q_2 \cup Q_3}}{n}\,\text{IBS}_{\text{GRPO}, Q_2 \cup Q_3} + \frac{n_{Q_4}}{n}\,\text{IBS}_{B_1, Q_4} \approx 0.0069
$$
— a $\sim 17\%$ improvement over B1's $0.0083$.

This is left as an open experiment (option 4 in the next-steps menu).
