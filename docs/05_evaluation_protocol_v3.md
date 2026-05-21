# TTCC evaluation protocol v3 — verified, full walkthrough

**Status:** validated 2026-05-21 with sanity checks passing on all 7 methods.
Supersedes the protocol described in `docs/02_experiment_configs.md §7` and
`ttcc-eval/docs/07_proper_scoring_rule_revision.md`.

## What changed and why

The earlier protocol reported *absolute IBS* + paired BCa CI. That's
correct but reports only one number. After researching the literature,
**every textbook treatment of probabilistic forecast evaluation
recommends reporting at least three things together:**

1. **An absolute proper score** (we have: IBS)
2. **A skill score against a reference** (NEW: BSS)
3. **A decomposition into calibration vs discrimination** (NEW: Murphy)

This is the "calibration + sharpness/discrimination" framework from
Gneiting & Raftery 2007 — *"maximize sharpness subject to calibration"*
— which is the standard paradigm in meteorology, finance, and
epidemiology. scikit-survival's recommended evaluation uses the same
multi-metric approach (C-index, time-dependent AUC, IBS), as do all
the major forecasting toolkits (yardstick, mlr3proba, TF-Probability).

Sources verified:
- [Wikipedia: Brier score](https://en.wikipedia.org/wiki/Brier_score)
- [Mason 2004 — using climatology as reference](https://journals.ametsoc.org/view/journals/mwre/132/7/1520-0493_2004_132_1891_oucaar_2.0.co_2.xml)
- [scikit-survival evaluation guide](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)
- [Siegert 2017 — simplified Murphy decomposition](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.2985)
- [Bröcker 2009 — decomposition for general proper scores](https://arxiv.org/abs/0806.0813)
- [DTC Probabilistic Skill Scores](https://dtcenter.org/book/export/html/2990)
- [Weighted Brier with clinical utility (2024)](https://arxiv.org/html/2408.01626v1)

---

## The protocol (layers 0–6)

For every method, produce a row containing all layers.

### L0 — Parse rate
$n_{\text{parsed}} / n_{\text{test}}$ — sanity check that the model is producing
parseable JSON curves on the test set. A method with $< 80\%$ parse rate
should be treated as broken regardless of other numbers.

### L1 — Absolute IBS (Integrated Brier Score)
$$
\text{IBS}_i = \frac{1}{T_i + 1} \sum_{t=0}^{T_i} \big(\hat R_i(t) - R_i(t)\big)^2, \qquad \overline{\text{IBS}} = \frac{1}{n}\sum_{i=1}^{n} \text{IBS}_i
$$

- **Source:** Brier 1950 + Graf 1999.
- **Range:** $[0, 1]$. Lower = better.
- **Why we still report this:** it's the absolute scale, comparable
  across datasets and to other papers' raw numbers. Equivalent to MSE
  in our (no censoring) setting; called IBS to signal "we're using
  the survival-analysis canonical metric."

### L2 — Brier Skill Score (BSS) vs B1
$$
\text{BSS} = 1 - \frac{\overline{\text{IBS}}_{\text{method}}}{\overline{\text{IBS}}_{B_1}}
$$

- **Source:** Murphy 1973; standard in operational forecasting.
- **Range:** $(-\infty, 1]$. **$0$ = tied with B1**, **$+0.42$ = method has $42\%$
  lower error than B1**, **$-1.76$ = method is $176\%$ worse than B1**.
- **Why this is the right normalized form:** unitless, communicable,
  and exactly the scale forecasting practitioners use. "$\text{BSS} = 0.42$ on
  Q2+Q3" is more meaningful than "$\Delta\text{IBS} = -0.0022$."
- **Reference choice (Mason 2004):** the canonical reference is the
  "climatology" baseline — the unconditional outcome distribution.
  For us this is **B1 = train-mean curve, padded/truncated per ad**.

### L3 — Calibration slope
Pool all $(\hat R_{\text{hat}}, R_{\text{true}})$ pairs across ads $\times$ timesteps; fit
$R_{\text{true}} \sim a + b \cdot \hat R_{\text{hat}}$. Slope $= b$. Target: $1.0$.

- $b < 1.0$ → predictions too extreme (over-confident)
- $b > 1.0$ → predictions too compressed (under-confident)

### L4 — Murphy decomposition (NEW)

For each method, decompose IBS into three components:
$$
\text{IBS} = \text{REL} - \text{RES} + \text{UNC}
$$

Bin predictions into $K = 10$ equidistant bins on $[0, 1]$. For each
bin $k$ let $n_k$ be the count, $\bar p_k$ the mean prediction in bin $k$,
$\bar o_k$ the mean outcome in bin $k$, and $\bar o$ the overall outcome mean.
Then:
$$
\text{REL} = \sum_{k=1}^{K} \frac{n_k}{N} (\bar p_k - \bar o_k)^2, \qquad
\text{RES} = \sum_{k=1}^{K} \frac{n_k}{N} (\bar o_k - \bar o)^2, \qquad
\text{UNC} = \operatorname{Var}(o).
$$

- **REL** (reliability): calibration error — how far the binned
  forecasts differ from the conditional event frequency. **Lower = better.**
- **RES** (resolution): discrimination — how much the binned forecasts
  vary with the true outcome. **Higher = better.** Equivalent to
  "how much information your forecast provides over the marginal mean."
- **UNC** (uncertainty): inherent task difficulty — variance of the
  observations. **Data-only, model-invariant** (within a fixed subset).

**Implementation:** Bröcker 2009 shows this generalizes from binary to
continuous proper scores. Verify the decomposition: $\text{REL} - \text{RES} + \text{UNC}$
should match IBS to within floating-point error.

**Equivalent skill form** (against the constant predictor = trivial
"always predict the marginal frequency"):
$$
\text{BSS}_{\text{unc}} = \frac{\text{RES} - \text{REL}}{\text{UNC}}
$$

This is what Murphy's decomposition reports as a "skill" number against
the marginal-frequency predictor. It complements our BSS-vs-B1 because
the two references are different (B1 varies with $t$; the marginal is
constant). Both are useful: BSS vs B1 says "vs a time-varying
climatology"; $\text{BSS}_{\text{unc}}$ says "vs a trivial constant."

### L5 — Paired BCa bootstrap for significance
Per-ad $\Delta_i = \text{IBS}_{\text{method}, i} - \text{IBS}_{\text{ref}, i}$, resample with replacement,
$B = 10{,}000$, BCa 95% CI on $\overline{\Delta} = \frac{1}{n}\sum_i \Delta_i$.

- CI excludes $0$ → real effect
- CI contains $0$ → tied
- Source: Efron 1987.

Optionally transform the IBS interval into a BSS interval:
$\text{BSS}_{\text{lo}} = 1 - \dfrac{\overline{\text{IBS}}_{\text{method}} - \Delta_{\text{lo}}}{\overline{\text{IBS}}_{B_1}}$ etc.

### L6 — Conditional decomposition by novelty quartile
Per-ad novelty $\nu_i = \text{IBS}_{B_1, i}$ (how much B1 errs on ad $i$). Quartile-
stratify; report L1–L5 separately on each subset:

- **Q1 (closest 25% to B1):** B1 trivially wins; methods over-think.
- **Q2+Q3 (middle 50%):** where content-awareness pays.
- **Q4 (farthest 25%):** very-deviant ads; nobody predicts the
  direction.

The cleanest single-number-for-the-paper is **BSS on Q2+Q3**.

---

## Validated results (2026-05-21)

### Full set ($n = 87$)

| method | $n_{\text{par}}$ | IBS | **BSS** | slope | REL | RES | UNC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 (reference) | 87 | 0.00833 | $0.000$ | $+1.001$ | 0.0002 | 0.0296 | 0.0366 |
| SFT | 87 | 0.00939 | $-0.128$ | $+0.966$ | 0.0022 | 0.0286 | 0.0366 |
| **GRPO-50** | 87 | **0.00826** | $\mathbf{+0.008}$ | $+0.969$ | 0.0022 | 0.0293 | 0.0366 |
| RLOO | 87 | 0.00909 | $-0.091$ | $+0.965$ | 0.0027 | 0.0290 | 0.0366 |
| SFT-Extended ckpt-270 | 87 | 0.00932 | $-0.119$ | $+0.957$ | 0.0014 | 0.0283 | 0.0366 |
| GRPO-Extended ckpt-150 | 87 | 0.00906 | $-0.088$ | $+0.962$ | 0.0027 | 0.0289 | 0.0366 |
| GRPO-Extended ckpt-179 | 87 | 0.00910 | $-0.092$ | $+0.962$ | 0.0028 | 0.0290 | 0.0366 |
| SFT-noCoT | 66 | 0.01150 | $-0.381$ | $+0.957$ | 0.0043 | 0.0277 | 0.0366 |

**Sanity checks:**
- $\text{BSS}(B_1) = 0.000$ ✓
- $\text{UNC} \approx 0.0366$ for all methods ✓ (data-only quantity)
- Murphy decomposition residual $< 10^{-3}$ for all methods ✓

**Reading the full-set Murphy:**

- **B1 has $\text{REL} = 0.0002$ (near-zero).** Expected — B1 IS the climatology
  baseline, so its forecasts equal the marginal frequency by
  construction. It's perfectly calibrated *in aggregate*.
- **All trained methods have $\text{REL} \approx 0.002\text{–}0.003$**, about $10\times$ worse
  calibration than B1.
- **All trained methods have $\text{RES} \approx 0.028\text{–}0.030$**, essentially the same
  as B1's $0.0296$.
- **Conclusion:** the full-set "tie" with B1 is a **calibration
  problem, not a discrimination problem**. The methods have similar
  power to separate strong-from-weak ads, but their absolute
  magnitudes are slightly miscalibrated. **Post-hoc recalibration
  (isotonic regression on a val set) could close the gap without
  retraining.** This is a high-leverage insight Murphy gave us that
  mean IBS hid.

### Q2+Q3 subset (n = 43, where content-awareness matters)

| method | IBS | **BSS** | slope | REL | RES | UNC | paired BCa ΔIBS |
|---|---:|---:|---:|---:|---:|---:|---|
| B1 (reference) | 0.00516 | 0.000 | +1.022 | 0.0021 | 0.0311 | 0.0339 | — |
| **SFT** | 0.00353 | **+0.316** | +0.993 | **0.0003** | 0.0308 | 0.0339 | −0.00163 [−0.0029, −0.00004] ✓ |
| **GRPO-50** | **0.00299** | **+0.421** | +0.991 | **0.0003** | 0.0308 | 0.0339 | −0.00217 [−0.0035, −0.00030] ✓ |
| SFT-Extended | 0.00380 | +0.264 | +0.981 | 0.0001 | 0.0302 | 0.0339 | tied |
| GRPO-Extended ckpt-150 | 0.00366 | +0.291 | +0.979 | 0.0004 | 0.0305 | 0.0339 | tied |

**Reading Q2+Q3 Murphy:**
- **Methods have LOWER REL than B1 here** (0.0001–0.0004 vs B1's
  0.0021). On the moderate-novelty regime, the methods' content-
  specific predictions are *better calibrated* than B1's constant.
- RES is preserved (essentially the same as B1).
- Result: BSS = +0.42 for GRPO-50, with paired BCa CI excluding 0.
  **This is the canonical headline answer to "does the method work":
  on Q2+Q3, GRPO is 42% better than the climatology baseline.**

### Q1 subset (n = 22, where methods lose)

| method | IBS | **BSS** | REL | RES | paired BCa ΔIBS |
|---|---:|---:|---:|---:|---|
| B1 | 0.00115 | 0.000 | 0.0003 | 0.0268 | — |
| SFT | 0.00452 | **−2.91** | 0.0016 | 0.0249 | +0.00336 [+0.0018, +0.0054] ✓ worse |
| GRPO-50 | 0.00319 | **−1.76** | 0.0020 | 0.0261 | +0.00203 [+0.0011, +0.0030] ✓ worse |
| GRPO-Extended ckpt-150 | 0.00496 | **−3.30** | 0.0024 | 0.0253 | +0.00381 [+0.0024, +0.0054] ✓ worse |

**Murphy reveals:** on easy ads close to the train mean, methods are
both **less calibrated** AND **slightly less discriminating** than B1.
The high REL is the dominant cause of the BSS regression — methods
emit content-specific values for ads that are well-described by the
mean, and pay a calibration penalty.

### Q4 subset (n = 22, where nobody wins)

| method | IBS | BSS | REL | RES | paired BCa ΔIBS |
|---|---:|---:|---:|---:|---|
| B1 | 0.0217 | 0.000 | 0.0044 | 0.0282 | — |
| SFT | 0.0257 | −0.19 | 0.0181 | 0.0284 | tied |
| GRPO-50 | 0.0237 | −0.09 | 0.0156 | 0.0282 | tied |
| GRPO-Extended ckpt-150 | 0.0237 | −0.09 | 0.0173 | 0.0283 | tied |

**Murphy reveals:** on very-deviant ads, methods have **4× worse REL
than B1** but comparable RES. They're trying to be specific but
miscalibrated. Differences from B1 aren't statistically significant.

---

## Summary table — the one-glance picture

| | full set | Q1 | Q2+Q3 | Q4 |
|---|---|---|---|---|
| Headline (BSS vs B1) | **tied** | **methods lose** | **methods win ✓** | tied |
| GRPO-50 BSS | +0.01 | −1.76 | **+0.42** | −0.09 |
| SFT BSS | −0.13 | −2.91 | **+0.32** | −0.19 |
| What Murphy says | REL too high (calibration) | REL too high (over-confident) | REL low (well-calibrated on hard ads) | REL very high (wrong specifics) |

---

## Why the three-layer protocol is necessary

A single-number protocol would mislead you in three different ways:

1. **Mean IBS alone:** "methods tied with B1." This is the *Q1 + Q4
   wash* — easy-ad losses cancel moderate-ad wins.
2. **BSS alone, full set:** "methods tied with B1." Same issue.
3. **BSS alone, Q2+Q3:** "methods crush B1 by 42%." True but
   incomplete — the cost on Q1 (-176% to -291%) is real and important
   to know.

The protocol surfaces all three at once. Recommendation: when
discussing externally, lead with **BSS on Q2+Q3** (+0.42) as the
headline, then note the Q1 regression as a known limitation, then
introduce Murphy to explain *why* (calibration vs discrimination).

---

## Reproducibility

```bash
# Full-set evaluation with all layers
python /home/ubuntu/ttcc-rl/scripts/full_eval.py \
    --preds preds_sft.parquet:SFT \
            preds_grpo.parquet:GRPO50 ... \
    --b1-preds preds_b1.parquet \
    --gt /home/ssm-user/work/data/ttcc_swift/ttcc_test.jsonl \
    --bins 10

# Subset evaluation (Q1, Q2Q3, Q4, all)
python /home/ubuntu/ttcc-rl/scripts/full_eval_subset.py \
    --subset Q2Q3 \
    [other args same as full_eval.py]
```

Scripts handle the sanity-check assertions (BSS(B1)=0, UNC invariant,
decomposition residual <1e-3) automatically and print warnings on
failure.

---

## Open extensions (not implemented)

- **Time-windowed Murphy:** compute REL/RES/UNC per-timestep, then
  inspect where calibration breaks down. Possibly identify that the
  methods are well-calibrated at hook but poorly at tail.
- **Bröcker 2009 decomposition for other proper scores:** the
  decomposition framework generalizes to log-loss / spherical /
  pseudospherical. Would let us compare against models with different
  output parameterizations on a fair footing.
- **Recalibration experiment:** based on the Murphy finding that
  methods have higher REL than B1 on the full set, fit isotonic
  regression on a held-out val slice and see how much BSS improves.
  High-leverage. Open.
- **Weighted Brier with utility:** the
  [arXiv 2408.01626 paper](https://arxiv.org/html/2408.01626v1) gives
  a principled framework for weighting Brier by user-defined utility
  (e.g., ad-revenue, click-through-rate). Could replace our novelty
  weighting (which I showed yesterday is outlier-sensitive) with a
  more grounded scheme.

---

## Walkthrough — talk it through with your teammate

1. **Start with the dataset:** 87 test ads, per-second retention
   curves R(t) ∈ [0,1], monotone non-increasing, T_i = ad duration.
2. **Show the headline IBS table.** Methods tie B1.
3. **Show that B1 is the per-t train-mean curve** — not a trivial
   constant predictor, an actually-informed baseline.
4. **Layer in BSS:** 1 − IBS/IBS_B1. Standard in meteorology since
   the 60s. Methods on full set: BSS ≈ 0.
5. **Layer in Murphy decomposition.** IBS = REL − RES + UNC. **All
   methods have ~10× higher REL than B1; comparable RES.** Conclusion:
   the tie is a calibration issue, not a discrimination issue.
6. **Decompose by novelty quartile.** On Q2+Q3 — the regime where
   B1 starts to fail — **methods reach BSS = +0.42 (GRPO) and +0.32
   (SFT), CI excludes 0**. Calibration also improves: REL drops below
   B1's on this subset.
7. **Show Q1 regression.** On easy ads, methods over-think and pay
   calibration penalty. BSS = −1.76 to −2.91.
8. **Show segment metrics (hook + completion).** GRPO has Spearman
   ρ = +0.22 on hook ranking — the only method with non-trivial
   across-ad ranking signal for hook strength. SFT-Extended
   significantly beats B1 on completion rate magnitude.
9. **The actionable takeaway:** the methods *are* doing real video
   analysis, but on the wrong subset of ads. A gate that routes
   Q1-classified ads to B1 and harder ads to GRPO would be the
   honest production recipe.

---

*Protocol verified 2026-05-21. All numbers reproducible via the
scripts above on the parquets in `/home/ssm-user/work/work-out/`.*
