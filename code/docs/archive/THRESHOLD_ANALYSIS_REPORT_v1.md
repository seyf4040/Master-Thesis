# Threshold Analysis — Detoxify-Multilingual (Two-Tier Architecture)

**Date:** 2026-04-16
**Script:** `code/phase1_baseline/analyze_threshold_detoxify.py`
**Results:** `results/threshold_analysis/`
**Purpose:** Find optimal T_low / T_high thresholds for a two-tier moderation system where:
- score < T_low  → Tier 1 passes as **safe** (high confidence)
- score > T_high → Tier 1 flags as **unsafe** (high confidence)
- T_low ≤ score ≤ T_high → **deferred to Tier 2** (uncertain)

---

## Datasets

| Dataset | n | Hateful | Safe |
|---------|--:|:-------:|:----:|
| HateCheck-FR | 3,718 | 2,600 (69.9%) | 1,118 (30.1%) |
| French Hate Superset | 18,071 | 4,340 (24.0%) | 13,731 (76.0%) |
| Reddit-FR | 5,119 | 2,280 (44.5%) | 2,839 (55.5%) |

---

## Single-Threshold Sweep

Optimal single threshold per dataset, compared to the default T=0.50.

| Dataset | Default T=0.50 F1 | Best T | Best F1 | Δ F1 | Best TPR | Best TNR |
|---------|:-----------------:|:------:|:-------:|:----:|:--------:|:--------:|
| HateCheck-FR | 0.787 | **0.06** | **0.836** | +0.049 | 0.963 | 0.211 |
| FR-Hate Superset | 0.292 | **0.01** | **0.435** | +0.143 | 0.747 | 0.468 |
| Reddit-FR | 0.408 | **0.00** | **0.616** | +0.208 | 1.000 | 0.000 |

### Interpretation

**HC-FR:** Detoxify-M performs well at default T=0.50 (F1=0.787). The optimal threshold is T=0.06 — very low, meaning the model tends to score HC-FR hateful content relatively high but also over-fires on safe content. Lowering the threshold gains recall (0.787→0.963) at the cost of precision and TNR (0.505→0.211).

**FR-Hate Superset:** Default T=0.50 is nearly useless (F1=0.292). The best achievable F1 is 0.435 at T=0.01 — meaning the model must flag almost everything to catch hateful content. Detoxify-M cannot separate classes on this dataset at any reasonable threshold.

**Reddit-FR:** The optimal threshold is T=0.00 (flag everything, F1=0.616). This is not a classifier — it is a trivial all-positive predictor. Detoxify-M assigns low toxicity scores to most Reddit-FR content regardless of label. No single threshold produces a useful classifier here.

---

## Two-Threshold Analysis (Operating Points)

For each dataset, three operating points representing different deferral budgets.

> **T1_FNR** = P(hateful | score < T_low) — hateful content leaking through as safe
> **T1_FPR** = P(safe | score > T_high) — safe content wrongly flagged as unsafe

### HateCheck-FR

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR | Coverage | n_safe | n_unsafe | n_defer |
|----------------|:-----:|:------:|:--------:|:------:|:------:|:--------:|:------:|:--------:|:-------:|
| Low deferral   | 0.00  | 0.10   | 11.6%    | 0.0%   | 25.1%  | 88.4%    | 0      | 3,288    | 430     |
| Mid deferral   | 0.00  | 0.45   | 27.5%    | 0.0%   | 22.1%  | 72.5%    | 0      | 2,694    | 1,024   |
| High deferral  | 0.00  | 0.85   | 53.7%    | 0.0%   | 18.0%  | 46.3%    | 0      | 1,723    | 1,995   |

**Key finding:** T_low is always 0.00 — Detoxify-M never assigns near-zero scores to HC-FR content (even safe content scores above 0). The "safe" bin is empty. Tier 1 can only flag; it cannot pass. T1_FNR = 0% is therefore trivial (nothing in the safe bin). T1_FPR of 18–25% is significant: a quarter of safe HC-FR content is flagged as unsafe at the low deferral setting.

### French Hate Superset

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR | Coverage | n_safe | n_unsafe | n_defer |
|----------------|:-----:|:------:|:--------:|:------:|:------:|:--------:|:------:|:--------:|:-------:|
| Low deferral   | 0.80  | 1.00   | 10.2%    | 22.8%  | 0.0%   | 89.8%    | 16,222 | 0        | 1,849   |
| Mid deferral   | 0.20  | 1.00   | 26.1%    | 20.1%  | 0.0%   | 73.9%    | 13,360 | 0        | 4,711   |
| High deferral  | 0.05  | 1.00   | 40.4%    | 18.0%  | 0.0%   | 59.6%    | 10,777 | 0        | 7,294   |

**Key finding:** T_high is always 1.00 — Detoxify-M almost never assigns a score near 1.0 on FHS content. The "unsafe" bin is empty. Tier 1 can only pass; it cannot flag. T1_FPR = 0% is trivial. T1_FNR of 18–23% is the real cost: even in the confident-safe bin, ~1 in 5 items is actually hateful. Lowering T_low (deferring more to Tier 2) reduces FNR only slightly.

### Reddit-FR

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR | Coverage | n_safe | n_unsafe | n_defer |
|----------------|:-----:|:------:|:--------:|:------:|:------:|:--------:|:------:|:--------:|:-------:|
| Low deferral   | 0.80  | 1.00   | 11.0%    | 40.9%  | 0.0%   | 89.0%    | 4,557  | 0        | 562     |
| Mid deferral   | 0.20  | 1.00   | 27.2%    | 37.0%  | 0.0%   | 72.8%    | 3,725  | 0        | 1,394   |
| High deferral  | 0.05  | 1.00   | 42.9%    | 34.3%  | 0.0%   | 57.1%    | 2,921  | 0        | 2,198   |

**Key finding:** Same asymmetry as FHS — T_high=1.00 always, no confident-unsafe bin. T1_FNR of 34–41% is alarming: even at 43% deferral, more than a third of content in the "safe" bin is actually hateful. Detoxify-M is not a reliable Tier 1 gatekeeper for Reddit-FR content at any threshold.

---

## Structural Asymmetry: What Detoxify-M Actually Does

The operating points reveal a consistent pattern across datasets:

| Dataset | Tier 1 can confidently... | T1 FNR risk |
|---------|--------------------------|-------------|
| HC-FR | **Flag** (not pass) | Low — safe bin empty |
| FHS | **Pass** (not flag) | High — 18–23% hateful in safe bin |
| Reddit-FR | **Pass** (not flag) | Very high — 34–41% hateful in safe bin |

Detoxify-M scores for French content cluster near 0 (FHS, Reddit-FR) or are spread across [0,1] without clean separation (HC-FR). The model's toxicity score is poorly calibrated for informal French hate speech.

---

## Implications for Two-Tier Architecture

**The two-tier design must be reconsidered for French informal content.**

The original assumption was that Detoxify-M would produce a bimodal score distribution — low scores for safe content, high scores for hateful content — with a manageable "uncertain" middle zone sent to Tier 2. This holds partially for HC-FR (formal, controlled test cases) but **fails for FHS and Reddit-FR** (the realistic Shareish proxies):

- On FHS and Reddit-FR, Detoxify-M scores nearly all content below 0.5. There is no high-confidence-unsafe bin.
- The confident-safe bin (low scores) contains 18–41% actually-hateful content depending on dataset — too high for a safety-critical application.

**Viable two-tier design for Shareish:**

Rather than using the score as a three-zone filter (safe / defer / unsafe), the most defensible approach is:

1. **Detoxify-M as a recall-maximising pre-filter** with a very low threshold (T≈0.05–0.10):
   - Everything below T_low → confident safe, Tier 1 passes (FNR ≈ 18–34% — still high)
   - Everything above → sent to Tier 2 (SG-2b Reddit-FR LoRA)
   - This is a binary gate, not a three-way split

2. **Accept that Detoxify-M cannot produce confident-unsafe verdicts on French content.** The "flag directly" pathway must either be removed or reserved for very high thresholds that effectively never trigger.

3. **SG-2b Reddit-FR LoRA as primary classifier**, with Detoxify-M only used to fast-pass the lowest-scoring items to save compute. At T_low=0.05 on Reddit-FR: 57% of content passes through Tier 1 (FNR=34%), 43% goes to Tier 2.

**Recommended operating point for Shareish** (Reddit-FR distribution):

| Setting | T_low | T_high | Deferral to T2 | T1 FNR |
|---------|:-----:|:------:|:--------------:|:------:|
| Aggressive (low compute cost) | 0.05 | 1.00 | 43% | 34.3% |
| Conservative (higher safety) | 0.20 | 1.00 | 27% | 37.0% |

> Note: Even the conservative setting misses 37% of hateful content in the safe bin. This is the fundamental limitation of Detoxify-M on informal French. The two-tier architecture reduces Tier 2 compute load but does not achieve near-zero FNR at Tier 1.

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig_score_distributions.png` | Safe vs hateful score histograms — shows class separation per dataset |
| `fig_threshold_sweep.png` | F1/P/R/TNR vs threshold — single threshold optimisation |
| `fig_two_threshold_hatecheck_fr.png` | Heatmaps: deferral rate, T1 FNR, T1 FPR for HC-FR |
| `fig_two_threshold_french_hate_superset.png` | Same for FHS |
| `fig_two_threshold_reddit_fr.png` | Same for Reddit-FR |
| `raw_scores.json` | Raw (score, label) pairs — rerun plots without re-running model |
| `threshold_analysis.json` | Full sweep data + operating points |
