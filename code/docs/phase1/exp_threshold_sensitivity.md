# Threshold Sensitivity — Detoxify-Multilingual Two-Tier Analysis

**Phase:** 1 | **ID:** P1-E3 | **Status:** ✅ Complete
**Date:** 2026-04-16 | **Script:** `code/phase1_baseline/analyze_threshold_detoxify.py`
**Results dir:** `results/threshold_analysis/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `unitary/multilingual-toxic-xlm-roberta` (Detoxify-multilingual backbone) |
| Datasets | HateCheck-FR (3,718 samples), FR-Hate Superset (18,071), Reddit-FR (5,119) |
| Analysis | Single-threshold sweep + two-threshold (T_low, T_high) operating points |
| Three-zone logic | score < T_low → Tier 1 safe; score > T_high → Tier 1 unsafe; middle → deferred to Tier 2 |

## Dataset Statistics

| Dataset | n | Hateful | Safe |
|---------|--:|:-------:|:----:|
| HateCheck-FR | 3,718 | 2,600 (69.9%) | 1,118 (30.1%) |
| French Hate Superset | 18,071 | 4,340 (24.0%) | 13,731 (76.0%) |
| Reddit-FR | 5,119 | 2,280 (44.5%) | 2,839 (55.5%) |

## Single-Threshold Sweep

| Dataset | Default T=0.5 F1 | Best T | Best F1 | Δ F1 | Best TPR | Best TNR |
|---------|:-----------------:|:------:|:-------:|:----:|:--------:|:--------:|
| HateCheck-FR | 0.787 | **0.06** | **0.836** | +0.049 | 0.963 | 0.211 |
| FR-Hate Superset | 0.292 | **0.01** | **0.435** | +0.143 | 0.747 | 0.468 |
| Reddit-FR | 0.408 | **0.00** | **0.616** | +0.208 | 1.000 | 0.000 |

**Interpretation:**
- **HC-FR:** Optimal T=0.06 — model tends to score HC-FR content relatively high; lowering threshold gains recall (0.963) at cost of precision. Default T=0.5 is already reasonable (0.787).
- **FR-Hate Superset:** Default T=0.5 nearly useless (F1=0.292). Best achievable 0.435 at T=0.01 — model must flag almost everything. Cannot separate classes.
- **Reddit-FR:** Optimal T=0.00 (flag everything, F1=0.616) — this is a trivial all-positive predictor, not a classifier. Detoxify-M assigns low scores to all Reddit-FR content regardless of label.

## Two-Threshold Operating Points

> T1_FNR = P(hateful | score < T_low) — hateful content leaking through as safe (safety-critical)
> T1_FPR = P(safe | score > T_high) — safe content wrongly flagged as unsafe

### HateCheck-FR

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR | n_safe | n_unsafe | n_defer |
|----------------|:-----:|:------:|:--------:|:------:|:------:|:------:|:--------:|:-------:|
| Low deferral   | 0.00  | 0.10   | 11.6%    | 0.0%   | 25.1%  | 0      | 3,288    | 430     |
| Mid deferral   | 0.00  | 0.45   | 27.5%    | 0.0%   | 22.1%  | 0      | 2,694    | 1,024   |
| High deferral  | 0.00  | 0.85   | 53.7%    | 0.0%   | 18.0%  | 0      | 1,723    | 1,995   |

**Key finding:** T_low is always 0.00 — Detoxify-M never assigns near-zero scores to HC-FR content (even safe content scores above 0). The "safe" bin is empty. Tier 1 can only flag; it cannot pass. T1_FNR=0% is trivial (nothing in the safe bin). T1_FPR of 18–25% is significant.

### French Hate Superset

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR | n_safe | n_unsafe | n_defer |
|----------------|:-----:|:------:|:--------:|:------:|:------:|:------:|:--------:|:-------:|
| Low deferral   | 0.80  | 1.00   | 10.2%    | 22.8%  | 0.0%   | 16,222 | 0        | 1,849   |
| Mid deferral   | 0.20  | 1.00   | 26.1%    | 20.1%  | 0.0%   | 13,360 | 0        | 4,711   |
| High deferral  | 0.05  | 1.00   | 40.4%    | 18.0%  | 0.0%   | 10,777 | 0        | 7,294   |

**Key finding:** T_high is always 1.00 — Detoxify-M almost never assigns a score near 1.0 on FHS content. The "unsafe" bin is empty. Tier 1 can only pass; it cannot flag. T1_FNR of 18–23%: even in the confident-safe bin, ~1 in 5 items is actually hateful.

### Reddit-FR (Shareish proxy — most critical)

| Operating point | T_low | T_high | Deferral | **T1 FNR** | T1 FPR | n_safe | n_unsafe | n_defer |
|----------------|:-----:|:------:|:--------:|:----------:|:------:|:------:|:--------:|:-------:|
| Low deferral   | 0.80  | 1.00   | 11.0%    | **40.9%**  | 0.0%   | 4,557  | 0        | 562     |
| Mid deferral   | 0.20  | 1.00   | 27.2%    | **37.0%**  | 0.0%   | 3,725  | 0        | 1,394   |
| High deferral  | 0.05  | 1.00   | 42.9%    | **34.3%**  | 0.0%   | 2,921  | 0        | 2,198   |

**Key finding:** Same asymmetry as FHS — T_high=1.00 always, no confident-unsafe bin. T1_FNR of 34–41% is alarming: even at 43% deferral, more than a third of content in the "safe" bin is actually hateful.

## Structural Asymmetry

| Dataset | Tier 1 can confidently... | T1 FNR risk |
|---------|--------------------------|-------------|
| HC-FR | **Flag** (not pass) — safe bin empty | Low |
| FHS | **Pass** (not flag) — unsafe bin empty | High — 18–23% hateful in safe bin |
| Reddit-FR | **Pass** (not flag) — unsafe bin empty | Very high — 34–41% hateful in safe bin |

Detoxify-M scores cluster near 0 (FHS, Reddit-FR) or spread without clean separation (HC-FR). The toxicity score is poorly calibrated for informal French hate speech.

## Implications for Two-Tier Architecture

The original assumption of a bimodal score distribution (low = safe, high = hateful) fails for FHS and Reddit-FR. The viable design is:
1. **Use a very low T_low** (T≈0.05–0.10): everything below → confident safe, everything above → Tier 2. Binary gate, not three-way split.
2. **Accept that Detoxify-M cannot produce confident-unsafe verdicts on French content.** The "flag directly" path must be removed.
3. **Recommended operating point (Reddit-FR):**

| Setting | T_low | T_high | Deferral to T2 | T1 FNR |
|---------|:-----:|:------:|:--------------:|:------:|
| Aggressive (low compute cost) | 0.05 | 1.00 | 43% | 34.3% |
| Conservative (higher safety) | 0.20 | 1.00 | 27% | 37.0% |

> Even the conservative setting misses 37% of hateful content in the safe bin. This is the fundamental limitation of Detoxify-M on informal French — and the direct motivation for Phase 4 (fine-tuning the backbone).

## Figures

| Figure | Description |
|--------|-------------|
| `fig_score_distributions.png` | Safe vs hateful score histograms — class separation per dataset |
| `fig_threshold_sweep.png` | F1/P/R/TNR vs threshold |
| `fig_two_threshold_hatecheck_fr.png` | Heatmaps: deferral rate, T1 FNR, T1 FPR for HC-FR |
| `fig_two_threshold_french_hate_superset.png` | Same for FHS |
| `fig_two_threshold_reddit_fr.png` | Same for Reddit-FR |
| `raw_scores.json` | Raw (score, label) pairs — rerun plots without re-running model |
| `threshold_analysis.json` | Full sweep data + operating points |

## Cross-references

- Motivated by: [P1-E1 (full baseline)](exp_full_baseline.md)
- Motivates: [P4-E1 (Track A)](../phase4/exp_t1_pretrained_baseline.md), [P4-E2 (Track B fine-tuning)](../phase4/exp_t1_finetuned_base.md)
