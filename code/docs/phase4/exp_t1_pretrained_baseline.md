# T1 Pretrained Baseline — unitary/multilingual-toxic-xlm-roberta (Track A)

**Phase:** 4 | **ID:** P4-E1 | **Status:** ✅ Complete
**Date:** 2026-04-16 | **Script:** `code/phase4_two_tier/analyze_threshold_tier1.py`
**Results dir:** `results/tier1_comparison/unitary_multilingual-toxic-xlm-roberta/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `unitary/multilingual-toxic-xlm-roberta` via HuggingFace `pipeline("text-classification")` |
| Datasets | HC-FR (3,718), FR-Hate Superset (18,071), Reddit-FR (5,119) |
| Analysis | Single-threshold sweep + two-threshold (T_low, T_high) operating points |

## Single-Threshold Sweep

| Dataset | Default T=0.5 F1 | Best T | Best F1 | Δ vs default | TPR | TNR |
|---------|:----------------:|:------:|:-------:|:------------:|:---:|:---:|
| HateCheck-FR | 0.634 | 0.00 | 0.823 | +0.189 | 1.000 | 0.000 |
| FR-Hate Superset | 0.315 | 0.02 | 0.412 | +0.097 | 0.819 | 0.318 |
| Reddit-FR | 0.366 | 0.00 | 0.616 | +0.250 | 1.000 | 0.000 |

HC-FR and Reddit-FR best F1 achieved at T=0.00 (trivial all-positive predictor). The model cannot usefully separate classes on Reddit-FR at any single threshold.

## Two-Threshold Operating Points

### HateCheck-FR

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR |
|----------------|:-----:|:------:|:--------:|:------:|:------:|
| Low deferral | 0.70 | 1.00 | 11.1% | 41.0% | 0.0% |
| Mid deferral | 0.25 | 1.00 | 26.0% | 37.6% | 0.0% |
| High deferral | 0.05 | 1.00 | 52.3% | 34.7% | 0.0% |

### French Hate Superset

*(T_high always 1.00 — no confident-unsafe bin)*

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR |
|----------------|:-----:|:------:|:--------:|:------:|:------:|
| Low deferral | 0.80 | 1.00 | 10.2% | 22.8% | 0.0% |
| Mid deferral | 0.20 | 1.00 | 26.1% | 20.1% | 0.0% |
| High deferral | 0.05 | 1.00 | 40.4% | 18.0% | 0.0% |

### Reddit-FR (Shareish proxy — most critical)

| Operating point | T_low | T_high | Deferral | **T1 FNR** | T1 FPR |
|----------------|:-----:|:------:|:--------:|:----------:|:------:|
| Low deferral | 0.70 | 1.00 | 11.1% | **41.0%** | 0.0% |
| Mid deferral | 0.25 | 1.00 | 26.0% | **37.6%** | 0.0% |
| High deferral | 0.05 | 1.00 | 52.3% | **34.7%** | 0.0% |

**T_high always 1.00 — no confident-unsafe bin on Reddit-FR.** Structurally identical to the Detoxify-M Phase 1 baseline. This is expected: `unitary/multilingual-toxic-xlm-roberta` is the same backbone that `Detoxify('multilingual')` loads internally. Track A was confirmatory, not investigative.

## Cross-model Comparison — Reddit-FR (~25% deferral)

| Model | T1 FNR | Deferral | T_low | T_high | Notes |
|-------|:------:|:--------:|:-----:|:------:|-------|
| Detoxify-M (Phase 1 baseline) | 37.0% | 27.2% | 0.20 | 1.00 | No unsafe bin |
| **Track A (pretrained unitary)** | **37.6%** | **26.0%** | **0.25** | **1.00** | **Same failure** |
| Success criterion | < 15% | ~25% | — | — | Not met |

## Conclusion

Track A confirms that the Phase 1 Detoxify-M failure is a backbone property, not a wrapper artifact. The model's score distribution clusters near zero for informal French — T_high=1.00 always, T1_FNR 34–41% at any deferral rate. Track A adds no investigative value but establishes the pretrained-backbone baseline for the Group 1 end-to-end comparison. Fine-tuning the backbone (Track B) is the only path forward.

## Cross-references

- Motivated by: [P1-E3 (Detoxify threshold analysis)](../phase1/exp_threshold_sensitivity.md)
- Motivates: [P4-E2 (Track B fine-tuning)](exp_t1_finetuned_base.md)
- Used in Group 1: [P4-E2](exp_t1_finetuned_base.md)
