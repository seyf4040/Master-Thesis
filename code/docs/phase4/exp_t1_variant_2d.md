# T1 Variant 2d — Soft Labels + Synthetic Data (Combined)

**Phase:** 4 | **ID:** P4-E6 | **Status:** ✅ Complete
**Date:** 2026-04-19 | **Script:** `code/phase4_two_tier/finetune_detoxify_tier1.py`
**Results dir:** `results/tier1_finetuned_2d/` + `results/tier1_comparison_honest/2d_combined/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Change from 2c | Add soft labels ε=0.05 on top of 2c (Reddit-FR + 1,215 synthetic) |
| Change from 2b | Add 1,215 synthetic items on top of 2b |
| Training data | 5,363 total (same as 2c) |
| Label smoothing | ε=0.05, targets {0.05, 0.95} |
| Epochs | 5 (best = epoch 2, val_loss=0.2025) |
| Checkpoint | `results/tier1_finetuned_2d/best/` |

## Training Log

| Epoch | Train loss | Val loss |
|------:|:----------:|:--------:|
| **1** | 0.1820 | 0.2047 |
| **2 (best)** | 0.1463 | **0.2025** |
| (further) | — | — |

Best val_loss=0.2025 falls between 2b (0.2058) and 2c (0.1912) — consistent with smoothed targets reducing effective loss magnitude without providing additional signal.

## Single-Threshold Best F1 (honest 511-sample holdout)

| Dataset | 2d (combined) | 2c (synthetic) | 2b (soft) | Δ vs 2c |
|---------|:-------------:|:--------------:|:---------:|:-------:|
| HateCheck-FR | **0.820** | 0.816 | 0.726 | +0.004 |
| FR-Hate Superset | 0.411 | 0.415 | 0.396 | −0.004 |
| Reddit-FR (honest) | 0.634 | **0.668** | 0.619 | **−0.034** |

## Reddit-FR Two-Threshold Operating Points (honest 511-sample holdout)

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR |
|----------------|:-----:|:------:|:--------:|:------:|:------:|
| Low deferral   | 0.15  | 0.80   | 9.8%     | 28.8%  | 20.5%  |
| Mid deferral   | 0.00  | 0.80   | 15.1%    | 27.1%  | 20.5%  |
| High deferral  | 0.00  | 0.80   | 15.1%    | 27.1%  | 20.5%  |

T_high=0.80 is maintained (confirming synthetic data, not soft labels, drives the distribution spread).

## Full Group 2 Comparison — Reddit-FR, Honest 511-Sample Holdout

| Model | Best F1 | T1_FNR (mid) | Deferral | T_high | T1_FPR | Best? |
|-------|:-------:|:------------:|:--------:|:------:|:------:|:-----:|
| 2a — 10 epochs | 0.615 | 28.5% | 16.4% | 0.95 | 23.8% | ✗ |
| 2b — soft labels | 0.619 | 28.3% | 22.5% | 0.95 | **13.5%** | ✗ |
| **2c — synthetic** | **0.668** | **25.2%** | 12.3% | **0.80** | 18.5% | **✓** |
| 2d — soft + synthetic | 0.634 | 27.1% | 15.1% | **0.80** | 20.5% | ✗ |

## Conclusion

Combining soft labels and synthetic data does not stack their benefits — it hurts the primary metric (Reddit-FR F1: 0.668→0.634) without recovering the FPR reduction from 2b (T1_FPR stays at 20.5% vs 2b's 13.5%). Explanation: synthetic data already acts as a natural regulariser by the same mechanism as label smoothing (data diversity prevents overconfident extreme predictions), so stacking both strategies is redundant. The smoothed targets interfere with the score calibration signal established by the synthetic data. T_high=0.80 is maintained (confirming it is driven by data diversity, not target modification). HC-FR marginally improves (0.820 vs 0.816) but within noise. 2c is the final Tier 1 model.

## Cross-references

- Motivated by: [P4-E4 (2b)](exp_t1_variant_2b.md) + [P4-E5 (2c)](exp_t1_variant_2c.md)
- Outcome: [P4-E5 (2c)](exp_t1_variant_2c.md) confirmed as final Tier 1 — Group 2 complete
