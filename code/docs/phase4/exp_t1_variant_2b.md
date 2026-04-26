# T1 Variant 2b — Soft Labels (ε=0.05)

**Phase:** 4 | **ID:** P4-E4 | **Status:** ✅ Complete
**Date:** 2026-04-19 | **Script:** `code/phase4_two_tier/finetune_detoxify_tier1.py`
**Results dir:** `results/tier1_finetuned_soft/` + `results/tier1_comparison_honest/2b_soft/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Change from Track B | Hard labels {0.0, 1.0} → soft labels {0.05, 0.95} (ε=0.05) |
| Epochs | 5 (best = epoch 2, val_loss=0.2058) |
| Training data | Reddit-FR: 4,148 train / 460 val / 511 test (same split) |
| Checkpoint | `results/tier1_finetuned_soft/best/` |

## Training Log

| Epoch | Train loss | Val loss |
|------:|:----------:|:--------:|
| 1 | 0.1928 | 0.2213 |
| **2 (best)** | **0.1674** | **0.2058** |
| 3 | 0.1440 | 0.2063 |
| 4 | 0.1223 | 0.2079 |
| 5 | 0.1072 | 0.2102 |

## Single-Threshold Best F1 (honest 511-sample holdout)

| Dataset | 2b (soft) | 2a (hard) | Track B (leaky†) |
|---------|:---------:|:---------:|:----------------:|
| HateCheck-FR | 0.726 | 0.690 | 0.722 |
| FR-Hate Superset | 0.396 | 0.391 | 0.385 |
| Reddit-FR (honest) | 0.619 | 0.615 | 0.704†* |

## Reddit-FR Two-Threshold Operating Points (honest 511-sample holdout)

| Operating point | T_low | T_high | Deferral | T1 FNR | **T1 FPR** |
|----------------|:-----:|:------:|:--------:|:------:|:----------:|
| Low deferral   | 0.50  | 0.95   | 11.0%    | 32.0%  | **13.5%**  |
| Mid deferral   | 0.00  | 0.95   | 22.5%    | 28.3%  | **13.5%**  |
| High deferral  | 0.00  | 0.95   | 22.5%    | 28.3%  | **13.5%**  |

**T1_FPR drops from 23.8% → 13.5%** — soft labels reduce false positives on safe content. Mid-deferral target now achieves 22.5% (vs 2a's collapsed 16.4%) — partial score distribution improvement. T_high still 0.95 (bimodal not resolved). T1_FNR similar to 2a (28.3% vs 28.5%).

## Conclusion

Soft labels reduce T1_FPR from 23.8%→13.5% — label smoothing prevents the model from over-confidently predicting "safe" for safe content. However, T1_FNR stays similar to 2a (28.3%) and T_high remains at 0.95 (bimodal not fixed). The effect is asymmetric: soft labels reduce false positives but do not improve recall. Synthetic data (2c) is needed to break the bimodal structure itself.

## Cross-references

- Motivated by: [P4-E2 (Track B, bimodal collapse)](exp_t1_finetuned_base.md)
- Compared with: [P4-E3 (2a)](exp_t1_variant_2a.md), [P4-E5 (2c)](exp_t1_variant_2c.md), [P4-E6 (2d)](exp_t1_variant_2d.md)
