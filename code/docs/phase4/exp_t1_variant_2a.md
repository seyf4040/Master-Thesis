# T1 Variant 2a — Extended Epochs (10 epochs, hard labels)

**Phase:** 4 | **ID:** P4-E3 | **Status:** ✅ Complete
**Date:** 2026-04-19 | **Script:** `code/phase4_two_tier/finetune_detoxify_tier1.py`
**Results dir:** `results/tier1_finetuned_e10/` + `results/tier1_comparison_honest/2a_e10/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Change from Track B | 3 epochs → 10 epochs |
| Labels | Hard {0.0, 1.0} (same as Track B) |
| Training data | Reddit-FR: 4,148 train / 460 val / 511 test (same split) |
| Best epoch | **2** (val_loss=0.2083; overfits from epoch 3) |
| Checkpoint | `results/tier1_finetuned_e10/best/` |

## Training Log

| Epoch | Train loss | Val loss |
|------:|:----------:|:--------:|
| 1 | 0.2366 | 0.2207 |
| **2 (best)** | **0.2159** | **0.2083** |
| 3 | 0.1851 | 0.2083↑ |
| 4 | 0.1581 | 0.2173↑ |
| 5 | 0.1323 | 0.2259↑ |
| 6–10 | … | 0.254↑ |

## Single-Threshold Best F1 (honest 511-sample holdout)

| Dataset | 2a (10 epochs) | Track B 3ep (leaky†) |
|---------|:-------------:|:--------------------:|
| HateCheck-FR | 0.690 | 0.722 |
| FR-Hate Superset | 0.391 | 0.385 |
| Reddit-FR (honest) | 0.615 | 0.704*† |

†Track B 3ep Reddit-FR F1=0.704 was leakage-inflated. Honest comparison: Group 1 combined F1≈0.626.

## Reddit-FR Two-Threshold Operating Points (honest 511-sample holdout)

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR |
|----------------|:-----:|:------:|:--------:|:------:|:------:|
| Low deferral   | 0.20  | 0.95   | 11.0%    | 29.9%  | 23.8%  |
| Mid deferral   | 0.00  | 0.95   | 16.4%    | 28.5%  | 23.8%  |
| High deferral  | 0.00  | 0.95   | 16.4%    | 28.5%  | 23.8%  |

T_high still 0.95. Mid and high deferral collapse to same point (16.4%). Bimodal pattern persists.

## Conclusion

Running 10 epochs does not fix the bimodal collapse — it only confirms that epoch 2 is the optimal checkpoint across all conditions. The slight T1_FNR improvement (41.7%→28.5%) comes from epoch 2 being better calibrated than epoch 3, not from extra training. The model overfits from epoch 3 onward (val_loss increases monotonically). Future runs should use `--epochs 2`. Soft labels (2b) or synthetic data (2c) are needed to address the bimodal structure itself.

## Cross-references

- Motivated by: [P4-E2 (Track B, bimodal collapse)](exp_t1_finetuned_base.md)
- Compared with: [P4-E4 (2b)](exp_t1_variant_2b.md), [P4-E5 (2c)](exp_t1_variant_2c.md), [P4-E6 (2d)](exp_t1_variant_2d.md)
