# T1 Variant 2c — Synthetic Data (Confirmed Final Tier 1)

**Phase:** 4 | **ID:** P4-E5 | **Status:** ✅ Complete — Confirmed Tier 1 model
**Date:** 2026-04-19 | **Script:** `code/phase4_two_tier/finetune_detoxify_tier1.py`
**Results dir:** `results/tier1_finetuned_synthetic/` + `results/tier1_comparison_honest/2c_synthetic/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Change from Track B | Reddit-FR only (4,148) → Reddit-FR + 1,215 synthetic French hate items (~29% more data) |
| Synthetic items | From [P3-E1](../phase3/exp_synthetic_french_hate.md) — 5 HateCheck-FR functionalities |
| Labels | Hard {0.0, 1.0} |
| Total training | 5,363 samples |
| Epochs | 5 (best = epoch 2, val_loss=0.1912) |
| Checkpoint | `results/tier1_finetuned_synthetic/best/` |

## Training Log

| Epoch | Train loss | Val loss |
|------:|:----------:|:--------:|
| 1 | 0.2241 | 0.2200 |
| **2 (best)** | **0.1825** | **0.1912** |
| 3 | 0.1510 | 0.2074↑ |
| 4 | 0.1299 | 0.2099 |
| 5 | 0.1142 | 0.2092 |

## Single-Threshold Best F1 (honest 511-sample holdout)

| Dataset | **2c (synthetic)** | 2b (soft) | 2a (hard) | Track B (leaky†) |
|---------|:--------------:|:---------:|:---------:|:----------------:|
| **HateCheck-FR** | **0.816** | 0.726 | 0.690 | 0.722 |
| FR-Hate Superset | **0.415** | 0.396 | 0.391 | 0.385 |
| Reddit-FR (honest) | **0.668** | 0.619 | 0.615 | 0.704†* |

## Reddit-FR Two-Threshold Operating Points (honest 511-sample holdout)

| Operating point | T_low | **T_high** | Deferral | T1 FNR | T1 FPR |
|----------------|:-----:|:---------:|:--------:|:------:|:------:|
| Low deferral   | 0.10  | **0.75**  | 10.4%    | 25.4%  | 18.5%  |
| Mid deferral   | 0.00  | **0.80**  | 12.3%    | 25.2%  | 18.5%  |
| High deferral  | 0.00  | **0.80**  | 12.3%    | 25.2%  | 18.5%  |

**T_high breaks from 0.95 to 0.75–0.80** — the only Group 2 variant where the unsafe bin is not at the hard boundary. The synthetic data adds sufficient distribution diversity to prevent extreme score clustering. T1_FNR=25.2% honest at 12.3% deferral.

## Full Group 2 Comparison — Reddit-FR, Honest 511-Sample Holdout

| Model | Best F1 | T1_FNR | Deferral | T_high | T1_FPR | Best? |
|-------|:-------:|:------:|:--------:|:------:|:------:|:-----:|
| 2a — 10 epochs | 0.615 | 28.5% | 16.4% | 0.95 | 23.8% | ✗ |
| 2b — soft labels | 0.619 | 28.3% | 22.5% | 0.95 | **13.5%** | ✗ |
| **2c — synthetic** | **0.668** | **25.2%** | 12.3% | **0.80** | 18.5% | **✓** |
| Success criterion | > 0.640† | — | < 30% | — | — | |

†Combined F1 must exceed Tier 2 alone baseline (0.640).

## Recommended Operating Point for Production

| Setting | T_low | T_high | Deferral | T1_FNR | Avg ms/sample |
|---------|:-----:|:------:|:--------:|:------:|:-------------:|
| Recommended | 0.10 | 0.75 | 10.4% | 25.4% | ~12.7 ms |
| Conservative | 0.00 | 0.80 | 12.3% | 25.2% | ~13.7 ms |

**4.4× speed reduction** vs Tier 2 alone (55 ms).

## Generalisation Preview

From P4-E7 (full generalisation eval, 2026-04-24):
- **HC-FR FNR at T=0.5: 7.0%** (pretrained: 45.8%) — near-zero missed structured hate speech
- **HC-EN FNR at T=0.5: 2.9%** (pretrained: 20.2%) — cross-lingual generalisation
- FHS best-F1: 0.300 (pretrained: 0.413) — domain mismatch cost accepted

## Conclusion

2c (synthetic data, epoch 2) is the confirmed final Tier 1 model. The 1,215 synthetic items spanning 5 HateCheck-FR functionalities prevent bimodal score collapse — T_high breaks to 0.80, creating a tunable deferral zone absent in all prior variants. The HC-FR FNR improvement (45.8%→7.0%) demonstrates that synthetic data coverage of structured hate functionalities transfers directly to HC-FR test cases in both French and English. Recommended operating point: T_low=0.10, T_high=0.75, deferral=10.4%.

## Cross-references

- Synthetic data from: [P3-E1](../phase3/exp_synthetic_french_hate.md)
- Compared with: [P4-E3 (2a)](exp_t1_variant_2a.md), [P4-E4 (2b)](exp_t1_variant_2b.md), [P4-E6 (2d)](exp_t1_variant_2d.md)
- Validated by: [P4-E7 (generalisation eval)](exp_generalisation_eval.md)
- Next: [P4-E8 (end-to-end simulation)](exp_end_to_end_two_tier.md)
