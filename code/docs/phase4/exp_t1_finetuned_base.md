# T1 Fine-tuned Base (Track B + Group 1 End-to-End Eval)

**Phase:** 4 | **ID:** P4-E2 | **Status:** ✅ Complete
**Date:** 2026-04-19 | **Scripts:** `finetune_detoxify_tier1.py` + `score_two_tier.py` + `simulate_thresholds.py`
**Results dir:** `results/tier1_detoxify_finetuned/` + `results/two_tier_scores/finetuned/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `unitary/multilingual-toxic-xlm-roberta` |
| Training data | Reddit-FR: 4,148 train / 460 val / **511 test** (seed=42, honest holdout) |
| Loss | MSE regression on `sigmoid(logit)` with hard float targets {0.0, 1.0} |
| Epochs | 3 (best = epoch 3, val_loss=0.2136, monotonically improving — no overfitting detected) |
| Hardware | A5000 (~90 s/epoch) |
| Checkpoint | `results/tier1_detoxify_finetuned/best/` |

## Track B — Training Log

| Epoch | Train loss | Val loss | Δ Val loss | Time |
|------:|:----------:|:--------:|:----------:|-----:|
| 1 | 0.2344 | 0.2171 | — | 33s |
| 2 | 0.2076 | 0.2157 | −0.0014 | 29s |
| **3 (best)** | **0.1863** | **0.2136** | **−0.0021** | 29s |

## Track B — Single-Threshold Sweep (⚠️ Leaky — 80% training data in test set)

| Dataset | Default T=0.5 F1 | Best T | Best F1 | Δ vs pretrained best F1 |
|---------|:----------------:|:------:|:-------:|:-----------------------:|
| HateCheck-FR | 0.691 | 0.00 | 0.722 | −0.101 |
| FR-Hate Superset | 0.375 | 0.00 | 0.385 | −0.027 |
| Reddit-FR | **0.662** | 0.00 | **0.704** | **+0.088** |

## Track B — Reddit-FR Two-Threshold Operating Points (⚠️ Leaky)

| Operating point | T_low | T_high | Deferral | T1 FNR | T1 FPR |
|----------------|:-----:|:------:|:--------:|:------:|:------:|
| Low deferral   | 0.00  | 0.95   | 11.4%    | **25.2%** | 15.0% |
| Mid deferral   | 0.00  | 0.95   | 11.4%    | **25.2%** | 15.0% |
| High deferral  | 0.00  | 0.95   | 11.4%    | **25.2%** | 15.0% |

All three deferral targets resolve to the same operating point — bimodal score distribution. T_high=0.95 flags 88.6% of content as "confident-unsafe". This is a qualitative shift from the pretrained model (T_high=1.00 = empty unsafe bin). **⚠️ T1_FNR=25.2% is leakage-inflated** — honest figure from Group 1 is 41.7%.

## Track B — FHS Note

FHS T1_FPR=67.4%: the fine-tuned model massively over-flags FHS safe content as "unsafe". Root cause: FHS is formal, curated hate speech — different register than Reddit-FR. Acceptable trade-off for Shareish (which resembles Reddit-FR, not FHS).

## Group 1 — Inference Timing (honest 511-sample holdout)

| Component | avg ms/sample |
|-----------|:-------------:|
| Tier 1 — pretrained (unitary XLM-R) | 7.3 |
| Tier 1 — fine-tuned (Detoxify-M) | 7.0 |
| Tier 2 — SG-2b Reddit-FR LoRA | ~55 |

Tier 1 is ~8× faster than Tier 2. At 10% deferral, combined avg ≈ 13 ms/sample — **4× speed reduction** vs Tier 2 alone.

## Group 1 — Tier 2 Alone Baseline (honest 511-sample holdout, 216 hate / 295 safe)

| Tier 2 alone | F1 | FNR | FPR | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| SG-2b Reddit-FR LoRA | **0.640** | 32.9% | 31.2% | 0.612 | 0.671 |

> Note: Slightly lower than Phase 2 fair eval figure (0.662) due to different train/test split in this 511-sample holdout.

## Group 1 — Pretrained T1 + T2 Operating Points

| Operating point | T_low | T_high | Deferral | Combined F1 | FNR | FPR | T1_FNR | Avg_ms |
|----------------|:-----:|:------:|:--------:|:-----------:|:---:|:---:|:------:|:------:|
| Low deferral   | 0.10  | 0.20   | 10.8%    | 0.558       | 50.0% | 21.4% | 47.2% | 13.3 |
| Mid deferral   | 0.05  | 0.25   | 28.6%    | 0.581       | 45.8% | 23.7% | 35.2% | 23.3 |
| High deferral  | 0.00  | 0.20   | 72.4%    | 0.639       | 29.2% | 37.3% | 0.0% | 47.7 |
| **Best F1 (grid)** | 0.00 | 0.25 | 75.5% | **0.643** | 30.1% | — | 0.0% | — |

Best combined F1 (0.643) requires 75.5% deferral — essentially routing everything to Tier 2.

## Group 1 — Fine-tuned T1 + T2 Operating Points

| Operating point | T_low | T_high | Deferral | Combined F1 | FNR | FPR | T1_FNR | Avg_ms |
|----------------|:-----:|:------:|:--------:|:-----------:|:---:|:---:|:------:|:------:|
| Low deferral   | 0.00  | 0.60   | 10.2%    | 0.626       | 44.9% | 15.3% | **41.7%** | 12.6 |
| Mid deferral   | 0.00  | 0.65   | 10.4%    | 0.626       | 44.9% | 15.3% | **41.7%** | 12.7 |
| High deferral  | 0.00  | 0.65   | 10.4%    | 0.626       | 44.9% | 15.3% | **41.7%** | 12.7 |
| **Best F1 (grid)** | 0.00 | 0.15 | 2.5% | **0.628** | 42.1% | — | — | — |

All three deferral targets collapse to the same operating point (~10.4% deferral). **Honest T1_FNR = 41.7%** (not 25.2% from Track B leaky eval).

## Group 1 — Cross-model Comparison

| Configuration | Combined F1 | FNR | Deferral | Avg_ms | vs T2 alone |
|---|:---:|:---:|:---:|:---:|:---:|
| **Tier 2 alone** (SG-2b LoRA) | **0.640** | 32.9% | 100% | 55.9 | — |
| Pretrained T1 + T2 (low deferral) | 0.558 | 50.0% | 10.8% | 13.3 | −0.082 / **4.2× faster** |
| Pretrained T1 + T2 (best F1) | 0.643 | 30.1% | 75.5% | 47.7 | +0.003 |
| Fine-tuned T1 + T2 (any deferral) | 0.626 | 44.9% | 10.4% | 12.6 | −0.014 / **4.4× faster** |
| **Success criterion** | > 0.640 | < 33% | < 30% | < 55 | — |

## Key Findings

1. **Data leakage correction:** Honest T1_FNR = 41.7% (not 25.2%). Track B threshold analysis evaluated on training data.
2. **Bimodal collapse confirmed:** Hard {0,1} MSE targets create score clusters at extremes — no tunable middle zone. Explains why all three deferral targets collapse to the same operating point.
3. **Speed advantage is real:** At 10% deferral, 4.4× faster than Tier 2 alone (12.6 ms vs 55.9 ms). Valid thesis argument.
4. **Pretrained T1 is worse than fine-tuned** at all useful deferral rates — confirms fine-tuning direction is correct.
5. **Group 2 is the critical path:** Soft labels, more epochs, synthetic data directly address the bimodal collapse.

## Conclusion

Track B fine-tuning creates a qualitative improvement: a confident-unsafe bin (T_high=0.95) now exists on Reddit-FR where the pretrained model had none. However, the honest Group 1 evaluation reveals the leakage-inflated Track B T1_FNR was misleading. The combined system does not exceed Tier 2 alone in F1. The bimodal score collapse (hard {0,1} targets) prevents tunable deferral. Group 2 experiments address this directly.

## Cross-references

- Motivated by: [P4-E1 (Track A)](exp_t1_pretrained_baseline.md)
- Group 2 experiments: [P4-E3 (2a)](exp_t1_variant_2a.md), [P4-E4 (2b)](exp_t1_variant_2b.md), [P4-E5 (2c)](exp_t1_variant_2c.md), [P4-E6 (2d)](exp_t1_variant_2d.md)
- Tier 2 used: [P2-E2 (SG-2b Reddit-FR LoRA)](../phase2/exp_lora_reddit_fr.md)
