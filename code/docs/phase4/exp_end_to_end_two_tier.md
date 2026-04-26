# End-to-End Two-Tier Evaluation (2c as Tier 1)

**Phase:** 4 | **ID:** P4-E8 | **Status:** ⏳ Pending
**Date:** — | **Script:** `code/phase4_two_tier/slurm/score_two_tier_finetuned.sbatch` (2c checkpoint)
**Results dir:** `results/two_tier_scores/finetuned_2c/` (to be created)

## Configuration

| Parameter | Value |
|-----------|-------|
| Tier 1 | 2c_synthetic checkpoint (T_low=0.10, T_high=0.75, deferral=10.4%) |
| Tier 2 | SG-2b Reddit-FR LoRA |
| Eval set | 511-sample honest holdout (216 hate / 295 safe) |
| Pipeline | `score_two_tier.py` → `simulate_thresholds.py` |

## Reference — Prior End-to-End (Track B, Group 1)

| Configuration | Combined F1 | FNR | Deferral | Avg_ms |
|---|:---:|:---:|:---:|:---:|
| **Tier 2 alone** (baseline to beat) | **0.640** | 32.9% | 100% | 55.9 |
| Track B T1 + T2 (honest) | 0.626 | 44.9% | 10.4% | 12.6 |
| **Success criterion** | **> 0.640** | < 33% | **< 30%** | < 55 | 

## Expected Outcome

2c achieves F1=0.668 on the same 511-sample holdout (single-model). The end-to-end combined system may exceed Tier 2 alone (0.640) if 2c correctly routes low-confidence samples to Tier 2. The 4.4× speed advantage is a guaranteed result regardless of F1 outcome.

## Cross-references

- Tier 1 used: [P4-E5 (2c)](exp_t1_variant_2c.md)
- Tier 2 used: [P2-E2 (SG-2b Reddit-FR LoRA)](../phase2/exp_lora_reddit_fr.md)
- Reference baseline: [P4-E2 (Group 1)](exp_t1_finetuned_base.md)

## Revision (update when complete)

_Add combined F1 table, operating points, and conclusion here after running `score_two_tier_finetuned.sbatch`._
