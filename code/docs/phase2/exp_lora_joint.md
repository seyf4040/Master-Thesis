# LoRA — Joint Adapter (FHS + Reddit-FR)

**Phase:** 2 | **ID:** P2-E3 | **Status:** ✅ Complete
**Date:** 2026-04-18 | **Script:** `code/phase2_lora/finetune_lora.py` (joint config)
**Results dir:** `results/phase2_eval/{french_hate_superset,reddit_fr}/lora_joint/` + `results/phase2_eval/lora_full_french_joint/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Models | LG-1B, SG-2b |
| Training data | FHS + Reddit-FR combined — LG-1B: 16,677 train / SG-2b: 16,686 train |
| Corpus ratio | FHS:Reddit-FR ≈ 3.5:1 (unbalanced — same as original datasets) |
| Split | Same seed=42 as single-dataset splits → identical test sets → direct 3-way comparison |
| Best epoch | **1** |
| Val loss | LG-1B: 0.2115 / SG-2b: 0.2078 |
| Adapter paths | `lora_adapters/llama_guard_1b/french_joint/best/` |
|               | `lora_adapters/shieldgemma_2b/french_joint/best/` |

## 3-Way Comparison: Baseline → Single LoRA → Joint LoRA

### French Hate Superset (held-out test set, n=3,614)

| Metric | LG-1B base | LG-1B single | LG-1B joint | SG-2b base | SG-2b single | SG-2b joint |
|--------|:----------:|:------------:|:-----------:|:----------:|:------------:|:-----------:|
| **F1** | 0.364 | 0.561 | **0.596** | 0.413 | 0.534 | **0.633** |
| Precision | 0.277 | 0.636 | 0.662 | 0.405 | 0.758 | 0.657 |
| Recall (TPR) | 0.531 | 0.502 | 0.541 | 0.420 | 0.413 | **0.611** |
| TNR | 0.576 | 0.912 | 0.915 | 0.811 | 0.960 | 0.902 |

**Pattern:** Joint LoRA substantially improves over single FHS LoRA on FHS — and the mechanism changes. Single FHS LoRA was purely precision-driven (recall flat at ~0.41). Joint LoRA adds recall (SG-2b 0.611, +0.198 vs single) while maintaining high precision. The Reddit-FR training data taught the model to recall more hate speech.

### Reddit-FR (held-out test set, n=1,023)

| Metric | LG-1B base | LG-1B single | LG-1B joint | SG-2b base | SG-2b single | SG-2b joint |
|--------|:----------:|:------------:|:-----------:|:----------:|:------------:|:-----------:|
| **F1** | 0.417 | 0.513 | **0.573** | 0.335 | **0.662** | 0.632 |
| Precision | 0.624 | 0.650 | 0.674 | 0.855 | 0.706 | 0.756 |
| Recall (TPR) | 0.313 | 0.424 | 0.499 | 0.208 | 0.623 | 0.543 |
| TNR | 0.851 | 0.820 | 0.814 | 0.972 | 0.795 | 0.862 |

**Pattern:** LG-1B joint (+0.060) improves over single Reddit-FR LoRA. SG-2b joint (0.632) slightly trails single Reddit-FR LoRA (0.662, −0.030). The FHS corpus is 3.5× larger — Reddit-FR signal is diluted, reducing the recall surge that made SG-2b × Reddit-FR the standout result.

## English Generalisation (8-dataset eval, uncontaminated)

| Dataset | LG-1B joint | SG-2b joint |
|---------|:-----------:|:-----------:|
| HateCheck-EN | 0.855 | 0.839 |
| Reddit-EN | 0.579 | 0.540 |
| ToxiGen | 0.636 | 0.675 |
| OpenAI | 0.663 | 0.687 |
| Civil Comments | 0.238 | 0.306 |

No catastrophic English forgetting — HC-EN F1 ≥ 0.839. Civil Comments collapses (same as single adapters).

## Summary: Does Joint Training Solve Domain-Locking?

| Goal | Achieved? | Note |
|------|:---------:|------|
| Retain FHS competency | ✅ Exceeds single FHS LoRA | SG-2b +0.099 |
| Retain Reddit-FR competency | ✅ Near single Reddit-FR LoRA | SG-2b −0.030 |
| Best Reddit-FR F1 overall | ❌ | Single Reddit-FR LoRA 0.662 > Joint 0.632 |

## Conclusion

The joint adapter partially solves domain-locking: it avoids the catastrophic FHS-adapter collapse on Reddit-FR and substantially improves FHS recall (adding the recall mechanism from Reddit-FR training). However, it does not surpass the single Reddit-FR adapter on Reddit-FR (0.632 < 0.662) because the 3.5:1 FHS:Reddit-FR corpus imbalance dilutes the Reddit-FR signal. The corpus imbalance is the likely root cause — tested in P2-E4 (balanced joint).

## Cross-references

- Motivated by: [P2-E1 (FHS adapter)](exp_lora_fhs.md), [P2-E2 (Reddit-FR adapter)](exp_lora_reddit_fr.md)
- Followed by: [P2-E4 (balanced joint)](exp_lora_joint_balanced.md)
