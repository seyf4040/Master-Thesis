# LoRA — Balanced Joint Adapter (FHS:Reddit-FR = 1:1)

**Phase:** 2 | **ID:** P2-E4 | **Status:** ✅ Complete
**Date:** 2026-04-20 | **Script:** `code/phase2_lora/finetune_lora.py` (balanced config)
**Results dir:** `results/phase2_eval/{french_hate_superset,reddit_fr}/lora_joint_balanced/` + `results/phase2_eval/lora_full_french_joint_balanced/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Models | LG-1B, SG-2b |
| Training data | FHS subsampled to `len(reddit_train)` ≈ 3,665 samples + full Reddit-FR train → ~7,330 total |
| Corpus ratio | FHS:Reddit-FR = 1:1 (was 3.5:1 in unbalanced joint) |
| FHS val/test | Untouched → fair FHS evaluation preserved |
| Seed | 42 (identical to all prior experiments → identical test sets → direct 4-way comparison) |
| Best epoch | **1** |
| Adapter paths | `lora_adapters/llama_guard_1b/french_joint_balanced/best/` |
|               | `lora_adapters/shieldgemma_2b/french_joint_balanced/best/` |

## Hypothesis

> The unbalanced joint adapter underperforms on Reddit-FR because the FHS training corpus is 3.5× larger, causing FHS patterns to dominate gradient updates and diluting the Reddit-FR signal. Subsampling FHS to match Reddit-FR size should recover Reddit-FR performance while retaining the FHS recall gains.

## 4-Way Comparison: Baseline → Single → Joint → Balanced Joint

### Reddit-FR (held-out test set, n=1,023)

| Metric | LG-1B base | LG-1B single | LG-1B joint | LG-1B balanced | SG-2b base | SG-2b single | SG-2b joint | SG-2b balanced |
|--------|:----------:|:------------:|:-----------:|:--------------:|:----------:|:------------:|:-----------:|:--------------:|
| **F1** | 0.417 | 0.513 | 0.557 | 0.551 | 0.335 | **0.662** | 0.632 | **0.611** |
| Precision | 0.624 | 0.650 | 0.665 | 0.670 | 0.855 | 0.706 | 0.756 | 0.751 |
| Recall | 0.313 | 0.424 | 0.479 | 0.468 | 0.208 | 0.623 | 0.543 | 0.514 |
| TNR | 0.851 | 0.820 | 0.809 | 0.818 | 0.972 | 0.795 | 0.862 | 0.865 |

### French Hate Superset (held-out test set, n=3,614)

| Metric | LG-1B base | LG-1B single | LG-1B joint | LG-1B balanced | SG-2b base | SG-2b single | SG-2b joint | SG-2b balanced |
|--------|:----------:|:------------:|:-----------:|:--------------:|:----------:|:------------:|:-----------:|:--------------:|
| **F1** | 0.364 | 0.561 | 0.587 | 0.530 | 0.413 | 0.534 | **0.633** | 0.585 |
| Precision | 0.277 | 0.636 | 0.642 | 0.596 | 0.405 | 0.758 | 0.657 | 0.537 |
| Recall | 0.531 | 0.502 | 0.540 | 0.478 | 0.420 | 0.413 | 0.611 | **0.643** |
| TNR | 0.576 | 0.912 | 0.908 | 0.901 | 0.811 | 0.960 | 0.902 | 0.830 |

## English Generalisation (8-dataset, uncontaminated)

| Dataset | LG-1B joint | LG-1B balanced | SG-2b joint | SG-2b balanced |
|---------|:-----------:|:--------------:|:-----------:|:--------------:|
| HateCheck-EN | 0.855 | **0.840** | **0.839** | 0.837 |
| Reddit-EN | 0.579 | **0.585** | 0.540 | 0.544 |
| ToxiGen | 0.636 | 0.659 | 0.675 | **0.706** |
| OpenAI | **0.663** | 0.641 | **0.687** | 0.662 |
| Civil Comments | **0.238** | 0.234 | **0.306** | 0.273 |

SG-2b balanced ToxiGen (0.706) is the best ToxiGen result across all Phase 2 adapters — the balanced training's more aggressive recall style transfers to implicit toxicity detection. No catastrophic English forgetting.

## Hypothesis Verdict: ❌ Balancing does not help

| Expectation | Outcome |
|---|---|
| Reddit-FR F1 recovers toward 0.662 | **No** — drops 0.632 → 0.611 (−0.021) for SG-2b |
| FHS F1 stays above single LoRA (0.534) | **Barely** — 0.585 (+0.051 vs single) but worse than unbalanced joint (0.633) |
| Simultaneous F1 > 0.62 on both | **No** — balanced gets 0.611 / 0.585 simultaneously |

**Root cause:** The hypothesis assumed equal data weighting would restore Reddit-FR signal. In practice, removing 3.5× of FHS training data reduces FHS capacity without meaningfully improving Reddit-FR — the two domains compete for model capacity regardless of corpus balance. The unbalanced joint already provides a better Reddit-FR vs FHS trade-off.

**Note on SG-2b balanced FHS recall (0.643 > 0.611 joint):** The balanced training makes FHS predictions more aggressive (higher recall, lower precision 0.537 vs 0.657). This is an artifact of equal weighting — Reddit-FR's balanced class distribution (45/55) biases the model toward higher recall across both domains.

## Conclusion

Balancing the FHS:Reddit-FR corpus ratio at 1:1 hurts both domains simultaneously (SG-2b: Reddit-FR 0.632→0.611, FHS 0.633→0.585). The hypothesis is disproven — the SG-2b × Reddit-FR single LoRA adapter (F1=0.662) remains the best Tier 2 model. Phase 2 is complete.

## Cross-references

- Motivated by: [P2-E3 (joint adapter, corpus imbalance hypothesis)](exp_lora_joint.md)
- Final verdict: [P2-E2 (Reddit-FR adapter)](exp_lora_reddit_fr.md) confirmed as Tier 2
