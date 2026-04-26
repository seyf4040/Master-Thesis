# LoRA — French Hate Superset Adapter

**Phase:** 2 | **ID:** P2-E1 | **Status:** ✅ Complete
**Date:** 2026-04-07 | **Script:** `code/phase2_lora/finetune_lora.py`
**Results dir:** `results/phase2_eval/french_hate_superset/{baseline,lora}/` + `results/phase2_eval/lora_full/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Models | LG-1B, SG-2b |
| Training data | French Hate Superset — 13,012 train / 1,445 val / 3,614 test (seed=42, random shuffle) |
| Class balance | FHS: 24% hateful / 76% safe (3:1 safe-to-hateful ratio) |
| Best epoch | **1** (both models overfit from epoch 2) |
| Val loss (best) | LG-1B: 0.1903 / SG-2b: 0.1862 |
| Adapter paths | `lora_adapters/llama_guard_1b/french_hate_superset/best/` |
|               | `lora_adapters/shieldgemma_2b/french_hate_superset/best/` |

## Training Diagnostics

| Run | E1 val_loss | E2 val_loss | E3 val_loss | Status |
|-----|:-----------:|:-----------:|:-----------:|--------|
| LG-1B × FHS | **0.1903** | 0.2076 | 0.4630 | best=epoch1 |
| SG-2b × FHS | **0.1862** | 0.2050 | 0.3320 | best=epoch1 |

## Fair Eval — FHS Held-out Test Set (n=3,614)

| Metric | LG-1B base | LG-1B LoRA | Δ | SG-2b base | SG-2b LoRA | Δ |
|--------|:----------:|:----------:|:---:|:----------:|:----------:|:---:|
| **F1** | 0.364 | **0.561** | **+0.197** | 0.413 | **0.534** | **+0.121** |
| Precision | 0.277 | 0.636 | +0.359 | 0.405 | 0.758 | +0.353 |
| Recall (TPR) | 0.531 | 0.502 | −0.029 | 0.420 | 0.413 | −0.007 |
| TNR | 0.576 | **0.912** | **+0.336** | 0.811 | **0.960** | **+0.149** |
| Accuracy | 0.565 | 0.816 | +0.251 | 0.719 | 0.831 | +0.112 |

**Pattern: precision-driven.** Recall stays nearly flat while TNR surges. The models learned to stop false-alarming on non-hateful formal French text.

## Biased Initial Eval (DO NOT CITE — training data in test set)

| Dataset | LG-1B v3 | LG-1B LoRA (biased) | SG-2b v3 | SG-2b LoRA (biased) |
|---------|:--------:|:-------------------:|:--------:|:-------------------:|
| FR-Hate | 0.372 | 0.858 ⚠️ | 0.441 | 0.673 ⚠️ |
| Reddit-FR | 0.398 | 0.159 | 0.311 | 0.071 |

## Generalisation — FHS Adapter on All 8 Datasets (`lora_full/`)

| Dataset | LG-1B base | LG-1B LoRA (FHS) | Δ | SG-2b base | SG-2b LoRA (FHS) | Δ |
|---------|:----------:|:----------------:|:---:|:----------:|:----------------:|:---:|
| HC-FR | 0.674 | 0.650 | −0.024 | 0.858 | **0.780** | **−0.078** |
| HC-EN | 0.816 | 0.766 | −0.050 | 0.902 | 0.875 | −0.027 |
| FR-Hate (full) | 0.372 | **0.634** | **+0.262** | 0.441 | **0.627** | **+0.186** |
| **Reddit-FR (full)** | 0.407 | **0.108** | **−0.299** | 0.311 | **0.074** | **−0.237** |
| Reddit-EN | 0.187 | 0.112 | −0.075 | 0.315 | 0.049 | −0.266 |
| ToxiGen | 0.486 | 0.322 | −0.164 | 0.486 | 0.415 | −0.071 |
| OpenAI | 0.556 | 0.584 | +0.028 | 0.632 | 0.575 | −0.057 |
| CivComm | 0.651 | 0.223 | **−0.428** | 0.499 | 0.173 | **−0.326** |

**FHS adapter does NOT generalise.** Reddit-FR and Reddit-EN collapse catastrophically. Civil Comments collapses entirely (−0.428). The adapter learned formal French hate speech patterns that actively hurt performance on informal content.

## Diagnosis

FHS is a formal, clearly-labelled academic dataset. The base models had decent recall but poor precision — over-triggering on non-hateful formal French. LoRA corrected this by teaching which formal French patterns are unambiguously hateful. The precision-driven gain has a direct cost: the adapter is domain-locked. For Shareish (informal, colloquial register), the FHS adapter is worse than baseline on the most relevant dataset (Reddit-FR).

## Conclusion

FHS fine-tuning produces genuine improvements on the FHS test set (LG-1B +0.197, SG-2b +0.121) via a precision-surge mechanism. However, the adapter catastrophically collapses Reddit-FR (SG-2b −0.237) and Reddit-EN (−0.266), making it actively harmful for Shareish deployment. The FHS adapter is only useful if input is known to be formal French hate speech. For two-tier architecture purposes, the SG-2b Reddit-FR adapter is the clear choice.

## Cross-references

- Motivated by: [P1-E1 (FR-Hate baseline F1=0.441)](exp_full_baseline.md)
- Contrasted with: [P2-E2 (Reddit-FR adapter)](exp_lora_reddit_fr.md)
- Joint adapter context: [P2-E3](exp_lora_joint.md)
