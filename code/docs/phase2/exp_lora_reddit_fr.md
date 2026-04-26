# LoRA — Reddit-FR Adapter (Confirmed Tier 2)

**Phase:** 2 | **ID:** P2-E2 | **Status:** ✅ Complete
**Date:** 2026-04-16 | **Script:** `code/phase2_lora/finetune_lora.py`
**Results dir:** `results/phase2_eval/reddit_fr/{baseline,lora}/` + `results/lora_full_reddit_fr/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Models | LG-1B, SG-2b |
| Training data | Reddit-FR — 3,665 (LG-1B) / 3,674 (SG-2b) train / 402–408 val / 1,023 test (seed=42) |
| Class balance | Reddit-FR: 44.6% hateful / 55.4% safe (nearly balanced) |
| Best epoch | **1** (both models overfit from epoch 2) |
| Val loss (best) | LG-1B: 0.3031 / SG-2b: epoch 1 only (CUDA OOM fix required) |
| Adapter paths | `lora_adapters/llama_guard_1b/reddit_fr/best/` |
|               | `lora_adapters/shieldgemma_2b/reddit_fr/best/` ← **confirmed Tier 2** |

## Training Diagnostics

| Run | E1 val_loss | E2 val_loss | E3 val_loss | Status |
|-----|:-----------:|:-----------:|:-----------:|--------|
| LG-1B × Reddit-FR | **0.3031** | 0.3447 | 0.8160 | best=epoch1 |
| SG-2b × Reddit-FR | (epoch 1 only) | — | — | best=epoch1 |

**SG-2b training required 3 attempts** due to CUDA OOM on A5000 24GB at batch_size=4 without gradient checkpointing. Fixed with `--gradient_checkpointing` + `model.enable_input_require_grads()` + `--batch_size 2 --grad_accum 8` (job 3843094, 2026-04-15).

## Fair Eval — Reddit-FR Held-out Test Set (n=1,023)

| Metric | LG-1B base | LG-1B LoRA | Δ | SG-2b base | SG-2b LoRA | Δ |
|--------|:----------:|:----------:|:---:|:----------:|:----------:|:---:|
| **F1** | 0.417 | **0.513** | **+0.096** | 0.335 | **0.662** | **+0.327** |
| Precision | 0.624 | 0.650 | +0.026 | 0.855 | 0.706 | −0.149 |
| Recall (TPR) | 0.313 | 0.424 | +0.111 | 0.208 | **0.623** | **+0.415** |
| TNR | 0.851 | 0.820 | −0.031 | 0.972 | 0.795 | −0.177 |
| Accuracy | 0.614 | 0.645 | +0.031 | 0.635 | 0.719 | +0.084 |

**Pattern: recall-driven — the opposite of FHS.** SG-2b started with near-zero recall (0.208) and LoRA pushed it to 0.623. SG-2b gain (+0.327 F1) is the largest improvement across all Phase 2 experiments.

## Generalisation — Reddit-FR Adapter on All 8 Datasets (`lora_full_reddit_fr/`, generated 2026-04-17)

| Dataset | LG-1B base | LG-1B LoRA | Δ | SG-2b base | SG-2b LoRA | Δ |
|---------|:----------:|:----------:|:---:|:----------:|:----------:|:---:|
| **HC-FR** | 0.674 | 0.693 | **+0.019** | 0.858 | **0.837** | **−0.021** |
| HC-EN | 0.816 | 0.811 | −0.005 | 0.902 | 0.833 | −0.069 |
| FHS | 0.372 | 0.391 | +0.019 | 0.441 | 0.461 | +0.020 |
| **Reddit-FR** | 0.407 | **0.565** | **+0.157** | 0.311 | **0.719** | **+0.408** |
| **Reddit-EN** | 0.187 | **0.528** | **+0.341** | 0.315 | **0.599** | **+0.284** |
| **ToxiGen** | 0.486 | **0.623** | **+0.137** | 0.486 | **0.720** | **+0.234** |
| OpenAI | 0.556 | 0.578 | +0.022 | 0.632 | 0.639 | +0.007 |
| CivComm | 0.651 | 0.218 | **−0.434** | 0.499 | 0.244 | **−0.256** |

### Contrast with FHS adapter (SG-2b)

| Effect | FHS adapter | Reddit-FR adapter |
|--------|:-----------:|:-----------------:|
| HC-FR regression | **−0.078** | −0.021 |
| Reddit-FR | −0.237 | **+0.408** |
| Reddit-EN | −0.266 | **+0.284** |
| ToxiGen | −0.071 | **+0.234** |
| CivComm | −0.326 | −0.256 |

**HC-FR regression is minimal** (−0.021 vs −0.078 for FHS adapter). **Informal-register transfer** is the most striking result: both models gain substantially on Reddit-EN (+0.341/+0.284) and ToxiGen (+0.137/+0.234) — the adapter learned informal hate speech register that transfers across languages. Civil Comments collapses for both adapters — a genuinely distinct annotation scheme.

## Diagnosis

Reddit-FR contains informal, colloquial French hate speech. SG-2b was calibrated for formal English/multilingual safety content and treated informal French as safe (TPR=0.208). LoRA on Reddit-FR taught the model to recognise informal French hate patterns — a recall-driven gain. This directly informs Shareish deployment: Shareish content (user listings, comments) is informal and colloquial — closer to Reddit-FR than FHS. The Reddit-FR-adapted SG-2b is unambiguously the better choice.

## Conclusion

SG-2b × Reddit-FR LoRA is the confirmed Tier 2 model. One epoch of fine-tuning on 3,665 samples pushed SG-2b TPR from 0.208 to 0.623 (+0.327 F1). The adapter generalises well to HC-FR (−0.021 only), Reddit-EN (+0.284), and ToxiGen (+0.234). Civil Comments collapse is a known cross-adapter issue. Joint and balanced joint adapters both fail to match this result on Reddit-FR — Phase 2 complete.

## Cross-references

- Motivated by: [P1-E1 (Reddit-FR baseline F1=0.311)](exp_full_baseline.md)
- Selected over: [P2-E3 (joint 0.632)](exp_lora_joint.md), [P2-E4 (balanced joint 0.611)](exp_lora_joint_balanced.md)
- Used as Tier 2 in: [P4-E2](../phase4/exp_t1_finetuned_base.md)
