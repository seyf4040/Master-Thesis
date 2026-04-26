# Phase 2 — LoRA Fine-tuning Index

**Status:** ✅ Complete | **Date:** 2026-04-20
**Conclusion:** SG-2b × Reddit-FR LoRA (F1=0.662) is the confirmed Tier 2 model. Balanced joint disproves the corpus-imbalance hypothesis — single Reddit-FR LoRA remains best.

## Experiments

| ID    | Name | Status | F1 (Reddit-FR) | FNR | One-line result | File |
|-------|------|--------|:--------------:|:---:|-----------------|------|
| P2-E1 | LoRA — FHS adapter | ✅ | 0.534 (SG-2b, FHS test set) | — | Precision-driven gain; catastrophic Reddit-FR collapse (−0.237) | [exp_lora_fhs.md](exp_lora_fhs.md) |
| P2-E2 | LoRA — Reddit-FR adapter | ✅ | **0.662** (SG-2b) | — | Recall-driven +0.327; HC-FR regression only −0.021; confirmed Tier 2 | [exp_lora_reddit_fr.md](exp_lora_reddit_fr.md) |
| P2-E3 | LoRA — Joint adapter | ✅ | 0.632 (SG-2b) | — | Joint beats single FHS but trails single Reddit-FR by −0.030 (3.5× corpus dilution) | [exp_lora_joint.md](exp_lora_joint.md) |
| P2-E4 | LoRA — Balanced joint | ✅ | 0.611 (SG-2b) | — | Balancing hurts both domains; hypothesis disproven | [exp_lora_joint_balanced.md](exp_lora_joint_balanced.md) |

## Results data paths

- `results/phase2_eval/french_hate_superset/{baseline,lora}/` — FHS fair eval
- `results/phase2_eval/reddit_fr/{baseline,lora}/` — Reddit-FR fair eval
- `results/phase2_eval/lora_full/` — FHS adapter on all 8 datasets
- `results/lora_full_reddit_fr/` — Reddit-FR adapter on all 8 datasets
- `lora_adapters/shieldgemma_2b/reddit_fr/best/` — **Confirmed Tier 2 adapter**

## Key reminders

- All 3 runs overfit after epoch 1 — future retrains use `--epochs 1`
- SG-2b Reddit-FR required `--gradient_checkpointing` + `enable_input_require_grads()` + `--batch_size 2 --grad_accum 8` (CUDA OOM fix)
- Split method: `random.shuffle(seed=42)` — not stratified, but class ratios preserved within ±0.5 pp
- FHS adapter: useful only for formal French; actively harmful on Reddit/informal content
