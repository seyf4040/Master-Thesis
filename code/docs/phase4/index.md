# Phase 4 — Two-Tier Architecture Index

**Status:** 🔄 In progress | **Date range:** 2026-04-16 → ongoing
**Conclusion:** 2c-synthetic (epoch 2) confirmed as Tier 1. Neither T1 variant exceeds Tier 2 alone in F1 at useful deferral rates, but the 4× speed advantage is the deployability argument. Group 3 pending.

## Experiments

| ID    | Name | Status | F1 (honest) | T1_FNR | One-line result | File |
|-------|------|--------|:-----------:|:------:|-----------------|------|
| P4-E1 | T1 pretrained baseline (Track A) | ✅ | 0.616 (best-T) | 37.6% | Identical failure to Detoxify-M; same backbone → same score collapse | [exp_t1_pretrained_baseline.md](exp_t1_pretrained_baseline.md) |
| P4-E2 | T1 fine-tuned base (Track B + Group 1) | ✅ | 0.626 (combined) | 41.7% | Honest FNR 41.7% vs leaky 25.2%; bimodal collapse; speed 4.4× | [exp_t1_finetuned_base.md](exp_t1_finetuned_base.md) |
| P4-E3 | T1 variant 2a (10 epochs) | ✅ | 0.615 | 28.5% | Overfits after ep2; extra epochs wasted; T1_FNR drops to 28.5% | [exp_t1_variant_2a.md](exp_t1_variant_2a.md) |
| P4-E4 | T1 variant 2b (soft labels ε=0.05) | ✅ | 0.619 | 28.3% | T1_FPR drops 23.8%→13.5%; partial distribution fix; T_high still 0.95 | [exp_t1_variant_2b.md](exp_t1_variant_2b.md) |
| P4-E5 | T1 variant 2c (synthetic data) | ✅ | **0.668** | **25.2%** | **Best: T_high=0.80, HC-FR FNR=7.0%. Confirmed final Tier 1.** | [exp_t1_variant_2c.md](exp_t1_variant_2c.md) |
| P4-E6 | T1 variant 2d (soft + synthetic) | ✅ | 0.634 | 27.1% | Effects do not stack; worse than 2c; 2c is the ceiling for this approach | [exp_t1_variant_2d.md](exp_t1_variant_2d.md) |
| P4-E7 | Generalisation eval (6 ckpts × 8 datasets) | ✅ | — | 7.0% HC-FR | 2c overwhelmingly best on structured hate speech; ToxiGen/OpenAI broken | [exp_generalisation_eval.md](exp_generalisation_eval.md) |
| P4-E8 | End-to-end two-tier (2c simulation) | ⏳ | — | — | Pending: score_two_tier_finetuned.sbatch with 2c checkpoint | [exp_end_to_end_two_tier.md](exp_end_to_end_two_tier.md) |

## Results data paths

- `results/tier1_comparison/` — Track A + B threshold analysis
- `results/two_tier_scores/{pretrained,finetuned}/` — Group 1 honest scores
- `results/tier1_finetuned_{e10,soft,synthetic}/` — Group 2 checkpoints
- `results/tier1_finetuned_2d/` — Group 2d checkpoint
- `results/tier1_comparison_honest/` — Group 2 threshold analysis (honest holdout)
- `results/tier1_generalisation/` — All 6 checkpoints × 8 datasets (2026-04-24)

## Key reminders

- **Best epoch is always 2** for all fine-tuning variants — use `--epochs 2`
- **Reddit-FR in generalisation eval is contaminated** — use Group 2 honest 511-sample figures
- **Success criterion:** combined F1 > 0.640 (Tier 2 alone) at < 30% deferral
- **Tier 1 best config (2c):** T_low=0.10, T_high=0.75, deferral=10.4%, combined ~12.7 ms/sample
- **Group 3 (T2 specialisation):** Collect deferred samples from Reddit-FR train through 2c, then fine-tune SG-2b
