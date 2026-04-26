# Phase 1 — Baseline Evaluation Index

**Status:** ✅ Complete | **Date:** 2026-03-29
**Conclusion:** ShieldGemma-2b is the best deployable French model (HC-FR 0.858, 5.7 GB VRAM, 24 ms/sample). No model exceeds F1=0.45 on FR-Hate or Reddit-FR — primary fine-tuning targets.

## Experiments

| ID    | Name | Status | F1 (HC-FR) | FNR | One-line result | File |
|-------|------|--------|:----------:|:---:|-----------------|------|
| P1-E1 | Full baseline (10×8) | ✅ | 0.858 (SG-2b) | — | SG-2b best deployable; KoalaAI/Detoxify-U not viable for French | [exp_full_baseline.md](exp_full_baseline.md) |
| P1-E2 | HateCheck breakdown | ✅ | — | — | SG-2b/9b counter-speech failure (correct-rate 0.054/0.090 FR) is primary deployment risk | [exp_hatecheck_breakdown.md](exp_hatecheck_breakdown.md) |
| P1-E3 | Threshold sensitivity | ✅ | — | 34–41% | Detoxify-M T1_FNR 34–41% on Reddit-FR at any deferral — motivates Phase 4 | [exp_threshold_sensitivity.md](exp_threshold_sensitivity.md) |

## Results data paths

- `results/full_baseline_v3/` — 10×8 grid (authoritative v3, 3 multi-runs)
- `results/hatecheck_analysis/` — EN + FR functionality breakdown
- `results/threshold_analysis/` — Detoxify-M two-threshold sweep

## Key reminders

- **VRAM guard** silently skips models that don't fit — check logs for `SKIPPED` lines
- **KoalaAI v2 WARNING:** v2 numbers are degenerate (all-positive classifier on French). Always cite v3.
- **CitizenLab:** may fail with `torch >= 2.6` CVE — needs safetensors conversion if repro needed
- **ShieldGemma v3 fix:** v1/v2 were broken (token-probability inference). v3 is authoritative.
- All 3 multi-runs show std ≤ 0.01 — results are fully reproducible.
