# HateCheck Analysis — Gaps from Phase 1

**Status:** not started | **Priority:** medium — needed for consistent thesis tables

---

## Problem

The per-functionality HateCheck breakdown (`run_hatecheck_analysis.py`) was run with
v1/v2 model implementations. Two models were never re-run with the v3 fixes:

| Model | Issue | Impact |
|---|---|---|
| ShieldGemma | v2 used text generation → all predictions meaningless. v3 uses token-probability scoring which works correctly. | The entire per-functionality breakdown for ShieldGemma is wrong |
| CitizenLab | Missing from the HateCheck analysis entirely | No per-functionality data for CitizenLab |

## What To Do

Re-run `run_hatecheck_analysis.py` for these two models only:

```bash
python ~/code/run_hatecheck_analysis.py \
    --output_dir ~/code/results/hatecheck_analysis_v3 \
    --cache_dir ~/datasets/cache \
    --models shieldgemma_2b,citizenlab \
    --datasets both
```

Or submit via SLURM using `run_hatecheck_light.sbatch` (1080Ti-compatible subset) if
ShieldGemma-2b fits within available VRAM.

## Why It Matters

The HateCheck per-functionality breakdown is the main diagnostic tool for understanding
*what kind* of hate speech each model detects and misses. Publishing ShieldGemma numbers
from the broken v2 implementation would be a significant error in the thesis.

ShieldGemma-2b is one of the two primary Phase 2 fine-tuning targets — its HateCheck
profile directly informs which fine-tuning datasets to prioritize.
