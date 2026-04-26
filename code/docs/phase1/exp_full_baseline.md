# Full Baseline Evaluation — 10 Models × 8 Datasets

**Phase:** 1 | **ID:** P1-E1 | **Status:** ✅ Complete
**Date:** 2026-03-29 | **Script:** `code/phase1_baseline/run_full_baseline_v3.py`
**Results dir:** `results/full_baseline_v3/` (3 multi-runs, std ≤ 0.01 everywhere)

## Configuration

| Parameter | Value |
|-----------|-------|
| Models | 10: detoxify-M, detoxify-U, EthicalEye, CitizenLab, KoalaAI, LG-1B, LG-8B, SG-2b, SG-9b, Mistral-7B |
| Datasets | 8: HC-EN, HC-FR, FR-Hate, ToxiGen, OpenAI, CivComm, Reddit-EN, Reddit-FR |
| Threshold | 0.50 default |
| Runs | 3 multi-runs for statistical stability (std ≤ 0.01 everywhere) |
| Hardware | A5000, a5000 partition |

## F1 Scores — `full_baseline_v3/` (authoritative, sorted by HC-FR)

> Deployment target is French content (Shareish). HC-FR and FR-Hate are primary; English datasets are for research comparability.

| Model | **HC-FR** | **FR-Hate** | **Red-FR** | HC-EN | ToxiGen | OpenAI | CivComm | Red-EN |
|-------|:---------:|:-----------:|:----------:|:-----:|:-------:|:------:|:-------:|:------:|
| ShieldGemma-9b | **0.883** | **0.442** | 0.375 | 0.913 | 0.640 | 0.712 | 0.302 | 0.269 |
| Llama-Guard-3-8B | **0.879** | 0.354 | 0.268 | **0.939** | 0.546 | 0.785 | 0.110 | 0.160 |
| ShieldGemma-2b | 0.858 | 0.441 | 0.311 | 0.902 | 0.632 | 0.499 | 0.315 | 0.227 |
| detoxify-multilingual | 0.787 | 0.292 | **0.408** | 0.803 | 0.486 | 0.688 | 0.723 | 0.332 |
| Mistral-7B | 0.783 | 0.391 | 0.319 | 0.921 | **0.669** | 0.762 | 0.295 | 0.332 |
| Llama-Guard-3-1B | 0.674 | 0.372 | 0.398 | 0.816 | 0.556 | 0.651 | 0.187 | 0.407 |
| CitizenLab | 0.644 | 0.281 | 0.318 | 0.702 | 0.430 | 0.264 | 0.293 | 0.323 |
| EthicalEye | 0.593 | 0.291 | 0.374 | 0.725 | 0.562 | 0.665 | 0.488 | 0.407 |
| detoxify-unbiased | 0.281 | 0.072 | 0.166 | 0.760 | 0.463 | 0.672 | **0.763** | 0.318 |
| KoalaAI | 0.008 | 0.040 | 0.326 | 0.694 | 0.502 | **0.938** | 0.245 | 0.299 |

⚠️ **detoxify-unbiased** (HC-FR 0.281) and **KoalaAI** (HC-FR 0.008) are **not viable for Shareish** regardless of English scores.

## v1 vs v3 Δ (key models with non-trivial changes)

| Model | HC-FR Δ | Notes |
|-------|:-------:|-------|
| KoalaAI | +0.006 | v1 0.002 → v3 0.008. Still near-zero. |
| KoalaAI HC-EN | **+0.146** | v1 0.548 → v3 0.694. Better inference method. |
| ShieldGemma-2b | **+0.858** | v1 ~0.000 (broken) → v3 0.858. Token-probability fix. |
| ShieldGemma-9b | **+0.883** | v1 ~0.000 (broken) → v3 0.883. Token-probability fix. |
| All others | ≈0 | Stable across versions. Max std ±0.01. |

> ⚠️ **KoalaAI v2 DEGENERATE CLASSIFIER WARNING:** v2 used `argmax` which always fired non-OK on French (near-uniform logits). v2 HC-FR 0.822 is an all-positive classifier artefact (matches 2×0.70/1.70). Never cite v2 for French or Reddit datasets — always use v3.

## Deployability — averaged across all 8 datasets

| Model | GPU MB | ms/sample | Total energy (kWh) |
|-------|:------:|:---------:|:------------------:|
| detoxify-unbiased | **497** | 6.6 | 0.026 |
| CitizenLab | 1,078 | **5.4** | 0.023 |
| EthicalEye | 1,078 | **5.4** | 0.023 |
| detoxify-multilingual | 1,079 | 6.4 | 0.025 |
| KoalaAI | 604 | 13.9 | 0.057 |
| Llama-Guard-3-1B | 2,976 | 32.9 | 0.134 |
| **ShieldGemma-2b** | **5,666** | **24.5** | **0.184** |
| ShieldGemma-9b | 18,416 | 40.9 | 0.487 |
| Llama-Guard-3-8B | 15,598 | 107.9 | 0.915 |
| Mistral-7B | 13,951 | 150.3 | 1.316 |

> **SG-2b ms/sample note:** v3 token-probability scoring uses a single forward pass (vs ~2000ms full generation in v1). SG-2b (24.5ms) is now faster than LG-1B (32.9ms) while using slightly more VRAM.
> **Mistral-7B ms:** 150ms/sample is authoritative from the 3-run average. Earlier intermediate log estimate of 330ms was wrong.

## Deployability Tiers for Shareish

| VRAM tier | Model | HC-FR F1 | ms/sample | Notes |
|-----------|-------|:--------:|:---------:|-------|
| ≤1.1 GB (CPU-feasible) | detoxify-multilingual | 0.787 | 6 | Best deployable; no GPU needed |
| ~3 GB | Llama-Guard-3-1B | 0.674 | 33 | Modest accuracy gain (+0.087 HC-FR) |
| ~5.7 GB | **ShieldGemma-2b** | **0.858** | 24 | Best deployable jump; counter-speech weakness manageable |
| ≥14 GB | LG-8B, SG-9b, Mistral-7B | 0.879–0.921 | 41–150 | Not viable for Shareish |

## Key Observations

1. **French/English gap is structural**: KoalaAI 0.694 HC-EN vs 0.008 HC-FR. detoxify-unbiased 0.760 HC-EN vs 0.281 HC-FR.
2. **FR-Hate and Reddit-FR are hardest**: no model exceeds F1=0.45 on either → primary fine-tuning targets.
3. **SG-2b counter-speech risk**: FR TNR 0.561 means counter-speech frequently flagged (see P1-E2).
4. **CivComm**: detoxify models dominate (0.72–0.76); all large models weak (<0.32). Less relevant for French deployment.
5. **Stability**: all deterministic models std=0.00. Only LG-1B shows any variance (max ±0.01 on CivComm).

## Cross-references

- HateCheck TPR/TNR breakdown: [P1-E2 (exp_hatecheck_breakdown.md)](exp_hatecheck_breakdown.md)
- Threshold sensitivity: [P1-E3 (exp_threshold_sensitivity.md)](exp_threshold_sensitivity.md)
- Motivates: [P2-E2 (LoRA Reddit-FR)](../phase2/exp_lora_reddit_fr.md), [P4-E1 (Tier 1 baseline)](../phase4/exp_t1_pretrained_baseline.md)
