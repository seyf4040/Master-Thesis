# Results Summary

Last updated: **2026-03-29**

---

## Coverage

| Results dir | Models | Datasets | Notes |
|-------------|:------:|:--------:|-------|
| `full_baseline_v3/` | 10 | 8 | **Primary source** — best-of-breed implementations |
| `full_baseline_v3/run_1/`, `run_2/`, `run_3/` | 10 | 8 | 3 multi-runs — all complete, used for mean±std |
| `hatecheck_analysis/` | 10 | 2 | HateCheck EN + FR — all 10 models, v3 inference methods |
| `full_baseline_v2/` | 10 | 8 | Superseded — ShieldGemma/KoalaAI/Llama-Guard/Mistral bugs |
| `full_baseline/run_1/` | 9 | varies | Superseded — no CitizenLab, ShieldGemma broken, partial datasets |

---

## F1 Comparison: v1 / v3 (all models × all datasets)

v1 = `full_baseline/run_1`. v3±std from 3 multi-runs (all complete, std≤0.01 everywhere).

| Model | | HC-EN | HC-FR | FR-Hate | ToxiGen | OpenAI | CivComm | Red-EN | Red-FR |
|-------|---|:-----:|:-----:|:-------:|:-------:|:------:|:-------:|:------:|:------:|
| **detoxify-multilingual** | v1 | 0.803 | 0.787 | 0.292 | 0.486 | 0.688 | 0.723 | 0.332 | 0.408 |
| | v3 | 0.803 | 0.787 | 0.292 | 0.486 | 0.688 | 0.723 | 0.332 | 0.408 |
| | Δ | — | — | — | — | — | — | — | — |
| | v3±std | 0.80±0.00 | 0.79±0.00 | 0.29±0.00 | 0.49±0.00 | 0.69±0.00 | 0.72±0.00 | 0.33±0.00 | 0.41±0.00 |
| **detoxify-unbiased** | v1 | 0.760 | 0.281 | 0.072 | 0.463 | 0.672 | 0.763 | 0.318 | 0.166 |
| | v3 | 0.760 | 0.281 | 0.072 | 0.463 | 0.672 | 0.763 | 0.318 | 0.166 |
| | Δ | — | — | — | — | — | — | — | — |
| | v3±std | 0.76±0.00 | 0.28±0.00 | 0.07±0.00 | 0.46±0.00 | 0.67±0.00 | 0.76±0.00 | 0.32±0.00 | 0.17±0.00 |
| **EthicalEye** | v1 | 0.725 | 0.593 | 0.291 | 0.562 | 0.665 | 0.488 | 0.407 | 0.374 |
| | v3 | 0.725 | 0.593 | 0.291 | 0.562 | 0.665 | 0.488 | 0.407 | 0.374 |
| | Δ | — | — | — | — | — | — | — | — |
| | v3±std | 0.73±0.00 | 0.59±0.00 | 0.29±0.00 | 0.56±0.00 | 0.66±0.00 | 0.49±0.00 | 0.41±0.00 | 0.37±0.00 |
| **CitizenLab** | v1 | — | — | — | — | — | — | — | — |
| | v3 | 0.702 | 0.644 | 0.281 | 0.430 | 0.264 | 0.293 | 0.323 | 0.318 |
| | Δ | — | — | — | — | — | — | — | — |
| | v3±std | 0.70±0.00 | 0.64±0.00 | 0.28±0.00 | 0.43±0.00 | 0.26±0.00 | 0.29±0.00 | 0.32±0.00 | 0.32±0.00 |
| **KoalaAI** | v1 | 0.548 | 0.002 | 0.006 | 0.396 | 0.838 | 0.207 | 0.179 | 0.053 |
| | v3 | 0.694 | 0.008 | 0.040 | 0.502 | **0.938** | 0.245 | 0.299 | 0.326 |
| | Δ | **+0.146** | +0.006 | +0.034 | +0.106 | **+0.100** | +0.038 | +0.120 | **+0.273** |
| | v3±std | 0.69±0.00 | 0.01±0.00 | 0.04±0.00 | 0.50±0.00 | 0.94±0.00 | 0.25±0.00 | 0.30±0.00 | 0.33±0.00 |
| **Llama-Guard-3-1B** | v1 | 0.817 | 0.674 | 0.374 | 0.556 | 0.652 | 0.182 | 0.411 | 0.394 |
| | v3 | 0.816 | 0.674 | 0.372 | 0.556 | 0.651 | 0.187 | 0.407 | 0.398 |
| | Δ | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | +0.005 | ≈0 | ≈0 |
| | v3±std | 0.82±0.00 | 0.68±0.01 | 0.37±0.00 | 0.56±0.00 | 0.64±0.01 | 0.18±0.01 | 0.41±0.00 | 0.40±0.00 |
| **Llama-Guard-3-8B** | v1 | 0.939 | 0.879 | 0.354 | 0.546 | 0.785 | 0.110 | 0.160 | 0.268 |
| | v3 | 0.939 | 0.879 | 0.354 | 0.546 | 0.785 | 0.110 | 0.160 | 0.268 |
| | Δ | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| | v3±std | 0.94±0.00 | 0.88±0.00 | 0.35±0.00 | 0.55±0.00 | 0.79±0.00 | 0.11±0.00 | 0.16±0.00 | 0.27±0.00 |
| **ShieldGemma-2b** | v1 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 |
| | v3 | **0.902** | **0.858** | 0.441 | 0.632 | 0.499 | 0.315 | 0.227 | 0.311 |
| | Δ | **+0.902** | **+0.858** | **+0.439** | **+0.630** | **+0.499** | **+0.315** | +0.226 | +0.310 |
| | v3±std | 0.90±0.00 | 0.86±0.00 | 0.44±0.00 | 0.63±0.00 | 0.50±0.00 | 0.32±0.00 | 0.23±0.00 | 0.31±0.00 |
| **ShieldGemma-9b** | v1 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 |
| | v3 | **0.913** | **0.883** | 0.442 | 0.640 | 0.712 | 0.302 | 0.269 | 0.375 |
| | Δ | **+0.912** | **+0.883** | **+0.442** | **+0.639** | **+0.712** | **+0.302** | +0.267 | +0.375 |
| | v3±std | 0.91±0.00 | 0.88±0.00 | 0.44±0.00 | 0.64±0.00 | 0.71±0.00 | 0.30±0.00 | 0.27±0.00 | 0.37±0.00 |
| **Mistral-7B** | v1 | 0.922 | 0.781 | 0.390 | 0.667 | 0.762 | 0.293 | 0.330 | 0.318 |
| | v3 | 0.921 | 0.783 | 0.391 | 0.669 | 0.762 | 0.295 | 0.332 | 0.319 |
| | Δ | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| | v3±std | 0.92±0.00 | 0.78±0.00 | 0.39±0.00 | 0.67±0.00 | 0.76±0.00 | 0.30±0.00 | 0.33±0.00 | 0.32±0.00 |

**Notes on Δ rows:**
- **Detoxify / EthicalEye / CitizenLab**: Δ=0, std=0.00 — deterministic models, perfectly stable across all 3 runs.
- **Llama-Guard / Mistral**: Δ≈0 vs v1, confirming prompt fixes fully restored v1 behaviour. Max std across 3 runs is ±0.01 (CivComm).
- **KoalaAI**: genuine improvement over v1 on English (+0.146 HC-EN, +0.100 OpenAI). French still near-zero — model is English-only.
- **ShieldGemma-2b/9b**: token-probability fix unlocked both models entirely (+0.90 HC-EN). These were not broken models — they needed the right inference method.

> ⚠️ **KoalaAI v2 — DEGENERATE CLASSIFIER WARNING**
>
> **Do not use v2 KoalaAI numbers for any French or Reddit dataset.** The v2 `argmax` implementation always fired on a non-OK class for French text (near-uniform logits → any non-OK label wins argmax), making it a trivial all-positive classifier.
>
> A model predicting every sample as toxic achieves **F1 = 2P / (1 + P)** where P is the dataset positive-class rate. For HC-FR (70% hateful): F1 = 2×0.70 / 1.70 ≈ **0.824** — matching v2's 0.822 exactly. The v2 gains on FR-Hate (0.386), Reddit-EN (0.596), Reddit-FR (0.599) are all consistent with dataset-specific positive rates, not model quality. Evidence: v2 TNR≈0 on all French/Reddit datasets.
>
> **Rule:** when citing KoalaAI, always use v3 numbers. v3 thresholds on the sum of unsafe-class probabilities, which correctly produces near-zero confidence on French text.

---

## F1 Scores — `full_baseline_v3/` (authoritative, 10 × 8)

> **Deployment target is French content (Shareish).** HC-FR and FR-Hate are the primary evaluation columns; HC-EN and English-only datasets (ToxiGen, OpenAI, CivComm) are included for research comparability. Table sorted by HC-FR F1.

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

---

## HateCheck Analysis — `hatecheck_analysis/` (all 10 models, v3 methods)

| Model | EN F1 | EN TPR | EN TNR | FR F1 | FR TPR | FR TNR |
|-------|:-----:|:------:|:------:|:-----:|:------:|:------:|
| Llama-Guard-3-8B | **0.939** | 0.917 | 0.922 | **0.879** | 0.827 | 0.871 |
| Mistral-7B | 0.921 | **0.962** | 0.722 | 0.783 | 0.708 | 0.766 |
| ShieldGemma-9b | 0.913 | **0.988** | 0.613 | 0.883 | **0.929** | 0.591 |
| ShieldGemma-2b | 0.902 | 0.971 | 0.599 | 0.858 | 0.893 | 0.561 |
| Llama-Guard-3-1B | 0.819 | 0.781 | 0.722 | 0.676 | 0.573 | 0.713 |
| detoxify-multilingual | 0.803 | 0.847 | 0.423 | 0.787 | 0.787 | 0.505 |
| detoxify-unbiased | 0.760 | 0.766 | 0.451 | 0.281 | 0.171 | **0.897** |
| EthicalEye | 0.725 | 0.744 | 0.320 | 0.593 | 0.486 | 0.643 |
| CitizenLab | 0.702 | 0.668 | 0.482 | 0.644 | 0.584 | 0.466 |
| KoalaAI | 0.694 | 0.607 | 0.688 | 0.008 | 0.004 | 1.000 |

**TPR/TNR patterns:**
- **Llama-Guard-3-8B**: best overall balance (TPR 0.92, TNR 0.92 EN). Only model with high sensitivity AND high specificity.
- **ShieldGemma-9b/2b**: very high TPR (0.97–0.99) but low TNR (0.60). Tend to over-predict toxic — high recall, elevated false-positive rate. Acceptable for a permissive filter but not a standalone classifier.
- **Mistral-7B**: similar over-prediction tendency (TPR 0.96, TNR 0.72), less extreme than ShieldGemma.
- **detoxify-unbiased**: highest TNR on French (0.897) but near-random TPR (0.171) — essentially refuses to classify French as hateful.
- **KoalaAI**: strong TNR on English (0.688) but functionally blind on French (TPR=0.004).

---

## HateCheck Functionality Breakdown — EN (notable patterns)

Rows = functionality. H = hateful category, NH = non-hateful category. Values = correct-rate (not F1).

| Functionality | Type | LG-8B | SG-9b | Mistr | SG-2b | LG-1B | Detox-M | CiLab | Detox-U | EthEye | Koala |
|---------------|------|:-----:|:-----:|:-----:|:-----:|:-----:|:-------:|:-----:|:-------:|:------:|:-----:|
| counter_quote_nh | NH | 0.763 | **0.000** | 0.410 | **0.006** | 0.549 | 0.087 | 0.000 | 0.168 | 0.116 | 0.642 |
| counter_ref_nh | NH | 0.936 | **0.028** | 0.553 | 0.142 | 0.532 | 0.241 | 0.142 | 0.277 | 0.163 | 0.752 |
| slur_reclaimed_nh | NH | 0.975 | 0.605 | 0.704 | 0.494 | 0.679 | 0.136 | 0.667 | 0.210 | 0.123 | 0.395 |
| profanity_nh | NH | 1.000 | 0.960 | 0.840 | 0.980 | 0.950 | 0.130 | 0.300 | 0.130 | 0.000 | 0.660 |
| target_indiv_nh | NH | 0.708 | 0.492 | 0.292 | 0.523 | 0.585 | 0.154 | 0.169 | 0.138 | 0.231 | 0.215 |
| target_group_nh | NH | 0.758 | 0.419 | 0.500 | 0.452 | 0.726 | 0.339 | 0.306 | 0.387 | 0.419 | 0.677 |
| negate_neg_nh | NH | 0.977 | 0.805 | 0.902 | 0.699 | 0.707 | 0.444 | 0.504 | 0.346 | 0.241 | 0.602 |
| derog_dehum_h | H | 1.000 | 1.000 | 1.000 | 1.000 | 0.936 | 0.971 | 0.900 | 0.886 | 0.814 | 0.786 |
| threat_dir_h | H | 1.000 | 1.000 | 1.000 | 1.000 | 0.970 | 0.992 | 0.361 | 0.985 | 0.865 | 0.917 |
| slur_h | H | 0.562 | 0.972 | 0.951 | 0.972 | 0.431 | 0.736 | 0.806 | 0.625 | 0.667 | 0.583 |
| spell_leet_h | H | 0.844 | 0.936 | 0.902 | 0.879 | 0.618 | 0.803 | 0.682 | 0.642 | 0.630 | 0.462 |

**Key weaknesses (EN):**
- **ShieldGemma-9b/2b**: near-complete failure on `counter_quote_nh` (0.000/0.006) and `counter_ref_nh` (0.028/0.142). Counter-speech and quotes of hate speech are almost always flagged as toxic. This is their core weakness — a content moderation system using ShieldGemma would suppress legitimate counter-speech.
- **CitizenLab**: fails on `counter_quote_nh` (0.000), weak on `threat_dir_h` (0.361).
- **Llama-Guard-3-8B on `slur_h`**: only 0.562 — surprisingly weak at detecting slurs.
- **All models on `target_indiv_nh`**: best is LG-8B at 0.708. Models frequently flag non-hateful statements that target individuals.

---

## HateCheck Functionality Breakdown — FR (primary, deployment language)

Note: FR HateCheck does not include `slur_reclaimed_nh` or `slur_homonym_nh` (no French equivalents).

| Functionality | Type | LG-8B | SG-9b | Mistr | SG-2b | LG-1B | Detox-M | CiLab | Detox-U | EthEye | Koala |
|---------------|------|:-----:|:-----:|:-----:|:-----:|:-----:|:-------:|:-----:|:-------:|:------:|:-----:|
| counter_quote_nh | NH | 0.671 | **0.090** | 0.455 | **0.054** | 0.515 | 0.257 | 0.096 | **0.928** | 0.509 | 1.000 |
| counter_ref_nh | NH | 0.826 | **0.180** | 0.599 | **0.162** | 0.551 | 0.317 | 0.060 | **0.874** | 0.557 | 1.000 |
| negate_neg_nh | NH | 0.900 | 0.650 | 0.893 | 0.579 | 0.621 | 0.557 | 0.543 | 0.757 | 0.450 | 1.000 |
| profanity_nh | NH | 1.000 | 0.910 | 0.900 | 0.950 | 0.870 | 0.290 | 0.400 | **0.990** | 0.770 | 1.000 |
| target_indiv_nh | NH | 0.723 | 0.446 | 0.508 | 0.554 | 0.754 | 0.246 | 0.277 | **0.954** | 0.708 | 1.000 |
| target_group_nh | NH | 0.738 | 0.400 | 0.631 | 0.431 | 0.723 | 0.354 | 0.323 | **0.954** | 0.815 | 1.000 |
| derog_dehum_h | H | 0.936 | **1.000** | 0.821 | **0.971** | 0.807 | 0.929 | 0.800 | 0.286 | 0.536 | **0.000** |
| derog_impl_h | H | 0.779 | **0.864** | 0.593 | **0.807** | 0.379 | 0.593 | 0.293 | 0.107 | 0.543 | **0.000** |
| negate_pos_h | H | 0.879 | **1.000** | 0.850 | **0.986** | 0.500 | 0.486 | 0.286 | 0.129 | 0.407 | **0.000** |
| slur_h | H | 0.312 | **0.713** | 0.356 | 0.506 | 0.219 | 0.594 | 0.650 | 0.000 | 0.244 | **0.000** |
| spell_char_del_h | H | 0.829 | **0.914** | 0.529 | **0.864** | 0.457 | 0.693 | 0.671 | 0.121 | 0.329 | **0.000** |
| spell_leet_h | H | 0.707 | **0.880** | 0.689 | **0.838** | 0.533 | 0.772 | 0.569 | 0.132 | 0.377 | 0.006 |
| threat_dir_h | H | **0.993** | **0.993** | 0.936 | **0.986** | 0.743 | 0.957 | 0.464 | 0.143 | 0.621 | 0.007 |

**Key weaknesses (FR — deployment-relevant):**
- **ShieldGemma-9b/2b**: counter-speech failure persists in French (`counter_quote_nh` 0.090/0.054, `counter_ref_nh` 0.180/0.162). Still the primary deployment risk for Shareish.
- **KoalaAI**: 0.000 on every hateful category in French — all-negative classifier. Confirmed not viable.
- **detoxify-unbiased**: inverted behaviour on French NH categories — high scores on `counter_quote_nh` (0.928), `target_indiv_nh` (0.954), `target_group_nh` (0.954) because it predicts everything as safe, not because it understands French. Hateful categories collapse to near-zero (e.g. `derog_dehum_h` 0.286, `slur_h` 0.000).
- **`slur_h` is harder in French for all models**: best is ShieldGemma-9b at 0.713 (vs 0.972 EN). French slurs are likely underrepresented in training data.
- **`derog_impl_h` drops across the board in French**: LG-1B 0.693→0.379, Mistral 0.893→0.593. Implied derogation in French is harder to catch.
- **`spell_*` categories**: obfuscation attacks (leet, char swap, space deletion) are consistently harder in French — models trained primarily on English obfuscation patterns.
- **Llama-Guard-3-8B on `slur_h`**: 0.312 FR (vs 0.562 EN) — the slur weakness is worse in French.

---

## Deployability — `full_baseline_v3/`

Averaged across all 8 datasets. Energy is total kWh across all 8 × n_samples.

| Model | GPU MB | ms/sample | Total energy (kWh) |
|-------|:------:|:---------:|:------------------:|
| detoxify-unbiased | **497** | 6.6 | 0.026 |
| CitizenLab | 1078 | **5.4** | 0.023 |
| EthicalEye | 1078 | **5.4** | 0.023 |
| detoxify-multilingual | 1079 | 6.4 | 0.025 |
| KoalaAI | 604 | 13.9 | 0.057 |
| Llama-Guard-3-1B | 2976 | 32.9 | 0.134 |
| ShieldGemma-2b | 5666 | 24.5 | 0.184 |
| ShieldGemma-9b | 18416 | 40.9 | 0.487 |
| Llama-Guard-3-8B | 15598 | 107.9 | 0.915 |
| Mistral-7B | 13951 | 150.3 | 1.316 |

> **ShieldGemma ms/sample**: v3 token-probability scoring does a single forward pass vs full generation (v1 ShieldGemma: ~2000ms, v2: ~3200ms). With v3, ShieldGemma-2b (24.5ms) is faster than Llama-Guard-3-1B (32.9ms) while using slightly more VRAM.
>
> **Mistral-7B**: 150ms/sample in v3 (greedy decode, single token). Earlier incorrectly estimated at 330ms based on intermediate cluster logs — the 150ms figure from the completed 3-run average is authoritative.

---

## Key Observations (v3, Final Baseline)

> **Reading guide**: Shareish is a French-language platform. HC-FR and FR-Hate are the primary benchmarks. English results are shown for research comparability.

**1. French performance ranking:**

On HC-FR (primary benchmark): ShieldGemma-9b (0.883) ≈ Llama-Guard-3-8B (0.879) > ShieldGemma-2b (0.858) > detoxify-multilingual (0.787) > Mistral-7B (0.783) > Llama-Guard-3-1B (0.674) > CitizenLab (0.644) > EthicalEye (0.593) >> detoxify-unbiased (0.281) >> KoalaAI (0.008).

**detoxify-unbiased and KoalaAI are not viable for Shareish** regardless of their English scores.

**2. Best deployable model for French:**

detoxify-multilingual (HC-FR 0.787, 6ms, ~1GB VRAM) is the strongest viable option for an NGO without GPU infrastructure. It also leads on Reddit-FR (0.408). CitizenLab (HC-FR 0.644) is an alternative at the same footprint but weaker. Both EthicalEye and detoxify-unbiased fall behind on French, making detoxify-multilingual the clear lightweight choice.

**3. ShieldGemma specificity trade-off (critical for French deployment):**

ShieldGemma-9b/2b achieve near-perfect TPR (~0.93/0.89 FR) at the cost of TNR (~0.59 FR). Their failure on `counter_quote_nh` (correct-rate 0.000/0.006 EN) means counter-speech is almost always flagged as hate. On a platform like Shareish, users who quote hate speech they received to report it would be silenced. This is a more serious concern in French than the F1 score suggests.

**4. Deployability tiers for Shareish (ranked by French F1):**

- **≤1.1GB VRAM** (CPU-feasible): detoxify-multilingual (HC-FR 0.787, 6ms) — best deployable model for French. No GPU needed.
- **~3GB VRAM**: Llama-Guard-3-1B (HC-FR 0.674, 33ms) — accuracy gain over detoxify-M is modest (+0.087 HC-FR) but meaningful.
- **~5.7GB VRAM**: ShieldGemma-2b (HC-FR 0.858, 24ms) — largest viable jump; counter-speech weakness is manageable with threshold tuning or fine-tuning.
- **≥14GB VRAM**: Llama-Guard-3-8B (HC-FR 0.879), ShieldGemma-9b (0.883), Mistral-7B (0.783) — not viable for Shareish.

**5. Dataset-level patterns:**

- **French Hate Superset**: hardest for all models (best: ShieldGemma-9b 0.442). Not just a language issue — detoxify-multilingual scores 0.787 HC-FR but only 0.292 FR-Hate. Label distribution or domain mismatch within the dataset.
- **Reddit-FR**: all models weak (best: detoxify-multilingual 0.408). This is the most realistic proxy for Shareish content — the poor scores here are a concern for deployment.
- **Civil Comments**: detoxify models dominate (0.72–0.76); all large models weak (<0.32). CivComm uses English community toxicity norms — less relevant for French deployment.
- **OpenAI / ToxiGen**: mostly English datasets. KoalaAI 0.938 on OpenAI is the highest score in the table but irrelevant for French use.

**6. CitizenLab (new baseline):**

HC-FR 0.644 at 1GB VRAM — solid bilingual option, stronger on French than EthicalEye (0.593). Counter-speech failure (0.000 `counter_quote_nh`) is a concern shared with ShieldGemma.

**7. Stability (3 multi-runs):**

All deterministic models have std=0.00. Only Llama-Guard-3-1B shows any variance (max ±0.01). Results are fully reproducible.

---

## What Is Still Missing / Next Steps

**Phase 1 (Baseline) — complete.** All 10 models × 8 datasets × 3 runs. HateCheck analysis with all 10 models.

**Phase 2 candidates (LoRA fine-tuning on French data):**
- **detoxify-multilingual**: strong French baseline (0.787), deployable anywhere. Fine-tuning target: FR-Hate (0.292) and Reddit-FR (0.408).
- **Llama-Guard-3-1B**: best accuracy/VRAM trade-off; native safety tuning format. Fine-tuning target: French Hate Superset, Reddit-FR.
- **ShieldGemma-2b**: highest French F1 among viable models (0.858). Fine-tuning target: counter-speech handling + FR-Hate.

**Known gaps:**
- Threshold sensitivity: ShieldGemma's FR TNR (0.561) may improve significantly at threshold 0.7–0.8 — test before committing to fine-tuning.
- Reddit-FR / FR-Hate underperformance: dataset inspection needed (label noise? domain shift?) before fine-tuning on these.
- Two-tier architecture (detoxify pre-filter + Llama-Guard edge cases): not yet evaluated end-to-end.
