# Phase 1 Baseline Review

**Date:** 2026-03-29 | **Source:** `full_baseline_v3/` + `hatecheck_analysis/` (authoritative)

---

## What's Been Done

Three pipeline versions were run. **v3 is the authoritative source** — v1/v2 had bugs:

- **v2 broke** Llama Guard / Mistral (wrong prompt format → TPR≈0), ShieldGemma (text generation instead of token-probability scoring → F1≈0), and KoalaAI (argmax always fired non-OK class on French → all-positive classifier)
- **v3 fixed** all of the above. Full 10 models × 8 datasets grid complete. **3 statistical runs** complete for mean ± std (std ≤ 0.01 for all models — no variance concerns).
- **HateCheck analysis** complete with all 10 models, v3 inference methods.

---

## Key Results (v3 F1 Scores)

> **Deployment target is French content (Shareish).** Table sorted by HC-FR. HC-EN and English-only datasets are shown for research comparability only.

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

⚠️ **detoxify-unbiased** and **KoalaAI** are not viable for Shareish (HC-FR 0.281 and 0.008 respectively).

---

## HateCheck TPR / TNR (sensitivity vs specificity)

| Model | EN F1 | EN TPR | EN TNR | FR F1 | FR TPR | FR TNR |
|-------|:-----:|:------:|:------:|:-----:|:------:|:------:|
| Llama-Guard-3-8B | 0.939 | 0.917 | 0.922 | 0.879 | 0.827 | 0.871 |
| Mistral-7B | 0.921 | 0.962 | 0.722 | 0.783 | 0.708 | 0.766 |
| ShieldGemma-9b | 0.913 | **0.988** | 0.613 | 0.883 | **0.929** | 0.591 |
| ShieldGemma-2b | 0.902 | 0.971 | 0.599 | 0.858 | 0.893 | 0.561 |
| Llama-Guard-3-1B | 0.819 | 0.781 | 0.722 | 0.676 | 0.573 | 0.713 |
| detoxify-multilingual | 0.803 | 0.847 | 0.423 | 0.787 | 0.787 | 0.505 |
| detoxify-unbiased | 0.760 | 0.766 | 0.451 | 0.281 | 0.171 | **0.897** |
| EthicalEye | 0.725 | 0.744 | 0.320 | 0.593 | 0.486 | 0.643 |
| CitizenLab | 0.702 | 0.668 | 0.482 | 0.644 | 0.584 | 0.466 |
| KoalaAI | 0.694 | 0.607 | 0.688 | 0.008 | 0.004 | 1.000 |

**ShieldGemma warning**: TPR ~0.97–0.99 but TNR ~0.60. Near-complete failure on `counter_quote_nh` (correct-rate 0.000/0.006) — counter-speech and quotes of hate are almost always flagged as toxic. Critical issue for Shareish where users report hate by quoting it.

---

## Deployability Tiers (Shareish context)

Ranked by **French F1** within each tier, since deployment is on French content.

| Tier | Models | HC-FR F1 | VRAM | Verdict |
|------|--------|:--------:|:----:|---------|
| Deployable (NGO) | detoxify-multilingual | 0.787 | ~1 GB | Best deployable bilingual model; no GPU needed |
| | CitizenLab | 0.644 | ~1 GB | Solid alternative at same cost |
| Mid-tier | Llama-Guard-3-1B | 0.674 | ~3 GB | Modest French gain over detoxify-M; needs small GPU |
| Sweet spot | ShieldGemma-2b | 0.858 | ~5.7 GB | Large French F1 gain; fast (25 ms/sample); counter-speech weakness |
| Top (not viable) | LG-3-8B / SGemma-9b / Mistral | 0.783–0.883 | 14–18 GB | Too heavy for Shareish |

Full deployability table (averaged across 8 datasets, v3):

| Model | GPU MB | ms/sample | Total energy (kWh) |
|-------|:------:|:---------:|:------------------:|
| detoxify-unbiased | 497 | 6.6 | 0.026 |
| CitizenLab | 1078 | 5.4 | 0.023 |
| EthicalEye | 1078 | 5.4 | 0.023 |
| detoxify-multilingual | 1079 | 6.4 | 0.025 |
| KoalaAI | 604 | 13.9 | 0.057 |
| Llama-Guard-3-1B | 2976 | 32.9 | 0.134 |
| ShieldGemma-2b | 5666 | 24.5 | 0.184 |
| ShieldGemma-9b | 18416 | 40.9 | 0.487 |
| Llama-Guard-3-8B | 15598 | 107.9 | 0.915 |
| Mistral-7B | 13951 | 150.3 | 1.316 |

---

## HateCheck Functionality Breakdown — FR (primary, deployment language)

Note: FR HateCheck does not include `slur_reclaimed_nh` or `slur_homonym_nh`.

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

**Key FR-specific weaknesses:**
- **ShieldGemma counter-speech failure persists in French** (`counter_quote_nh` 0.090/0.054) — critical deployment risk for Shareish.
- **KoalaAI**: 0.000 on every hateful category in French. All-negative classifier confirmed.
- **detoxify-unbiased on French NH categories**: scores look high (0.928 counter_quote, 0.954 target_indiv) but only because it predicts everything as safe. Hateful categories collapse to near-zero.
- **`slur_h` harder in French**: best is ShieldGemma-9b at 0.713 (vs 0.972 EN). LG-8B: 0.312 FR vs 0.562 EN.
- **`derog_impl_h` drops across models in French**: LG-1B 0.693→0.379, Mistral 0.893→0.593. Implied derogation harder to detect in French.
- **Obfuscation (`spell_*`) harder in French** for all models — likely English-biased obfuscation patterns in training.

---

## Notable Observations

- **Best deployable model for French**: detoxify-multilingual (HC-FR 0.787, 6ms, ~1 GB). No GPU required. Also leads on Reddit-FR (0.408) — the dataset most similar to actual Shareish content.
- **ShieldGemma unlocked**: was F1≈0 in v1/v2. Token-probability fix made it fully functional. ShieldGemma-2b (HC-FR 0.858, ~5.7 GB, 25ms) is the biggest accuracy jump among viable models — but counter-speech failure (correct-rate 0.000 on `counter_quote_nh`) must be addressed before Shareish deployment.
- **Llama-Guard-3-8B is the only model with balanced TPR/TNR on French** (TPR 0.827, TNR 0.871). All other high-F1 models over-predict toxic at the cost of specificity.
- **detoxify-unbiased fails on French** (HC-FR 0.281, TPR 0.171) — essentially refuses to classify French content as hateful. Not viable despite strong English scores.
- **KoalaAI is English-only** (HC-FR 0.008). The 0.938 OpenAI score is irrelevant for Shareish.
- **Reddit-FR is the most realistic Shareish proxy** and all models score poorly (best: detoxify-multilingual 0.408). This is the main concern for production deployment.
- **FR-Hate Superset**: hard for everyone (best: ShieldGemma-9b 0.442). Not just a French difficulty issue — detoxify-multilingual is 0.787 on HC-FR but only 0.292 here. Domain/label mismatch within the dataset.

---

## What Is Still Missing / Open Questions

- [ ] **Threshold sensitivity for ShieldGemma**: TPR/TNR imbalance may be correctable at threshold 0.7–0.8
- [ ] **FR-Hate / Reddit poor performance**: unexplained — label noise or domain shift? Dataset inspection needed before fine-tuning on these
- [ ] **Two-tier architecture** (detoxify pre-filter + Llama-Guard edge cases): not yet evaluated end-to-end

---

## Next Steps

- **Phase 2**: LoRA fine-tuning on **French data** — top candidates ranked by French F1 and deployability:
  1. detoxify-multilingual (HC-FR 0.787, deployable) — fine-tune on FR-Hate and Reddit-FR
  2. Llama-Guard-3-1B (HC-FR 0.674, ~3 GB) — fine-tune for better French coverage
  3. ShieldGemma-2b (HC-FR 0.858, ~5.7 GB) — fix counter-speech weakness, then fine-tune on Shareish data
- **Phase 3**: Active learning with Shareish data
- **Planned architecture**: Two-tier system — Detoxify as fast French pre-filter + Llama Guard for edge cases
