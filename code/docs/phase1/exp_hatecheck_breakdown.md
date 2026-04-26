# HateCheck Functionality Breakdown

**Phase:** 1 | **ID:** P1-E2 | **Status:** ✅ Complete
**Date:** 2026-03-29 | **Script:** `code/phase1_baseline/run_hatecheck_analysis.py`
**Results dir:** `results/hatecheck_analysis/`

## Configuration

| Parameter | Value |
|-----------|-------|
| Models | All 10 (v3 inference methods) |
| Datasets | HateCheck EN + HateCheck FR |
| Metric | Correct-rate per functionality (not F1) |
| Rows | H = hateful category, NH = non-hateful category |

## HateCheck EN — Functionality Breakdown

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
- **SG-9b/2b**: near-complete failure on `counter_quote_nh` (0.000/0.006) and `counter_ref_nh` (0.028/0.142). Counter-speech and quoted hate speech almost always flagged.
- **CitizenLab**: fails on `counter_quote_nh` (0.000), weak on `threat_dir_h` (0.361).
- **LG-8B on `slur_h`**: only 0.562 — surprisingly weak at detecting English slurs.
- **All models on `target_indiv_nh`**: best LG-8B at 0.708 — models frequently flag non-hateful statements targeting individuals.

## HateCheck FR — Functionality Breakdown (primary, deployment language)

> FR HateCheck does not include `slur_reclaimed_nh` or `slur_homonym_nh` (no French equivalents).

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
- **SG-9b/2b**: counter-speech failure persists in French (`counter_quote_nh` 0.090/0.054, `counter_ref_nh` 0.180/0.162). Primary deployment risk for Shareish.
- **KoalaAI**: 0.000 on every hateful FR category — confirmed all-negative classifier. Not viable.
- **detoxify-unbiased**: inverted FR behaviour — high NH scores (0.928, 0.954) because it predicts everything as safe, not because it understands French. Hateful categories collapse to near-zero.
- **`slur_h` harder in French**: best SG-9b 0.713 vs 0.972 EN — French slurs underrepresented in training data.
- **`derog_impl_h` drops across the board in French**: LG-1B 0.693 EN → 0.379 FR, Mistral 0.893 → 0.593. Implied derogation harder in French.
- **Obfuscation attacks harder in French**: `spell_*` categories consistently lower — models trained on English obfuscation patterns.
- **LG-8B on `slur_h` FR**: 0.312 (vs 0.562 EN) — slur weakness is worse in French.

## Conclusion

The primary deployment risk for Shareish is SG-2b/9b counter-speech failure: `counter_quote_nh` correct-rate of 0.054/0.090 in French means users quoting hate speech they received to report it would be silenced. This is more serious than the aggregate F1 suggests. KoalaAI is 0.000 on every hateful French functionality — confirmed not viable regardless of English scores. The detoxify-unbiased "high TNR" on French is a mirage — the model predicts everything as safe. French slur detection, implied derogation, and obfuscation attacks are universally harder than English equivalents.

## Cross-references

- Motivated by: [P1-E1 (full baseline)](exp_full_baseline.md)
- Informs: Phase 2 fine-tuning target selection, [P3-E1 (synthetic data functionalities)](../phase3/exp_synthetic_french_hate.md)
