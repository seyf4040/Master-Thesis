# Synthetic French Hate Speech — Track A

**Phase:** 3 | **ID:** P3-E1 | **Status:** ✅ Complete
**Date:** 2026-04-14 | **Script:** `code/phase3_data/generate_synthetic_data.py`
**Results dir:** `~/code/data/synthetic/{functionality}.jsonl`

## Configuration

| Parameter | Value |
|-----------|-------|
| Generator | `mistral-7b-instruct-v0.3` |
| Target | 300 items × 5 HateCheck-FR functionalities = 1,500 total |
| Hardware | A5000, ~22 min total |
| Output format | JSONL with `text`, `label` (1=hate, 0=safe) |

## Key Results

| Functionality | Label | Collected | Quality | Notes |
|---------------|:-----:|:---------:|:-------:|-------|
| slur_h | 1 | 300 | ✅ Good | Explicit slurs, varied groups |
| spell_char_del_h | 1 | 300 | ✅ Good | Character deletion confirmed |
| counter_quote_nh | 0 | 300 | ✅ Good | Counter-speech framing correct |
| spell_leet_h | 1 | 300 | ⚠️ Mixed | ~30% lack any obfuscation chars |
| derog_impl_h | 1 | 300 | ⚠️ Mixed | Drifts to neutral text; needs review |

## Conclusion

All 1,500 target items collected. Three functionalities (slur_h, spell_char_del_h, counter_quote_nh) are usable after the artifact strip. Two require pre-training filtering. These 1,500 items were used in Phase 4 T1 variant 2c and produced dramatic HC-FR FNR improvement (45.8%→7.0%), confirming that HateCheck-functionality coverage in training data directly generalises to HC-FR test cases.

## Known issues / caveats

- **Strip "1.1." prefix before ANY training use:** `re.sub(r"^\d+\.\d*\s*", "", text)`
- `spell_leet_h`: keep only examples with `[@$3!01€]` chars
- `derog_impl_h`: manual spot-check of 100 random samples before use

## Cross-references

- Used in: [P4-E5 (T1 variant 2c)](../phase4/exp_t1_variant_2c.md)
