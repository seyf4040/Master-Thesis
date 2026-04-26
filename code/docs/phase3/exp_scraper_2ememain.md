# Scraper — 2ememain.be (Track B)

**Phase:** 3 | **ID:** P3-E4 | **Status:** ❌ Needs fix
**Date:** 2026-04-17 | **Script:** `code/phase3_data/scrape_2ememain.py`
**Results dir:** `~/code/data/2ememain/2ememain.jsonl`

## Configuration

| Parameter | Value |
|-----------|-------|
| Source | 2ememain.be — Belgian French second-hand marketplace |
| Class | commercial_listing (label=0, negative class) |
| Target | 800 items |
| Importance | Primary source for commercial class (64:1 class imbalance without it) |

## Key Results

| Metric | Value |
|--------|-------|
| Collected | 51 / 800 (6%) |
| Issue 1 | Dutch condition-word filter (`"gebruikt"`, `"nieuw"` etc.) discards ~90% of listings |
| Issue 2 | JSON-LD `description` field truncated in static HTML; `window.__CONFIG__` fallback unreliable |
| Sample quality | ✅ Good (the 51 items that passed are correctly commercial) |

## Required fixes

1. Remove Dutch condition-word filter — rely on `langdetect` alone
2. Fix `window.__CONFIG__` description extraction for full text
3. Alternative: fetch only explicitly French-language subcategory pages

## Conclusion

The two compounding bugs (over-aggressive Dutch filter + truncated descriptions) reduced collection to just 6% of target. The 51 items that did pass are correctly formatted commercial listings, confirming the scraper logic is otherwise sound. This is the blocking issue for the content-type classifier (current imbalance 64:1) — until 2ememain.be is fixed and rerun, `generate_classifier_data.py` (300 synthetic commercial items) provides the interim commercial class.

## Cross-references

- Blocks: content-type classifier training
- Interim solution: `generate_classifier_data.py` (not yet submitted)
