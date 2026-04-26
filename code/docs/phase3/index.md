# Phase 3 — Data Collection Index

**Status:** 🔄 In progress | **Date range:** 2026-04-14 → ongoing
**Conclusion:** Track A (synthetic, 1,500 items) complete. Track B (scrapers) partially done — donnons.org complete (3,238), donnerie.be and 2ememain.be need fixes. Classifier cannot train until commercial class reaches ~500 items.

## Experiments

| ID    | Name | Status | Collected | Target | One-line result | File |
|-------|------|--------|:---------:|:------:|-----------------|------|
| P3-E1 | Synthetic French hate (Track A) | ✅ | 1,500 | 1,500 | 3/5 functionalities usable as-is; strip "1.1." artifact before training | [exp_synthetic_french_hate.md](exp_synthetic_french_hate.md) |
| P3-E2 | Scraper — donnons.org | ✅ | 3,238 | 3,400 | 17-category solidarity data; HTML entity issue in location field | [exp_scraper_donnons.md](exp_scraper_donnons.md) |
| P3-E3 | Scraper — donnerie.be | ❌ | 0 | 1,000 | Bug fixed (absolute href); resubmitted 2026-04-17, results pending | [exp_scraper_donnerie.md](exp_scraper_donnerie.md) |
| P3-E4 | Scraper — 2ememain.be | ❌ | 51 | 800 | Dutch filter + truncated descriptions; scraper needs fix | [exp_scraper_2ememain.md](exp_scraper_2ememain.md) |

## Results data paths

- `~/code/data/synthetic/{functionality}.jsonl` — Track A (1,500 items)
- `~/code/data/donnons/{category}.jsonl` — Track B (3,238 items)
- `~/code/data/donnerie/donnerie.jsonl` — Track B (pending rerun)
- `~/code/data/2ememain/2ememain.jsonl` — Track B (51 items, broken)

## Key reminders

- **Current class imbalance: 64:1** (3,289 solidarity : 51 commercial) — cannot train classifier yet
- Strip `"1.1. "` prefix from all synthetic items before training: `re.sub(r"^\d+\.\d*\s*", "", text)`
- `derog_impl_h` and `spell_leet_h` need manual review / filtering before use
- `generate_classifier_data.py` (not yet run) generates 300 synthetic commercial examples as interim fix
- Minimum viable set: ~1,000 solidarity + ~500 commercial (2:1 ratio)
