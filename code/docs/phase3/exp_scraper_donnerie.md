# Scraper — donnerie.be (Track B)

**Phase:** 3 | **ID:** P3-E3 | **Status:** ❌ Rerunning
**Date:** 2026-04-17 (fix applied) | **Script:** `code/phase3_data/scrape_donnerie.py`
**Results dir:** `~/code/data/donnerie/donnerie.jsonl`

## Configuration

| Parameter | Value |
|-----------|-------|
| Source | donnerie.be — Belgian French free-item donation platform |
| Class | solidarity_exchange (label=1) |
| Target | 1,000 items |
| Priority | Highest-relevance source (Belgian French, same register as Shareish) |

## Key Results

| Metric | Value |
|--------|-------|
| Collected | 0 / 1,000 |
| Failure cause | Scraper checked relative hrefs (`/annonces/slug/`) but site uses absolute hrefs (`https://donnerie.be/annonces/slug/`) |
| Fix applied | `parse_item_urls()` normalises hrefs to absolute before matching |
| Resubmitted | 2026-04-17 |
| Results | Pending |

## Conclusion

Zero items collected due to a URL matching bug — all 140 catalog pages were parsed but matched zero product URLs. Fix is straightforward (one-line href normalisation) and was resubmitted the same day. Once collected, donnerie.be will be the highest-quality positive-class data (Belgian French, identical register to Shareish).

## Cross-references

- Part of: Track B content-type classifier dataset
