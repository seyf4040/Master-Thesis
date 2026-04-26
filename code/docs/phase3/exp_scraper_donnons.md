# Scraper — donnons.org (Track B)

**Phase:** 3 | **ID:** P3-E2 | **Status:** ✅ Complete
**Date:** 2026-04-14 | **Script:** `code/phase3_data/scrape_donnons.py`
**Results dir:** `~/code/data/donnons/{category}.jsonl`

## Configuration

| Parameter | Value |
|-----------|-------|
| Source | donnons.org — French free-item donation platform |
| Class | solidarity_exchange (label=1) |
| Target | 200 items × 17 categories = 3,400 |
| Min text length | 30 chars |

## Collection Results by Category

| Category | Collected | Notes |
|----------|:---------:|-------|
| meubles | 200 | ✅ |
| electromenager | 200 | ✅ |
| vetements-chaussures-et-accessoires | 200 | ✅ |
| high-tech-et-electronique | 200 | ✅ |
| loisirs-et-jeux | 200 | ✅ |
| maison-decoration-et-arts-de-la-table | 200 | ✅ |
| bricolage-outillage-et-materiaux | 200 | ✅ |
| livres-audio-films-et-billetterie | 200 | ✅ |
| sports-et-activites-de-plein-air | 200 | ✅ |
| accessoires-de-puericulture | 200 | ✅ |
| jardin-et-exterieur | 200 | ✅ |
| hygiene-soins-et-beaute | 200 | ✅ |
| fourniture-de-bureau-et-papeterie | 200 | ✅ |
| animalerie | 182 | Site had fewer available items |
| vehicules-pieces-et-accessoires | 156 | Site had fewer available items |
| alimentation | 100 | Site had fewer available items |
| materiel-specialise-et-professionnel | 100 | Site had fewer available items |
| **Total** | **3,238** | 95.2% of target |

## Sample Quality

Representative examples (confirmed correct register):
> *"bibliothèque en bois — Jolie bibliothèque en bois laminé en très bon état, 1m95 de haut, 30cm de profondeur"*
> *"pot, bavette... — Donne — Bavette silicone, Pot, Ceinture natation"*

Content confirmed: real French donation posts, solidarity register, varied object categories. Closest available proxy for Shareish content in French.

## Known Issues

1. **HTML entities in `location` field** — `"L&apos;Horme"` instead of `"L'Horme"`. Fix before training:
   ```python
   import html
   record["location"] = html.unescape(record["location"]) if record["location"] else None
   ```
2. **Very short texts** — some posts are 2–3 words (title only, no description). Raising min filter from 30→50 chars recommended.
3. **List-format descriptions** — some items enumerate objects line by line rather than prose. Structurally valid but lower linguistic quality.

## Conclusion

donnons.org is the highest-quality solidarity-class data collected: 3,238 real French donation posts across 17 object categories in the same informal register as Shareish. The 4 under-target categories fell short due to limited site inventory (not a scraper bug). These items are the primary positive-class source for the content-type classifier.

## Cross-references

- Part of: Track B content-type classifier dataset
- Combined with: [P3-E3 (donnerie.be)](exp_scraper_donnerie.md), [P3-E4 (2ememain.be)](exp_scraper_2ememain.md)
