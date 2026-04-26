# Phase 3 Results Report — Data Collection and Generation

**Date:** 2026-04-17
**Scope:** Track A (synthetic hate speech) + Track B (content-type classifier data)
**Scripts:**
- `code/phase3_data/generate_synthetic_data.py` — Track A
- `code/phase3_data/scrape_donnons.py` — Track B (solidarity, FR)
- `code/phase3_data/scrape_donnerie.py` — Track B (solidarity, BE) ← rerunning
- `code/phase3_data/scrape_2ememain.py` — Track B (commercial, BE) ← needs fix
- `code/phase3_data/generate_classifier_data.py` — Track B supplement ← not yet run

**Data dirs:** `~/code/data/synthetic/`, `~/code/data/donnons/`, `~/code/data/donnerie/`, `~/code/data/2ememain/`

---

## Summary

| Source | Track | Class | Label | Target | Collected | Status |
|---|---|---|---|---|---|---|
| generate_synthetic_data.py | A | hate speech | 1 or 0 | 1,500 (300×5) | **1,500** | ✅ Complete |
| scrape_donnons.py | B | solidarity_exchange | 1 | 3,400 (200×17) | **3,238** | ✅ Complete |
| scrape_donnerie.py | B | solidarity_exchange | 1 | 1,000 | **0** | ❌ Rerunning (bug fixed) |
| scrape_2ememain.py | B | commercial_listing | 0 | 800 | **51** | ❌ Needs scraper fix |
| generate_classifier_data.py | B | solidarity + commercial | 1 / 0 | 600 (300×2) | **0** | ⏳ Not yet submitted |

**Total collected: 4,789 items** (4,738 usable pending cleaning)

---

## Track A — Synthetic French Hate Speech

### Counts

| Functionality | Label | Collected | Time |
|---|---|---|---|
| slur_h | 1 | 300 | 336s |
| spell_leet_h | 1 | 300 | 208s |
| spell_char_del_h | 1 | 300 | 196s |
| derog_impl_h | 1 | 300 | 245s |
| counter_quote_nh | 0 | 300 | 313s |
| **Total** | | **1,500** | **~22 min** |

Model: `mistral-7b-instruct-v0.3` on A5000. All 5 functionalities hit target exactly.

### Quality assessment

**slur_h — Good.** Explicit slurs targeting various identity groups, realistic social media register:
> *"Quand on voit un Noir, il faut savoir qu'il vole par nature !"*

**counter_quote_nh — Good.** Counter-speech framing consistently correct:
> *"L'antisémitisme viole l'universalité de notre humanité commune. Il doit être combattu avec force."*
> *(Minor spelling error "combatut" in one sample — acceptable noise)*

**spell_char_del_h — Good.** Character deletion/obfuscation confirmed in sampled examples.

**derog_impl_h — Mixed.** Some examples are genuinely implicit:
> *"Ils ont un sens de l'organisation totalement différent de nous, on devrait en prendre compte."*
But others drifted to neutral text with no derogatory content whatsoever:
> *"C'était tout naturellement une décision compliquée pour lui, étant donné sa nature impulsive."*
**⚠️ This functionality requires a manual review pass before use in fine-tuning.**

**spell_leet_h — Mixed.** Some examples correctly obfuscate, but a non-trivial proportion contains no leet-speak at all — the model produced plain hate speech instead:
> *"Ton genre fait peur aux enfants."* ← no obfuscation present
**⚠️ Recommend filtering: keep only examples containing at least one of `[@$3!01€]` before use.**

### Known data issue — numbering artifact

All examples carry a `"1.1. "` prefix from a list-parsing bug. **Must strip before training:**
```python
import re
record["text"] = re.sub(r"^\d+\.\d*\s*", "", record["text"])
```

### Usability verdict

| Functionality | Usable as-is | Action required |
|---|---|---|
| slur_h | ✅ After artifact strip | Strip "1.1." prefix |
| spell_char_del_h | ✅ After artifact strip | Strip "1.1." prefix |
| counter_quote_nh | ✅ After artifact strip | Strip "1.1." prefix |
| spell_leet_h | ⚠️ Partial | Strip prefix + filter for obfuscation chars |
| derog_impl_h | ⚠️ Partial | Strip prefix + manual review (100-sample spot check) |

---

## Track B — Content-Type Classifier Data

### donnons.org — 3,238 items ✅

**What it is:** French free-item donation platform. Posts are solidarity exchanges — the closest available proxy for Shareish content in French.

**Counts by category:**

| Category | Collected | Notes |
|---|---|---|
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
| animalerie | 182 | Site had fewer items |
| vehicules-pieces-et-accessoires | 156 | Site had fewer items |
| alimentation | 100 | Site had fewer items |
| materiel-specialise-et-professionnel | 100 | Site had fewer items |
| **Total** | **3,238** | |

**Quality:** Confirmed correct — real French donation posts, solidarity register, varied object categories. Representative sample:
> *"bibliothèque en bois\nJolie bibliothèque en bois laminé en très bon état, 1m95 de haut, 30cm de profondeur"*
> *"pot, bavette...\nDonne — Bavette silicone, Pot, Ceinture natation"*

**Known issues:**
1. **HTML entities in `location` field** — `"L&apos;Horme"` instead of `"L'Horme"`. Fix:
   ```python
   import html
   record["location"] = html.unescape(record["location"]) if record["location"] else None
   ```
2. **Very short texts** — Some posts are 2–3 words (title only, no description). Minimum length filter (30 chars already applied) retains some borderline cases. Consider raising to 50 chars for training.
3. **List-format descriptions** — Some items enumerate objects line by line rather than prose. Structurally valid but lower linguistic quality.

---

### donnerie.be — 0 items ❌

**What it is:** Belgian French free-item donation platform. Highest-relevance source — same country and language register as Shareish.

**Failure cause:** Scraper checked for relative hrefs (`/annonces/slug/`) but donnerie.be uses absolute hrefs (`https://donnerie.be/annonces/slug/`). Parser matched zero URLs across all 140 catalog pages. Ran for 4,419s producing nothing.

**Fix applied:** `parse_item_urls()` now normalises hrefs to absolute before matching. Resubmitted 2026-04-17. Results pending.

---

### 2ememain.be — 51/800 items ❌

**What it is:** Belgian French second-hand marketplace. Primary source for commercial_listing (label=0) negative class.

**Failure cause:** Two compounding issues:
1. **Dutch language filtering too aggressive** — 2ememain.be serves both French and Dutch speakers. The Dutch condition-word filter (`"gebruikt"`, `"nieuw"` etc.) combined with langdetect discarded ~90% of listings from even French-dominant categories. Only 51/450 expected listings passed.
2. **Descriptions truncated** — JSON-LD `description` field is cut off in static HTML. The `window.__CONFIG__` fallback did not parse reliably.

**Sample quality** (the 51 items that did pass): Good. Commercial framing confirmed:
> *"Enfilade scandinave vintage en teck 4 — Très jolie enfilade / commode vintage en teck, typique du design scandinave des années 60."*

**Required fixes:**
- Remove Dutch condition-word filter; rely on langdetect alone
- Fix `window.__CONFIG__` description extraction
- Or: switch to fetching only explicitly French-language subcategory pages

**Interim solution:** `generate_classifier_data.py` fills the commercial class gap with synthetic examples while the scraper fix is developed.

---

## Class Balance — Current State

| Class | Label | Source | Count |
|---|---|---|---|
| solidarity_exchange | 1 | donnons.org | 3,238 |
| solidarity_exchange | 1 | donnerie.be | 0 (pending) |
| solidarity_exchange | 1 | Shareish (supervisor) | TBD |
| commercial_listing | 0 | 2ememain.be | 51 |
| **Total** | | | **3,289** |

**Current imbalance: 64:1 solidarity:commercial.** The classifier cannot be trained until the commercial class is substantially increased. Required actions in priority order:

1. Fix and rerun `scrape_2ememain.py` → target 800 items
2. Run `generate_classifier_data.py` → +300 synthetic commercial examples
3. Receive donnerie.be rerun results → +1,000 solidarity examples
4. Receive Shareish data from supervisor → final positive-class gold standard

Minimum viable training set: ~1,000 solidarity + ~500 commercial (2:1 ratio).

---

## Pre-Training Cleaning Checklist

- [ ] Strip `"1.1. "` artifacts from all synthetic text fields
- [ ] Strip HTML entities from donnons.org `location` fields
- [ ] Filter `spell_leet_h`: keep only examples containing obfuscation characters (`[@$3!01€]`)
- [ ] Manual spot-check of `derog_impl_h` (review 100 random samples)
- [ ] Raise minimum text length to 50 chars (from 30) for donnons.org
- [ ] Wait for donnerie.be rerun
- [ ] Fix and rerun `scrape_2ememain.py`
- [ ] Submit `generate_classifier_data.py`
- [ ] Receive Shareish data from supervisor
- [ ] Merge all sources → `classifier_train.jsonl` with 80/10/10 train/val/test split

---

## Next Steps

| Priority | Action | Unblocks |
|---|---|---|
| 1 | Fix `scrape_2ememain.py` (Dutch filter + description extraction) | Commercial class volume |
| 2 | Submit `generate_classifier_data.py` | Commercial class interim coverage |
| 3 | Wait for donnerie.be rerun | Belgian solidarity data |
| 4 | Contact supervisor re Shareish data | Gold standard positive class |
| 5 | Apply cleaning checklist | Training-ready dataset |
| 6 | Train content-type classifier (DistilBERT or XLM-RoBERTa) | Phase 4 data filtering |
