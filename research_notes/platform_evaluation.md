# Platform Evaluation — Data Collection for Shareish Content-Type Classifier
**Phase 2, Track B** | Last updated: 2026-04-15 | Author: Seyfullah Ural

## Context

The Shareish content-type classifier requires two classes of training data:

| Class | Label | Description |
|---|---|---|
| **solidarity_exchange** | 1 | Offers/requests for goods or services with community/mutual-aid framing — *the Shareish distribution* |
| **commercial_listing** | 0 | Transactional, for-profit listings of similar objects — same vocabulary, different intent |

The goal of this document is to identify, for each candidate platform:
1. **Ethics** — does the platform allow scraping? Is the data GDPR-compliant for academic NLP use?
2. **Feasibility** — how accessible is the data technically?
3. **Relevance** — how close is the content distribution to Shareish posts?

---

## Comparison Tables

### Ethics & access — can we use it?

| Platform | robots.txt | AI-training clause | Login / JS | Descriptions | **Verdict** |
|---|---|---|---|---|---|
| **donnerie.be** | ✅ Listings allowed | None | No / No | Static HTML | ✅ Clear |
| **donnons.org** | ✅ Listings allowed | None | No / No | JSON-LD in `<script>` | ✅ Clear |
| **2ememain.be** | ✅ Explicitly allowed | None | No / No | Schema.org, static HTML | ✅ Clear |
| **troc.com** | ✅ Allowed | None | No / Partial | Templated text | ✅ Clear (low quality) |
| **brussels.craigslist.org** | ✅ Allowed | None | No / No | Static HTML | ✅ Clear (low quality) |
| **trashnothing.com** | ✅ No restrictions | None | **Yes** / likely | Behind login | ❌ Inaccessible |
| **Murmurations network** | ✅ Public API | None | No / No | Via API (no FR/BE nodes) | ⚠️ Wrong content |
| **vinted.fr** | ✅ Allowed | **Explicit ban on training data** | No / **Yes** | JS-only (Playwright) | ⚠️ Risky + hard |
| **facebook.com/marketplace** | ❌ `Disallow: /` | Implied by ToS | **Yes** / Yes | Behind login | ❌ Prohibited |
| **Facebook groups** | ❌ `Disallow: /` | Implied by ToS | **Yes** / Yes | Behind login | ❌ Prohibited |
| **leboncoin.fr** | ❌ Explicit written ban | — | No / No | HTTP 403 blocked | ❌ Prohibited |
| **pap.fr** | ❌ Listings disallowed | — | No / No | Blocked by robots.txt | ❌ Prohibited |

### Content fit — should we use it?

| Platform | Class | Language | Country | Content type | Register | **Relevance** |
|---|---|---|---|---|---|---|
| **donnerie.be** | Solidarity (+) | French | 🇧🇪 Belgium | Free giveaways | "à donner, gratuit" — personal | ⭐⭐⭐ Best match |
| **donnons.org** | Solidarity (+) | French | 🇫🇷 France | Free giveaways | "Je donne un…" — personal | ⭐⭐⭐ Excellent |
| **2ememain.be** | Commercial (−) | French | 🇧🇪 Belgium | Second-hand sales | "je vends" — price-tagged | ⭐⭐⭐ Best negative class |
| **facebook.com/marketplace** | Mixed | FR / NL | 🇧🇪 Belgium | Sales + free giveaways | Personal, "à donner" mixed | ⭐⭐⭐ Ideal — inaccessible |
| **vinted.fr** | Commercial (−) | French | 🇫🇷 France | Second-hand (clothing-heavy) | Short titles, price-tagged | ⭐⭐ Narrow domain |
| **leboncoin.fr** | Commercial (−) | French | 🇫🇷 France | General classifieds | Personal — price-tagged | ⭐⭐ Good — inaccessible |
| **Brussels Craigslist** | Mixed | FR / EN / NL | 🇧🇪 Belgium | General classifieds | Inconsistent | ⭐ Noisy |
| **troc.com** | Commercial (−) | French | 🇫🇷 France | Store second-hand | CMS template, not natural | ⭐ Templated |
| **trashnothing.com** | Solidarity (+) | English | 🌍 International | Free giveaways | "free to a good home" | ⭐ Wrong language |
| **Murmurations network** | Solidarity (+) | English | 🌍 International | Mutual aid | Solidarity-oriented | ⭐ No FR/BE nodes |
| **Facebook groups** | Mixed | FR / NL | 🇧🇪 Belgium | Community discussion | Broad, inconsistent | ⭐⭐ Inaccessible |
| **pap.fr** | Commercial (−) | French | 🇫🇷 France | Real estate only | Formal | ⭐ Wrong domain |

---

## Detailed Assessments

---

### donnerie.be ⭐⭐⭐ — Priority target (solidarity, positive class)

**URL:** https://donnerie.be/annonces/

**What it is:** Belgian free-item donation platform ("Le meilleur débarras en ligne de Belgique"). Posts are free giveaways — items people want to give away rather than sell, exactly the solidarity-exchange register of Shareish. Country of origin matches Shareish (Belgium), so linguistic register (Belgian French) is the best possible match.

**Sample content:**
> "Lot complet pour broder, tissus fil DMC et autre métallique..." — gratuit
> "Canape a donner en cuir de couleur bordeau, 3+2+1 places, à enlever..." — gratuit
> "Meubles de cuisine ayant été utilisés mais restant présentables." — gratuit

**Ethics:**
- robots.txt: no blocking of listing pages (whitelist approach for known search engines, implicit allow for general crawlers)
- GDPR: posts are public, no personal data retained if location stripped. Standard academic-use exemption applies (Art. 89 GDPR). Belgian platform, Belgian research institution — aligned jurisdiction.
- No explicit AI-training prohibition found.

**Feasibility:**
- Category browsing pages with pagination: `/annonces/page/N/`
- Individual item pages: `/annonces/{item-slug}/`
- Descriptions present in **static HTML** (no JS rendering needed)
- 140 pages of listings × ~10 items/page ≈ ~1,400 items available
- `requests` + `BeautifulSoup` sufficient

**Relevance:** Highest among all platforms. Belgian French, free-item solidarity framing, short to medium length descriptions of physical objects — identical register to Shareish.

**Action:** Build scraper. Priority over donnons.org for the positive class.

---

### donnons.org ⭐⭐⭐ — Already built (solidarity, positive class)

**URL:** https://www.donnons.org/catalogue

**What it is:** French free-item donation platform. Near-identical content to Shareish exchanges.

**Ethics:**
- robots.txt: whitelists major search engines, no blanket disallow on listings
- GDPR: public posts, academic use exemption
- No AI-training prohibition

**Feasibility:**
- Full descriptions in `<script type="application/ld+json">` (JSON-LD, static HTML)
- Pagination: `?page=N&sort=date`
- 17 categories, ~2,300+ items per category
- Scraper already implemented: `code/scrape_donnons.py`

**Relevance:** Excellent. French solidarity exchanges with same vocabulary and framing as Shareish. Slightly lower geographic relevance than donnerie.be (France vs Belgium) but far more data volume.

**Action:** Already implemented. Run to collect ~3,400 items (200/category × 17 categories).

---

### 2ememain.be ⭐⭐⭐ — Priority target (commercial, negative class)

**URL:** https://www.2ememain.be/

**What it is:** Belgian second-hand marketplace — the Belgian equivalent of Leboncoin. Users sell (not give) used items. Commercially framed, price-tagged listings in French. This is the ideal negative class: same object vocabulary as Shareish (sofas, bikes, appliances) but explicitly transactional rather than solidarity-oriented.

**Sample content:**
> "Je déménage et vends mon meuble tv bestå de chez ikea, utilisé dans mon salon. Meuble pratique avec rangements fermés."
> "Buffet en teck massif – fabrication artisanale, très bon état. €400."
> "Service lift + camionnette – Déménagement & transport"

**Ethics:**
- robots.txt: listing/annonce pages explicitly **allowed**. Only blocks messaging, auth, flagging, and search-parameter URLs.
- GDPR: public listings, no personal data if seller name stripped. Academic use.
- No AI-training prohibition found.

**Feasibility:**
- Full descriptions in **static HTML** (Schema.org structured data embedded in page)
- French language (2ememain.be = French side; 2dehands.be = Dutch)
- Standard pagination
- `requests` + `BeautifulSoup` sufficient; no Playwright needed

**Relevance:** Excellent for the negative class. Belgian French, second-hand objects, similar length/format to Shareish posts but commercial intent clearly distinguishable through pricing language ("je vends", prices, delivery fees).

**Action:** Build scraper. Best available source for commercial negative-class examples.

---

### vinted.fr ⭐⭐ — Backup (commercial, negative class)

**URL:** https://www.vinted.fr/catalog

**What it is:** Second-hand clothing and household items marketplace. Commercial, price-tagged, French.

**Ethics:**
- robots.txt: catalog pages crawling allowed
- **Explicit AI-training prohibition:** *"Use of any content for model training, fine-tuning, dataset creation, or embedding is strictly prohibited"*
- This prohibition makes Vinted ethically problematic for this use case. Must be disclosed if used.

**Feasibility:**
- Catalog pages: titles, condition, price visible in static HTML (~50 items/page)
- **Full descriptions require Playwright** (JS-rendered, API returns 401 without session cookie)
- More engineering effort than 2ememain.be

**Relevance:** Good for commercial class but mostly fashion/clothing items — narrower domain than 2ememain which covers all household goods.

**Action:** Skip in favour of 2ememain.be unless more commercial data is needed. If used, disclose ToS limitation in thesis.

---

### trashnothing.com ⭐ — Skip (solidarity, positive class)

**URL:** https://trashnothing.com/browse

**What it is:** International freecycling network (free item giveaways), similar model to donnons.org.

**Ethics:**
- robots.txt: no disallow rules — fully open to crawling
- GDPR: mostly non-EU users; mixed jurisdiction

**Feasibility:**
- Browse page returns only Bootstrap CSS without actual post content — posts likely behind a **login wall** or JavaScript-loaded
- Content not accessible without authentication

**Relevance:** Predominantly **English**. French posts exist but are a minority. Low signal for a French-language classifier.

**Action:** Skip. Login wall + English-dominant content makes it unsuitable.

---

### Murmurations Network ⭐ — Skip (solidarity, positive class)

**URL:** https://murmurmaps.murmurations.network / https://index.murmurations.network/v2/nodes

**What it is:** Federated mutual aid and community offers/wants directory. Decentralised protocol — data is stored on individual community nodes, indexed centrally.

**Ethics:**
- Public API, no authentication required
- Open protocol designed for discoverability — scraping is by design

**Feasibility:**
- Index API returns 88,163 nodes with `offers_wants_schema` — but actual content is at `profile_url` per node
- **No Belgian or French nodes found** in the index (queried `country=BE` → empty result)
- Individual profile URLs often return 403 (node-level access control)

**Relevance:** Near zero for this project. Content is predominantly English and US/UK community groups. No French solidarity exchange content found.

**Action:** Skip.

---

### brussels.craigslist.org ⭐ — Skip (mixed class)

**URL:** https://brussels.craigslist.org/search/sss

**What it is:** Craigslist for Brussels — general buy/sell/free classifieds.

**Ethics:**
- robots.txt: only blocks /reply, /fb/, /suggest, /flag — listings allowed
- Public platform, no explicit scraping prohibition

**Feasibility:**
- Static HTML, content visible without JS
- Description text present in page

**Relevance:** Very low. Spot-check showed posts in mixed languages (English, French, Dutch) and extremely heterogeneous content (hotel listings, inappropriate titles). No consistent language or topic distribution. Would introduce noise into the classifier.

**Action:** Skip.

---

### facebook.com/marketplace — Dead end (commercial + solidarity, mixed class)

**URL:** https://www.facebook.com/marketplace/brussels/

**What it is:** Meta's peer-to-peer marketplace. In Belgium it is heavily used for both commercial sales and free giveaways ("à donner"). Posts are in French, Dutch, or mixed depending on the seller. Structurally very close to Shareish: short personal descriptions of household objects, community tone, local pickup.

**Why it was worth checking:** The free-item section of Brussels Marketplace is essentially the same behaviour as Shareish — "Je donne un canapé, à récupérer à Ixelles" — making it a candidate for either positive or negative class depending on the post.

**Ethics:**
- robots.txt **explicitly states**: *"Collection of data on Facebook through automated means is prohibited unless you have express written permission from Facebook"*
- All general crawlers receive `Disallow: /` — blanket prohibition
- Meta actively litigates against scrapers (hiQ Labs v. LinkedIn precedent does not apply here — Meta has won on ToS grounds)
- GDPR risk: posts may contain personal data (full name, profile photo) embedded in the page even if the listing itself is public

**Feasibility:**
- Hard login wall — the Brussels Marketplace page returns only an empty React shell with no listing content for unauthenticated requests
- No public API for marketplace listings
- Even with a logged-in session (cookie injection), scraping would violate ToS and risk account ban

**Relevance:** Would be excellent in theory (Belgian French, exactly the right register), but entirely inaccessible in practice.

**Action:** Dead end. Do not attempt. If this distribution is specifically needed, generate synthetic examples using the same LLM pipeline as Track A.

---

### leboncoin.fr — Dead end

**Ethics:** robots.txt **explicitly states**: *"It is forbidden to use search robots or other automatic methods to access Leboncoin.fr"*

**Feasibility:** All pages return HTTP 403. Active bot detection.

**Action:** Do not attempt.

---

### pap.fr — Dead end

**Ethics:** robots.txt disallows most listing/annonce URL patterns for general crawlers.

**Feasibility:** Listings largely blocked at robots.txt level.

**Action:** Do not attempt.

---

### troc.com — Low value

**Ethics:** robots.txt only disallows `/ajax/` — listings allowed.

**Feasibility:** Accessible, but item descriptions follow the templated pattern: *"Achetez votre [item] occasion au meilleur prix : [price]€, disponible dans le magasin Troc [city]."*

**Relevance:** These descriptions are not natural user-written text — they are CMS-generated templates. A classifier trained on this would learn template formatting, not natural language solidarity vs. commercial intent.

**Action:** Skip in favour of 2ememain.be.

---

## Recommended Data Collection Plan

| Priority | Platform | Class | Target items | Method |
|---|---|---|---|---|
| 1 | **donnerie.be** | Solidarity (+1) | 500–1,000 | Build scraper (static HTML) |
| 2 | **2ememain.be** | Commercial (0) | 500–1,000 | Build scraper (Schema.org) |
| 3 | **donnons.org** | Solidarity (+1) | 3,400 (built) | Already done |
| 4 | Shareish (supervisor) | Solidarity (+1) | TBD | Awaiting data |

This gives:
- ~4,000–5,000 solidarity-class examples (donnerie.be + donnons.org + Shareish)
- ~500–1,000 commercial-class examples (2ememain.be)

---

## GDPR Note

All platforms targeted collect public posts with no retained personal identifiers (names, profile images stripped at collection time). Academic research exemption applies under GDPR Art. 89. Data used solely for training a classifier within the thesis scope, not redistributed or published raw.
