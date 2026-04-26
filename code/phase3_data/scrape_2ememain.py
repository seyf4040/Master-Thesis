#!/usr/bin/env python3
"""
scrape_2ememain.py — Scrape 2ememain.be for commercial-listing training data (Phase 2, Track B)

2ememain.be is the Belgian French second-hand marketplace (equivalent of Leboncoin).
Posts are commercially framed ("Je vends mon canapé, €80, à enlever sur place") —
same object vocabulary as Shareish but with clear transactional intent.

This provides the negative class (label=0) for the Shareish content-type classifier.

Pagination is infinite-scroll based (no URL parameters). Strategy: collect item URLs
from the first page of each target subcategory (30 items/page × N subcategories),
then fetch individual listing pages for full descriptions.

Full descriptions are in JSON-LD (schema.org Product) in static HTML.
French-language filter applied via langdetect (2ememain.be also serves Dutch speakers).

Label: 0 = commercial_listing

Output JSONL fields:
    text, label, source, url, category, location, price_eur, condition, timestamp

Usage:
    python code/phase3_data/scrape_2ememain.py \\
        --output_dir ~/code/data/2ememain \\
        --max_items 800 \\
        --delay 2.0

    # Quick test (3 categories only):
    python code/phase3_data/scrape_2ememain.py \\
        --output_dir /tmp/2ememain_test \\
        --categories maison-meubles,electromenager,enfants-bebes \\
        --max_items 30 \\
        --delay 1.5
"""

import json
import time
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = "https://www.2ememain.be"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research-bot/1.0; "
        "master-thesis content-moderation ULiege; respectful-crawl)"
    ),
    "Accept-Language": "fr-BE,fr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Target categories: household-domain items that overlap with Shareish vocabulary.
# These are the French-side categories most relevant for the negative class.
DEFAULT_CATEGORIES = [
    "maison-meubles",
    "electromenager",
    "enfants-bebes",
    "jardins-terrasses",
    "vetements-femmes",
    "vetements-hommes",
    "bricolage",
    "sports-fitness",
    "informatique",
    "tv-hi-fi-video",
    "livres-cd-dvd",
    "velos",
    "arts-antiquites",
    "telephonie",
    "instruments-de-musique",
]

# Dutch condition words — signal a Dutch-language listing, skip it
_DUTCH_SIGNALS = {"gebruikt", "nieuw", "zo goed als nieuw", "beschadigd", "goed"}


# ── HTTP ──────────────────────────────────────────────────────────────────────

def fetch(session: requests.Session, url: str, delay: float) -> Optional[str]:
    time.sleep(delay)
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        elif resp.status_code == 404:
            return None
        else:
            log.warning(f"HTTP {resp.status_code} — {url}")
            return None
    except requests.RequestException as e:
        log.warning(f"Request error ({url}): {e}")
        return None


# ── Language check ────────────────────────────────────────────────────────────

def is_french(text: str) -> bool:
    """Return True if langdetect identifies the text as French."""
    try:
        from langdetect import detect
        return detect(text) == "fr"
    except Exception:
        # Accept short strings we can't classify confidently
        return len(text.split()) >= 4


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_listing_urls(html: str) -> List[Tuple[str, str]]:
    """
    Extract (url, subcategory) tuples from a category listing page.
    Listing URLs follow: /v/{category}/{subcategory}/{id}-{slug}
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/v/") and href.count("/") >= 4:
            full = urljoin(BASE_URL, href)
            if full not in seen:
                seen.add(full)
                # Extract subcategory from path: /v/{cat}/{subcat}/{id}-{slug}
                parts = href.strip("/").split("/")
                subcat = parts[2] if len(parts) > 2 else "unknown"
                results.append((full, subcat))
    return results


def parse_listing_page(html: str, url: str, category: str) -> Optional[Dict]:
    """
    Extract item data from a 2ememain.be individual listing page.
    Description is in JSON-LD Product schema.
    """
    soup = BeautifulSoup(html, "html.parser")

    title = None
    description = None
    price_eur = None
    location = None
    condition = None

    # ── JSON-LD (primary) ─────────────────────────────────────────────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            schemas = data if isinstance(data, list) else [data]
            for schema in schemas:
                if schema.get("@type") not in ("Product", "product"):
                    continue
                desc = (schema.get("description") or "").strip()
                name = (schema.get("name") or "").strip()
                if not desc or len(desc) < 15:
                    continue

                title = name
                description = desc

                # Price
                offers = schema.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                price_raw = offers.get("price") or schema.get("price")
                if price_raw is not None:
                    try:
                        price_eur = float(str(price_raw).replace(",", "."))
                    except ValueError:
                        pass

                # Condition (may be Dutch: "Gebruikt" or French: "Utilisé")
                cond_raw = (
                    schema.get("itemCondition", "")
                    or offers.get("itemCondition", "")
                )
                if cond_raw:
                    condition = cond_raw.split("/")[-1]

                break
        except (json.JSONDecodeError, AttributeError):
            continue

    # ── window.__CONFIG__ fallback ────────────────────────────────────────
    if not description:
        for script in soup.find_all("script"):
            text = script.string or ""
            if "window.__CONFIG__" in text and '"description"' in text:
                try:
                    start = text.index("window.__CONFIG__") + len("window.__CONFIG__") + 3
                    # Find matching brace
                    brace_depth = 0
                    end = start
                    for i, ch in enumerate(text[start:], start):
                        if ch == "{":
                            brace_depth += 1
                        elif ch == "}":
                            brace_depth -= 1
                            if brace_depth == 0:
                                end = i + 1
                                break
                    config = json.loads(text[start:end])
                    listing = config.get("listing", {})
                    description = (listing.get("description") or "").strip()
                    title = (listing.get("title") or title or "").strip()
                    loc = listing.get("location", {})
                    location = loc.get("cityName")
                except (ValueError, KeyError):
                    pass

    if not description:
        return None

    # ── Location from JSON-LD or page text ────────────────────────────────
    if location is None:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                schemas = data if isinstance(data, list) else [data]
                for schema in schemas:
                    loc = (
                        schema.get("offers", {}).get("availableAtOrFrom", {}).get("name")
                        or schema.get("location", {}).get("name")
                    )
                    if loc:
                        location = loc
                        break
            except Exception:
                continue

    # Skip Dutch-language listings
    cond_lower = (condition or "").lower()
    if any(signal in cond_lower for signal in _DUTCH_SIGNALS):
        return None

    # Build text: title + description
    if title and title.lower() not in description.lower():
        text = f"{title}\n{description}"
    else:
        text = description

    if not is_french(text):
        return None

    return {
        "text": text,
        "label": 0,
        "source": "2ememain_be",
        "url": url,
        "category": category,
        "location": location,
        "price_eur": price_eur,
        "condition": condition,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Validation ────────────────────────────────────────────────────────────────

def is_valid(item: Dict, seen_hashes: Set[str], min_chars: int = 30) -> bool:
    text = (item.get("text") or "").strip()
    if len(text) < min_chars:
        return False
    h = hashlib.sha256(text.lower().encode()).hexdigest()
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True


# ── Checkpointing ─────────────────────────────────────────────────────────────

def load_scraped_urls(output_dir: Path) -> Set[str]:
    f = output_dir / "scraped_urls.txt"
    return set(f.read_text().splitlines()) if f.exists() else set()


def save_scraped_urls(output_dir: Path, urls: Set[str]) -> None:
    (output_dir / "scraped_urls.txt").write_text("\n".join(sorted(urls)))


def count_existing(output_dir: Path) -> int:
    f = output_dir / "2ememain.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def append_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Core scrape loop ──────────────────────────────────────────────────────────

def scrape(
    categories: List[str],
    max_items: int,
    output_dir: Path,
    session: requests.Session,
    delay: float,
) -> int:
    out_file = output_dir / "2ememain.jsonl"
    scraped_urls = load_scraped_urls(output_dir)
    seen_hashes: Set[str] = set()
    already = count_existing(output_dir)

    if already >= max_items:
        log.info(f"Already have {already}/{max_items} items. Nothing to do.")
        return already

    remaining = max_items - already
    log.info(f"Need {remaining} more items (have {already}/{max_items}).")
    log.info(f"Collecting from {len(categories)} categories.")

    # ── Phase 1: collect listing URLs from category pages ─────────────────
    # Since pagination is infinite-scroll, we take the first page of each
    # category (30 items). For more items, we can cycle through categories
    # multiple times or add subcategory pages.
    all_listing_urls: List[Tuple[str, str]] = []  # (url, category)
    for cat in categories:
        cat_url = f"{BASE_URL}/l/{cat}/"
        html = fetch(session, cat_url, delay)
        if not html:
            log.warning(f"Could not fetch category page: {cat}")
            continue
        urls = [(u, cat) for u, _ in parse_listing_urls(html) if u not in scraped_urls]
        log.info(f"  {cat}: {len(urls)} item URLs found.")
        all_listing_urls.extend(urls)

    log.info(f"Total item URLs to fetch: {len(all_listing_urls)}")

    if not all_listing_urls:
        log.warning("No item URLs collected. Exiting.")
        return already

    # ── Phase 2: fetch individual listing pages ───────────────────────────
    collected = 0
    buffer: List[Dict] = []

    for url, category in all_listing_urls:
        if collected >= remaining:
            break

        html = fetch(session, url, delay)
        scraped_urls.add(url)

        if not html:
            continue

        item = parse_listing_page(html, url, category)
        if item and is_valid(item, seen_hashes):
            buffer.append(item)
            collected += 1

        if len(buffer) >= 20:
            append_jsonl(out_file, buffer)
            save_scraped_urls(output_dir, scraped_urls)
            log.info(f"  {already + collected}/{max_items} saved.")
            buffer.clear()

    if buffer:
        append_jsonl(out_file, buffer)
        save_scraped_urls(output_dir, scraped_urls)

    total = already + collected
    log.info(f"Done. {collected} new items written (total: {total}).")
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Scrape 2ememain.be for Belgian French commercial listing data."
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--categories",
        default="all",
        help=(
            "Comma-separated category slugs or 'all'. "
            f"Default: all ({len(DEFAULT_CATEGORIES)} categories)."
        ),
    )
    p.add_argument(
        "--max_items", type=int, default=800,
        help="Stop after collecting this many items (default: 800).",
    )
    p.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between requests (default: 2.0).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.categories == "all":
        categories = DEFAULT_CATEGORIES
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    log.info("=" * 55)
    log.info("2ememain.be scraper — Phase 2, Track B")
    log.info(f"  Categories : {len(categories)}")
    log.info(f"  Max items  : {args.max_items}")
    log.info(f"  Delay      : {args.delay}s")
    log.info(f"  Output     : {output_dir}")
    log.info("=" * 55)

    session = requests.Session()
    t0 = time.time()
    n = scrape(categories, args.max_items, output_dir, session, args.delay)
    elapsed = time.time() - t0

    log.info(f"Total items: {n} in {elapsed/60:.1f} min")

    manifest = {
        "source": "2ememain_be",
        "label": 0,
        "label_name": "commercial_listing",
        "categories": categories,
        "max_items": args.max_items,
        "delay_s": args.delay,
        "n_collected": n,
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
