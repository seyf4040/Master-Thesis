#!/usr/bin/env python3
"""
scrape_donnons.py — Scrape donnons.org for solidarity-exchange training data (Phase 2, Track B)

Donnons.org is a French free-item donation platform. Posts are structurally and
linguistically identical to Shareish solidarity exchanges ("Je donne un lit 140x190,
cause déménagement, à venir récupérer sur place…"), making them ideal positive-class
training data for the Shareish content-type classifier.

Full item descriptions are embedded in JSON-LD structured data in the static HTML —
no JavaScript rendering or authentication required.

Label convention (for the content-type classifier):
    1 = solidarity_exchange (matches Shareish content)
    0 = off-domain / commercial  (Vinted, Reddit — scraped separately)

Output JSONL fields:
    text, label, category, source, url, location, condition, timestamp

Usage:
    python code/scrape_donnons.py \\
        --output_dir ~/code/data/donnons \\
        --categories all \\
        --max_per_category 200 \\
        --delay 2.0

    # Specific categories only:
    python code/scrape_donnons.py \\
        --output_dir ~/code/data/donnons \\
        --categories meubles,electromenager,vetements-chaussures-et-accessoires \\
        --max_per_category 300
"""

import json
import time
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://www.donnons.org"

# All categories available on donnons.org (sorted by relevance to Shareish)
ALL_CATEGORIES = [
    "meubles",
    "electromenager",
    "vetements-chaussures-et-accessoires",
    "high-tech-et-electronique",
    "loisirs-et-jeux",
    "maison-decoration-et-arts-de-la-table",
    "bricolage-outillage-et-materiaux",
    "livres-audio-films-et-billetterie",
    "sports-et-activites-de-plein-air",
    "accessoires-de-puericulture",
    "jardin-et-exterieur",
    "animalerie",
    "alimentation",
    "hygiene-soins-et-beaute",
    "fourniture-de-bureau-et-papeterie",
    "vehicules-pieces-et-accessoires",
    "materiel-specialise-et-professionnel",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research-bot/1.0; "
        "master-thesis content-moderation ULiege; respectful-crawl)"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def fetch(session: requests.Session, url: str, delay: float) -> Optional[str]:
    """Fetch URL with polite delay. Returns HTML string or None on failure."""
    time.sleep(delay)
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        elif resp.status_code == 404:
            return None  # item deleted, skip silently
        else:
            log.warning(f"HTTP {resp.status_code} for {url}")
            return None
    except requests.RequestException as e:
        log.warning(f"Request failed ({url}): {e}")
        return None


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_item_urls_from_catalog(html: str) -> List[str]:
    """Extract /don/{id} hrefs from a category listing page."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Item pages follow /don/{numeric_id}
        if href.startswith("/don/") and href[5:].isdigit():
            full_url = urljoin(BASE_URL, href)
            if full_url not in urls:
                urls.append(full_url)
    return urls


def parse_total_pages(html: str) -> int:
    """Extract total page count from pagination text like '2330 résultats (page 1 / 24)'."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(string=True):
        text = tag.strip()
        if "page" in text.lower() and "/" in text:
            # e.g. "2330 résultats (page 1 / 24)"
            try:
                parts = text.replace("(", "").replace(")", "").split("/")
                total = int(parts[-1].strip().split()[0])
                return total
            except (ValueError, IndexError):
                continue
    return 1


def parse_item_page(html: str, url: str, category: str) -> Optional[Dict]:
    """
    Extract item data from a donnons.org /don/{id} page.
    Description is embedded as JSON-LD structured data in the static HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    item = {
        "text": None,
        "label": 1,                   # solidarity_exchange
        "category": category,
        "source": "donnons_org",
        "url": url,
        "location": None,
        "condition": None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # ── Primary extraction: JSON-LD structured data ────────────────────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            # May be a list of schemas or a single object
            schemas = data if isinstance(data, list) else [data]
            for schema in schemas:
                desc = schema.get("description", "").strip()
                name = schema.get("name", "").strip()
                if desc and len(desc) > 10:
                    # Combine title + description into a single text field
                    # This mirrors how a Shareish post would appear
                    item["text"] = f"{name}\n{desc}".strip() if name else desc

                    # Location: may be in offers.availableAtOrFrom or address
                    loc = (
                        schema.get("offers", {}).get("availableAtOrFrom", {}).get("address", {}).get("addressLocality")
                        or schema.get("location", {}).get("name")
                        or schema.get("address", {}).get("addressLocality")
                    )
                    if loc:
                        item["location"] = loc

                    # Condition
                    cond = schema.get("itemCondition", "")
                    if cond:
                        item["condition"] = cond.split("/")[-1]  # strip schema.org URL

                    break
        except (json.JSONDecodeError, AttributeError):
            continue

    # ── Fallback: look for condition in visible HTML ───────────────────────
    if item["condition"] is None:
        for tag in soup.find_all(class_=lambda c: c and "condition" in c.lower()):
            item["condition"] = tag.get_text(strip=True)
            break

    # ── Location fallback: look for city mentions ──────────────────────────
    if item["location"] is None:
        for tag in soup.find_all(class_=lambda c: c and any(
            k in c.lower() for k in ["location", "ville", "city", "address"]
        )):
            text = tag.get_text(strip=True)
            if text:
                item["location"] = text
                break

    if not item["text"]:
        return None

    return item


# ── Validation ────────────────────────────────────────────────────────────────

def is_valid(item: Dict, seen_hashes: Set[str], min_chars: int = 30) -> bool:
    """Basic quality filters: minimum length and deduplication."""
    text = item.get("text", "") or ""
    if len(text) < min_chars:
        return False

    h = hashlib.sha256(text.lower().strip().encode()).hexdigest()
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True


# ── Checkpointing ─────────────────────────────────────────────────────────────

def load_scraped_ids(output_dir: Path, category: str) -> Set[str]:
    """Load previously scraped item URLs to avoid re-fetching."""
    checkpoint = output_dir / f"{category}_scraped_ids.txt"
    if checkpoint.exists():
        return set(checkpoint.read_text().splitlines())
    return set()


def save_scraped_ids(output_dir: Path, category: str, ids: Set[str]) -> None:
    checkpoint = output_dir / f"{category}_scraped_ids.txt"
    checkpoint.write_text("\n".join(sorted(ids)))


def count_existing(output_dir: Path, category: str) -> int:
    out_file = output_dir / f"{category}.jsonl"
    if not out_file.exists():
        return 0
    with out_file.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ── JSONL output ──────────────────────────────────────────────────────────────

def append_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Core scrape loop ──────────────────────────────────────────────────────────

def scrape_category(
    category: str,
    max_items: int,
    output_dir: Path,
    session: requests.Session,
    delay: float,
) -> int:
    """Scrape one category. Returns total items written."""
    already = count_existing(output_dir, category)
    if already >= max_items:
        log.info(f"[{category}] Already have {already}/{max_items}, skipping.")
        return already

    remaining = max_items - already
    scraped_ids = load_scraped_ids(output_dir, category)
    seen_hashes: Set[str] = set()
    out_file = output_dir / f"{category}.jsonl"
    collected = 0

    log.info(f"[{category}] Need {remaining} more items (have {already}/{max_items}).")

    # ── Phase 1: collect item URLs from catalog pages ──────────────────────
    catalog_base = f"{BASE_URL}/catalogue/categorie/{category}?sort=date&page="
    first_page_html = fetch(session, catalog_base + "1", delay)
    if not first_page_html:
        log.warning(f"[{category}] Could not fetch catalog page 1, skipping.")
        return already

    total_pages = parse_total_pages(first_page_html)
    log.info(f"[{category}] {total_pages} catalog pages available.")

    item_urls: List[str] = []
    item_urls.extend(parse_item_urls_from_catalog(first_page_html))

    page = 2
    while len(item_urls) < remaining * 3 and page <= total_pages:
        html = fetch(session, catalog_base + str(page), delay)
        if html:
            new_urls = parse_item_urls_from_catalog(html)
            item_urls.extend(new_urls)
            log.debug(f"[{category}] Page {page}: +{len(new_urls)} URLs (total {len(item_urls)})")
        page += 1

    # Remove already-scraped
    item_urls = [u for u in item_urls if u not in scraped_ids]
    # Deduplicate within this batch
    seen_urls = set()
    item_urls = [u for u in item_urls if not (u in seen_urls or seen_urls.add(u))]

    log.info(f"[{category}] {len(item_urls)} new item URLs to fetch.")

    # ── Phase 2: fetch individual item pages ──────────────────────────────
    buffer: List[Dict] = []
    for url in item_urls:
        if collected >= remaining:
            break

        html = fetch(session, url, delay)
        scraped_ids.add(url)

        if not html:
            continue

        item = parse_item_page(html, url, category)
        if item is None:
            continue

        if not is_valid(item, seen_hashes):
            continue

        buffer.append(item)
        collected += 1

        # Flush every 20 items
        if len(buffer) >= 20:
            append_jsonl(out_file, buffer)
            save_scraped_ids(output_dir, category, scraped_ids)
            log.info(f"[{category}] {already + collected}/{max_items} saved.")
            buffer.clear()

    # Final flush
    if buffer:
        append_jsonl(out_file, buffer)
        save_scraped_ids(output_dir, category, scraped_ids)

    total = already + collected
    log.info(f"[{category}] Done. {collected} new items written (total: {total}).")
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Scrape donnons.org for French solidarity-exchange training data."
    )
    p.add_argument(
        "--categories",
        default="all",
        help=(
            "Comma-separated category slugs, or 'all'. Default: all. "
            "Example: meubles,electromenager,vetements-chaussures-et-accessoires"
        ),
    )
    p.add_argument(
        "--max_per_category",
        type=int,
        default=200,
        help="Maximum items to collect per category (default: 200).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory for output JSONL files and checkpoints.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between HTTP requests (default: 2.0). Be polite.",
    )
    p.add_argument(
        "--min_chars",
        type=int,
        default=30,
        help="Minimum description length in characters (default: 30).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.categories == "all":
        targets = ALL_CATEGORIES
    else:
        targets = [c.strip() for c in args.categories.split(",") if c.strip()]
        unknown = [c for c in targets if c not in ALL_CATEGORIES]
        if unknown:
            raise ValueError(
                f"Unknown categories: {unknown}. "
                f"Available: {', '.join(ALL_CATEGORIES)}"
            )

    log.info("=" * 60)
    log.info("Donnons.org scraper — Phase 2, Track B")
    log.info(f"  Categories   : {len(targets)}")
    log.info(f"  Max/category : {args.max_per_category}")
    log.info(f"  Delay        : {args.delay}s")
    log.info(f"  Output dir   : {output_dir}")
    log.info("=" * 60)

    session = requests.Session()

    summary = {}
    for cat in targets:
        t0 = time.time()
        n = scrape_category(
            category=cat,
            max_items=args.max_per_category,
            output_dir=output_dir,
            session=session,
            delay=args.delay,
        )
        summary[cat] = {"n": n, "elapsed_s": round(time.time() - t0, 1)}

    log.info("")
    log.info("=" * 60)
    log.info("Summary")
    log.info("=" * 60)
    total = 0
    for cat, info in summary.items():
        log.info(f"  {cat:<50} n={info['n']:>4}  ({info['elapsed_s']:.0f}s)")
        total += info["n"]
    log.info(f"  {'TOTAL':<50} n={total:>4}")
    log.info("=" * 60)

    manifest = {
        "source": "donnons_org",
        "label": 1,
        "label_name": "solidarity_exchange",
        "categories": targets,
        "max_per_category": args.max_per_category,
        "delay_s": args.delay,
        "timestamp": datetime.utcnow().isoformat(),
        "results": summary,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info(f"Manifest written → {manifest_path}")


if __name__ == "__main__":
    main()
