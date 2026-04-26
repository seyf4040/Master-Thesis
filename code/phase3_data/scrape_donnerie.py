#!/usr/bin/env python3
"""
scrape_donnerie.py — Scrape donnerie.be for solidarity-exchange training data (Phase 2, Track B)

Donnerie.be is a Belgian free-item donation platform serving Wallonia and Brussels.
Posts are in Belgian French and follow the exact same register as Shareish:
"Canapé à donner, à venir chercher à Jette, bon état général."

This makes donnerie.be the highest-relevance source for positive-class training data —
same country, same language register, same object vocabulary as Shareish.

Full descriptions are embedded in JSON-LD (schema.org WebPage) in static HTML.
No JavaScript rendering or authentication required.

Label: 1 = solidarity_exchange

Output JSONL fields:
    text, label, source, url, location, timestamp

Usage:
    python code/phase3_data/scrape_donnerie.py \\
        --output_dir ~/code/data/donnerie \\
        --max_pages 50 \\
        --delay 2.0

    # Quick test (5 pages only):
    python code/phase3_data/scrape_donnerie.py \\
        --output_dir /tmp/donnerie_test \\
        --max_pages 5 \\
        --delay 1.5
"""

import json
import re
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

BASE_URL = "https://donnerie.be"
CATALOG_URL = BASE_URL + "/annonces/page/{}/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research-bot/1.0; "
        "master-thesis content-moderation ULiege; respectful-crawl)"
    ),
    "Accept-Language": "fr-BE,fr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Regex to strip phone numbers for GDPR compliance
_PHONE_RE = re.compile(
    r"(\+32|0032|0)[\s.\-/]?\d{2,3}[\s.\-/]?\d{2}[\s.\-/]?\d{2}[\s.\-/]?\d{2}"
)


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


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_item_urls(html: str) -> List[str]:
    """Extract /annonces/{slug}/ hrefs from a catalog page. Skip pagination links."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Item pages: /annonces/{slug}/ where slug is not a digit-only page number
        # Normalise to absolute URL (donnerie.be uses absolute hrefs)
        full = href if href.startswith("http") else urljoin(BASE_URL, href)
        if "/annonces/" in full and BASE_URL in full:
            # Exclude pagination and the bare /annonces/ index
            if "/page/" in full or full.rstrip("/").endswith("/annonces"):
                continue
            if full not in seen:
                seen.add(full)
                urls.append(full)
    return urls


def parse_total_pages(html: str) -> int:
    """Extract total page count from 'Page X of Y' text."""
    match = re.search(r"Page\s+\d+\s+of\s+(\d+)", html, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


def strip_phone(text: str) -> str:
    """Remove Belgian phone numbers for GDPR compliance."""
    return _PHONE_RE.sub("[contact supprimé]", text).strip()


def parse_item_page(html: str, url: str) -> Optional[Dict]:
    """Extract item data from a donnerie.be /annonces/{slug}/ page via JSON-LD."""
    soup = BeautifulSoup(html, "html.parser")

    title = None
    description = None
    location = None

    # ── JSON-LD (primary) ─────────────────────────────────────────────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            schemas = data if isinstance(data, list) else [data]
            for schema in schemas:
                desc = (schema.get("description") or "").strip()
                name = (schema.get("name") or "").strip()
                if desc and len(desc) > 15:
                    title = name
                    description = desc

                    # Location: donnerie embeds city in the schema or in <li> tags
                    loc = (
                        schema.get("contentLocation", {}).get("name")
                        or schema.get("address", {}).get("addressLocality")
                    )
                    if loc:
                        location = loc
                    break
        except (json.JSONDecodeError, AttributeError):
            continue

    # ── Location fallback: scan <li> elements ─────────────────────────────
    if location is None:
        for li in soup.find_all("li"):
            text = li.get_text(separator=" ", strip=True)
            # "Ville : Jette" or "Province : Bruxelles-Capitale"
            if "Ville" in text or "ville" in text:
                parts = text.split(":")
                if len(parts) > 1:
                    location = parts[-1].strip()
                    break

    if not description:
        return None

    # Combine title + description into a single text field
    if title and title.lower() not in description.lower():
        text = f"{title}\n{description}"
    else:
        text = description

    text = strip_phone(text)

    return {
        "text": text,
        "label": 1,
        "source": "donnerie_be",
        "url": url,
        "location": location,
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
    f = output_dir / "donnerie.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def append_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

def scrape(
    max_pages: int,
    max_items: int,
    output_dir: Path,
    session: requests.Session,
    delay: float,
) -> int:
    out_file = output_dir / "donnerie.jsonl"
    scraped_urls = load_scraped_urls(output_dir)
    seen_hashes: Set[str] = set()
    already = count_existing(output_dir)

    if already >= max_items:
        log.info(f"Already have {already}/{max_items} items. Nothing to do.")
        return already

    remaining = max_items - already
    log.info(f"Need {remaining} more items (have {already}/{max_items}).")

    # Discover total pages from page 1
    first_html = fetch(session, CATALOG_URL.format(1), delay)
    if not first_html:
        log.error("Could not fetch page 1. Aborting.")
        return already

    total_pages = min(parse_total_pages(first_html), max_pages)
    log.info(f"Catalog has {total_pages} pages (capped at {max_pages}).")

    collected = 0
    buffer: List[Dict] = []

    for page_num in range(1, total_pages + 1):
        if collected >= remaining:
            break

        html = first_html if page_num == 1 else fetch(session, CATALOG_URL.format(page_num), delay)
        if not html:
            continue

        item_urls = [u for u in parse_item_urls(html) if u not in scraped_urls]
        log.info(f"Page {page_num}/{total_pages}: {len(item_urls)} new item URLs.")

        for url in item_urls:
            if collected >= remaining:
                break

            item_html = fetch(session, url, delay)
            scraped_urls.add(url)

            if not item_html:
                continue

            item = parse_item_page(item_html, url)
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
        description="Scrape donnerie.be for Belgian French solidarity-exchange data."
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--max_pages", type=int, default=140,
        help="Max catalog pages to crawl (default: 140 = full site).",
    )
    p.add_argument(
        "--max_items", type=int, default=1000,
        help="Stop after collecting this many items (default: 1000).",
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

    log.info("=" * 55)
    log.info("Donnerie.be scraper — Phase 2, Track B")
    log.info(f"  Max pages  : {args.max_pages}")
    log.info(f"  Max items  : {args.max_items}")
    log.info(f"  Delay      : {args.delay}s")
    log.info(f"  Output     : {output_dir}")
    log.info("=" * 55)

    session = requests.Session()
    t0 = time.time()
    n = scrape(args.max_pages, args.max_items, output_dir, session, args.delay)
    elapsed = time.time() - t0

    log.info(f"Total items: {n} in {elapsed/60:.1f} min")

    manifest = {
        "source": "donnerie_be",
        "label": 1,
        "label_name": "solidarity_exchange",
        "max_pages": args.max_pages,
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
