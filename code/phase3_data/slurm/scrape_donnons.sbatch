#!/bin/bash
#SBATCH --job-name=scrape-donnons
#SBATCH --output=/home/sural/code/logs/donnons/%j.out
#SBATCH --error=/home/sural/code/logs/donnons/%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Donnons.org scraper — Phase 2, Track B
# Collects French solidarity-exchange posts (label=1) for the Shareish
# content-type classifier. CPU-only, no GPU required.
#
# Estimated time: ~3h for all 17 categories at 200 items each with 2s delay
#   (17 × 200 items × ~3 requests/item × 2s = ~6,800 requests ≈ 3.8h)
#
# Customise:
#   - CATEGORIES: comma-separated slugs or "all"
#   - MAX_PER_CATEGORY: items to collect per category
#   - DELAY: seconds between requests (keep ≥ 1.5 to be polite)

echo "=========================================="
echo "Donnons.org Scraper — Phase 2"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "=========================================="

# ── Environment ───────────────────────────────────────────────────────────────
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Install dependencies if not present
python -c "import requests, bs4" 2>/dev/null || pip install requests beautifulsoup4 --quiet

# ── Configuration ─────────────────────────────────────────────────────────────
CATEGORIES="all"
MAX_PER_CATEGORY=200
DELAY=2.0

OUTPUT_DIR="$HOME/code/data/donnons"
SCRIPT="$HOME/code/phase3_data/scrape_donnons.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/code/logs/donnons"

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "Configuration:"
echo "  Categories      : $CATEGORIES"
echo "  Max/category    : $MAX_PER_CATEGORY"
echo "  Delay           : ${DELAY}s"
echo "  Output dir      : $OUTPUT_DIR"
echo ""

python "$SCRIPT" \
    --categories      "$CATEGORIES"       \
    --max_per_category "$MAX_PER_CATEGORY" \
    --output_dir      "$OUTPUT_DIR"        \
    --delay           "$DELAY"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Job completed : $(date)"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (no JSONL files)"
echo ""
echo "Line counts (items per category):"
wc -l "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (no JSONL files)"
echo "=========================================="
