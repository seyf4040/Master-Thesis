#!/bin/bash
#SBATCH --job-name=scrape-2ememain
#SBATCH --output=/home/sural/code/logs/2ememain/%j.out
#SBATCH --error=/home/sural/code/logs/2ememain/%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# 2ememain.be scraper — Phase 2, Track B
# Collects Belgian French commercial listings (label=0, commercial_listing).
# CPU-only, no GPU required.
#
# Strategy: fetches first page of each of 15 target categories (30 items/page)
# then individual listing pages for full descriptions. French-language filter
# applied via langdetect to exclude Dutch-language listings.
#
# Estimated time: ~2h for 800 items at 2s delay
#   (15 cat pages + 800 listing pages) × 2s ≈ 1.6h
#
# Checkpointed — safe to cancel and resubmit.

echo "=========================================="
echo "2ememain.be Scraper — Phase 2"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

python -c "import requests, bs4, langdetect" 2>/dev/null || \
    pip install requests beautifulsoup4 langdetect --quiet

OUTPUT_DIR="$HOME/code/data/2ememain"
SCRIPT="$HOME/code/phase3_data/scrape_2ememain.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/code/logs/2ememain"

echo ""
echo "Configuration:"
echo "  Categories : all (15)"
echo "  Max items  : 800"
echo "  Delay      : 2.0s"
echo "  Output     : $OUTPUT_DIR"
echo ""

python "$SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --categories all           \
    --max_items  800           \
    --delay      2.0

echo ""
echo "=========================================="
echo "Job completed : $(date)"
echo "Items collected:"
wc -l "$OUTPUT_DIR/2ememain.jsonl" 2>/dev/null || echo "  (no output file)"
echo "=========================================="
