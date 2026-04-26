#!/bin/bash
#SBATCH --job-name=scrape-donnerie
#SBATCH --output=/home/sural/code/logs/donnerie/%j.out
#SBATCH --error=/home/sural/code/logs/donnerie/%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Donnerie.be scraper — Phase 2, Track B
# Collects Belgian French free-item donation posts (label=1, solidarity_exchange).
# CPU-only, no GPU required.
#
# Estimated time: ~3h for 1000 items at 2s delay
#   (1000 items × ~3 requests each × 2s ≈ 1.7h + catalog page fetches)
#
# Checkpointed — safe to cancel and resubmit.

echo "=========================================="
echo "Donnerie.be Scraper — Phase 2"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

python -c "import requests, bs4" 2>/dev/null || pip install requests beautifulsoup4 --quiet

OUTPUT_DIR="$HOME/code/data/donnerie"
SCRIPT="$HOME/code/phase3_data/scrape_donnerie.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/code/logs/donnerie"

echo ""
echo "Configuration:"
echo "  Max pages  : 140 (full site)"
echo "  Max items  : 1000"
echo "  Delay      : 2.0s"
echo "  Output     : $OUTPUT_DIR"
echo ""

python "$SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --max_pages  140           \
    --max_items  1000          \
    --delay      2.0

echo ""
echo "=========================================="
echo "Job completed : $(date)"
echo "Items collected:"
wc -l "$OUTPUT_DIR/donnerie.jsonl" 2>/dev/null || echo "  (no output file)"
echo "=========================================="
