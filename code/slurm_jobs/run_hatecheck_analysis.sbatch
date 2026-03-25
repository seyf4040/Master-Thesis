#!/bin/bash
#SBATCH --job-name=hatecheck-analysis
#SBATCH --output=/home/sural/code/logs/hatecheck_analysis/%j.out
#SBATCH --error=/home/sural/code/logs/hatecheck_analysis/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# HateCheck Functionality Analysis: all 10 models × HateCheck EN + FR
# Author: Ural Seyfullah

echo "=========================================="
echo "HateCheck Functionality Analysis"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "=========================================="

# ── GPU check ─────────────────────────────────────────────────────────────────
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. Exiting."
    exit 1
fi
nvidia-smi

# ── Environment ───────────────────────────────────────────────────────────────
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="$HOME/code/results/hatecheck_analysis"
SCRIPT="$HOME/code/run_hatecheck_analysis.py"
CACHE_DIR="$HOME/datasets/cache"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/code/logs/hatecheck_analysis"

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "Starting HateCheck analysis (EN + FR, all models)..."
echo ""

python "$SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir  "$CACHE_DIR"  \
    --models     all           \
    --datasets   both          \
    --threshold  0.5

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Job completed : $(date)"
echo "Output size   : $(du -sh $OUTPUT_DIR)"
echo "=========================================="
