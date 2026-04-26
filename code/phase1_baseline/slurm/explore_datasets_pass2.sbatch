#!/bin/bash
#SBATCH --job-name=dataset_stats_p2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/sural/code/logs/explore_datasets/%j.out
#SBATCH --error=/home/sural/code/logs/explore_datasets/%j.err

# ── Dataset Statistics Explorer — Pass 2 ─────────────────────────────────────
# Full-dataset statistics: label distributions, text length, null audit,
# per-category breakdowns. No sampling — processes all rows.
#
# Civil Comments train split (1.8M rows) dominates runtime (~10-15 min).
# 32G RAM and 1h limit are conservative; adjust if needed.
#
# Submit: sbatch code/phase1_baseline/slurm/explore_datasets_pass2.sbatch
# Output: ~/code/results/dataset_exploration/statistics.txt
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

echo "=========================================="
echo "Dataset Statistics — Pass 2"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Python: $(python --version)"
echo "Start:  $(date)"
echo "=========================================="

OUT_DIR=~/code/results/dataset_exploration
mkdir -p "$OUT_DIR"

python code/phase1_baseline/explore_datasets_pass2.py \
    --cache_dir      ~/datasets/cache \
    --openai_path    ~/datasets/openai/samples-1680.jsonl \
    --toxigen_path   ~/datasets/toxigen/toxigen_train.jsonl \
    --reddit_en_path ~/datasets/reddit/balanced/data-en/test-en.csv \
    --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv \
    --datasets       all \
    --output         "$OUT_DIR/statistics.txt"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Output: $OUT_DIR/statistics.txt"
echo "=========================================="
