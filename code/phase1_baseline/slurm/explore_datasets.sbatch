#!/bin/bash
#SBATCH --job-name=explore_datasets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sural/code/logs/explore_datasets/%j.out
#SBATCH --error=/home/sural/code/logs/explore_datasets/%j.err

# ── Dataset Structure Explorer — Pass 1 only ─────────────────────────────────
# Examines raw schema, splits, columns, label taxonomies, and sample rows
# for all 8 datasets BEFORE any label collapsing.
#
# Bring the output back to design Pass 2 (statistics) with correct label logic.
#
# Submit: sbatch code/phase1_baseline/slurm/explore_datasets.sbatch
# Output: ~/code/results/dataset_exploration/structure.txt
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

echo "=========================================="
echo "Dataset Structure Explorer — Pass 1"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "Python: $(python --version)"
echo "Env:    $CONDA_PREFIX"
echo "=========================================="

OUT_DIR=~/code/results/dataset_exploration
mkdir -p "$OUT_DIR"

python code/phase1_baseline/explore_datasets.py \
    --cache_dir      ~/datasets/cache \
    --openai_path    ~/datasets/openai/samples-1680.jsonl \
    --toxigen_path   ~/datasets/toxigen/toxigen_train.jsonl \
    --reddit_en_path ~/datasets/reddit/balanced/data-en/test-en.csv \
    --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv \
    --datasets       all \
    --output         "$OUT_DIR/structure.txt"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Output: $OUT_DIR/structure.txt"
echo "=========================================="
