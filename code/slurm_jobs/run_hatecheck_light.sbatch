#!/bin/bash
#SBATCH --job-name=hatecheck-light
#SBATCH --output=/home/sural/code/logs/hatecheck_analysis/%j.out
#SBATCH --error=/home/sural/code/logs/hatecheck_analysis/%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# HateCheck analysis — light models only, any available GPU node
# Passes --models all but relies on the built-in VRAM guard to automatically
# skip any model that does not fit on the assigned GPU.
# Author: Ural Seyfullah

echo "=========================================="
echo "HateCheck Analysis — Light Models"
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
# --models all: the VRAM guard in the script will automatically skip any model
# that does not fit on the assigned GPU and print a clear warning for each one.
echo ""
echo "GPU assigned: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Models that exceed available VRAM will be skipped automatically."
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
