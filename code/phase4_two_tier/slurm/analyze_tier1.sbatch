#!/bin/bash
#SBATCH --job-name=analyze-tier1
#SBATCH --output=/home/sural/code/logs/analyze_tier1/%j.out
#SBATCH --error=/home/sural/code/logs/analyze_tier1/%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Phase 4 Track A — Tier 1 candidate threshold analysis (CPU only)
#
# Tests unitary/multilingual-toxic-xlm-roberta as an alternative Tier 1 candidate.
# Note: this model is the same backbone as Detoxify-multilingual; results will closely
# match the Phase 1 threshold analysis (THRESHOLD_ANALYSIS_REPORT.md). Run both to confirm.
#
# Output: ~/code/results/tier1_comparison/unitary_multilingual-toxic-xlm-roberta/
#
# After completion, compare score distributions against:
#   ~/code/results/threshold_analysis/  (Detoxify-M Phase 1 results)
#   ~/code/results/tier1_detoxify_finetuned/  (Track B fine-tuned checkpoint)
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 4 — Tier 1 Threshold Analysis (Track A)"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Fix GLIBCXX_3.4.29 mismatch: use conda's own libstdc++ instead of system's older version.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

SCRIPT="$HOME/code/phase4_two_tier/analyze_threshold_tier1.py"
OUTPUT_DIR="$HOME/code/results/tier1_comparison"
CACHE_DIR="$HOME/datasets/cache"
REDDIT_FR="$HOME/datasets/reddit/balanced/data-fr/test-fr.csv"

mkdir -p "$HOME/code/logs/analyze_tier1"
mkdir -p "$OUTPUT_DIR"

# ── Track A: unitary/multilingual-toxic-xlm-roberta ──────────────────────────
echo ""
echo "=========================================="
echo "Model: unitary/multilingual-toxic-xlm-roberta"
echo "=========================================="
python "$SCRIPT" \
    --model          hf_classifier                          \
    --hf_model_id    unitary/multilingual-toxic-xlm-roberta \
    --hf_toxic_label toxic                                   \
    --output_dir     "$OUTPUT_DIR"                           \
    --cache_dir      "$CACHE_DIR"                            \
    --reddit_fr_path "$REDDIT_FR"                            \
    --device         cpu                                     \
    --t_step         0.05

echo ""
echo "=========================================="
echo "Track A complete: $(date)"
echo "Results: $OUTPUT_DIR/unitary_multilingual-toxic-xlm-roberta/"
echo ""
echo "Compare against Detoxify-M Phase 1 baseline:"
echo "  ~/code/results/threshold_analysis/"
echo "=========================================="
