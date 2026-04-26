#!/bin/bash
#SBATCH --job-name=finetune-tier1
#SBATCH --output=/home/sural/code/logs/finetune_tier1/%j.out
#SBATCH --error=/home/sural/code/logs/finetune_tier1/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Phase 4 Track B — Fine-tune Detoxify-M backbone + post-training threshold analysis
#
# Step 1: Fine-tune unitary/multilingual-toxic-xlm-roberta on Reddit-FR with MSE regression.
#         XLM-RoBERTa backbone preserved; regression head re-initialised for single-label output.
#         Training ~5 min/epoch on A5000.
#
# Step 2: Immediately run threshold analysis on the fine-tuned best checkpoint.
#         This gives score distribution plots and operating point tables directly comparable
#         to the original Detoxify-M results (Phase 1 THRESHOLD_ANALYSIS_REPORT.md).
#
# Success criterion: T1_FNR at 25% deferral on Reddit-FR drops from ~37% to < 15%.
#
# Output layout:
#   ~/code/results/tier1_detoxify_finetuned/    (fine-tuned model checkpoints)
#   ~/code/results/tier1_comparison/tier1_detoxify_finetuned_best/  (threshold analysis)
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 4 Track B — Detoxify-M Fine-tuning + Threshold Analysis"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "=========================================="

if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. Exiting."
    exit 1
fi
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Fix GLIBCXX_3.4.29 mismatch: compute nodes provide GLIBCXX <= 3.4.28 but the thesis
# conda env's pandas requires 3.4.29. Use conda's own libstdc++ instead of the system one.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

FINETUNE_SCRIPT="$HOME/code/phase4_two_tier/finetune_detoxify_tier1.py"
ANALYSIS_SCRIPT="$HOME/code/phase4_two_tier/analyze_threshold_tier1.py"
FINETUNE_DIR="$HOME/code/results/tier1_detoxify_finetuned"
ANALYSIS_DIR="$HOME/code/results/tier1_comparison"
CACHE_DIR="$HOME/datasets/cache"
REDDIT_FR="$HOME/datasets/reddit/balanced/data-fr/test-fr.csv"

mkdir -p "$HOME/code/logs/finetune_tier1"
mkdir -p "$FINETUNE_DIR"
mkdir -p "$ANALYSIS_DIR"

# ── Step 1: Fine-tuning ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Step 1/2: Fine-tuning Detoxify-M on Reddit-FR"
echo "=========================================="
python "$FINETUNE_SCRIPT" \
    --reddit_fr_path "$REDDIT_FR"    \
    --output_dir     "$FINETUNE_DIR" \
    --cache_dir      "$CACHE_DIR"    \
    --epochs         3               \
    --lr             2e-5            \
    --batch_size     16              \
    --grad_accum     2               \
    --max_length     128             \
    --test_fraction  0.1             \
    --val_fraction   0.1             \
    --seed           42              \
    --device         cuda

echo ""
echo "Fine-tuning complete: $(date)"
echo "Best checkpoint: $FINETUNE_DIR/best"

# ── Step 2: Threshold analysis on fine-tuned checkpoint ──────────────────────
echo ""
echo "=========================================="
echo "Step 2/2: Threshold analysis on fine-tuned checkpoint"
echo "=========================================="

# Use best/ checkpoint for analysis; model_key suffix identifies it as fine-tuned
python "$ANALYSIS_SCRIPT" \
    --model          hf_classifier              \
    --hf_model_id    "$FINETUNE_DIR/best"       \
    --hf_toxic_label toxic                       \
    --output_dir     "$ANALYSIS_DIR"             \
    --cache_dir      "$CACHE_DIR"                \
    --reddit_fr_path "$REDDIT_FR"                \
    --device         cpu                         \
    --t_step         0.05

echo ""
echo "=========================================="
echo "Track B complete: $(date)"
echo ""
echo "Compare operating points (Reddit-FR, 25%% deferral):"
echo "  Detoxify-M baseline  : ~/code/results/threshold_analysis/"
echo "  Fine-tuned checkpoint: $ANALYSIS_DIR/$(basename $FINETUNE_DIR)_best/"
echo ""
echo "Target: T1_FNR @ 25%% deferral < 15%% (baseline is ~37%%)"
echo "=========================================="
