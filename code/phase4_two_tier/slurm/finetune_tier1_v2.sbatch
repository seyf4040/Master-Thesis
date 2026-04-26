#!/bin/bash
#SBATCH --job-name=finetune-tier1-v2
#SBATCH --output=/home/sural/code/logs/finetune_tier1/%j.out
#SBATCH --error=/home/sural/code/logs/finetune_tier1/%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Phase 4 Group 2 — Improved Tier 1 fine-tuning (three parallel sub-experiments)
#
# Runs 2a, 2b, 2c sequentially in one job. After each training run, immediately
# performs threshold analysis on the honest 511-sample holdout (test_set.csv).
#
# 2a — Extended epochs (10 epochs, hard labels)
#       Goal: exploit remaining val-loss headroom (Δ≈−0.002/epoch at epoch 3)
#       Output: ~/code/results/tier1_finetuned_e10/
#
# 2b — Soft labels (5 epochs, label_smoothing=0.05, targets {0.05, 0.95})
#       Goal: reduce bimodal score collapse, create tunable uncertain middle zone
#       Output: ~/code/results/tier1_finetuned_soft/
#
# 2c — Synthetic data (5 epochs, Phase 3 Track A 1500 items added to Reddit-FR)
#       Goal: more diverse informal French hate speech for backbone adaptation
#       Output: ~/code/results/tier1_finetuned_synthetic/
#
# Threshold analysis for each run uses /tmp/test_set.csv (511 samples only)
# to avoid data leakage. Written once at job start.
#
# Success criterion: T1_FNR at ~25% deferral on honest holdout < 15%
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 4 Group 2 — Improved Tier 1 Fine-tuning"
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

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

FINETUNE_SCRIPT="$HOME/code/phase4_two_tier/finetune_detoxify_tier1.py"
ANALYSIS_SCRIPT="$HOME/code/phase4_two_tier/analyze_threshold_tier1.py"
REDDIT_FR="$HOME/datasets/reddit/balanced/data-fr/test-fr.csv"
CACHE_DIR="$HOME/datasets/cache"
SYNTHETIC_DIR="$HOME/code/data/synthetic"
TEST_SET_JSON="$HOME/code/results/tier1_detoxify_finetuned/test_set.json"
ANALYSIS_DIR="$HOME/code/results/tier1_comparison_honest"

mkdir -p "$HOME/code/logs/finetune_tier1"
mkdir -p "$ANALYSIS_DIR"

# ── Prepare honest test set CSV (511 held-out samples, no leakage) ───────────
echo ""
echo "Preparing honest test set CSV..."
python3 -c "
import json, pandas as pd
d = json.load(open('$TEST_SET_JSON'))
pd.DataFrame(d['samples']).to_csv('/tmp/test_set_honest.csv', index=False)
print(f'  Written {len(d[\"samples\"])} samples to /tmp/test_set_honest.csv')
"

# ── 2a: Extended epochs (10 epochs, hard labels) ─────────────────────────────
echo ""
echo "=========================================="
echo "Sub-experiment 2a: 10 epochs, hard labels {0,1}"
echo "=========================================="
OUT_2A="$HOME/code/results/tier1_finetuned_e10"
python "$FINETUNE_SCRIPT" \
    --reddit_fr_path "$REDDIT_FR"  \
    --output_dir     "$OUT_2A"     \
    --cache_dir      "$CACHE_DIR"  \
    --epochs         10            \
    --lr             2e-5          \
    --batch_size     16            \
    --grad_accum     2             \
    --max_length     128           \
    --test_fraction  0.1           \
    --val_fraction   0.1           \
    --seed           42            \
    --device         cuda

echo ""
echo "--- Threshold analysis: 2a checkpoint (honest holdout) ---"
python "$ANALYSIS_SCRIPT" \
    --model          hf_classifier         \
    --hf_model_id    "$OUT_2A/best"        \
    --hf_toxic_label toxic                  \
    --output_dir     "$ANALYSIS_DIR/2a_e10" \
    --cache_dir      "$CACHE_DIR"           \
    --reddit_fr_path /tmp/test_set_honest.csv \
    --device         cpu                    \
    --t_step         0.05

echo "Sub-experiment 2a complete: $(date)"

# ── 2b: Soft labels (5 epochs, label_smoothing=0.05) ─────────────────────────
echo ""
echo "=========================================="
echo "Sub-experiment 2b: 5 epochs, soft labels {0.05, 0.95}"
echo "=========================================="
OUT_2B="$HOME/code/results/tier1_finetuned_soft"
python "$FINETUNE_SCRIPT" \
    --reddit_fr_path  "$REDDIT_FR"  \
    --output_dir      "$OUT_2B"     \
    --cache_dir       "$CACHE_DIR"  \
    --epochs          5             \
    --lr              2e-5          \
    --batch_size      16            \
    --grad_accum      2             \
    --max_length      128           \
    --test_fraction   0.1           \
    --val_fraction    0.1           \
    --seed            42            \
    --label_smoothing 0.05          \
    --device          cuda

echo ""
echo "--- Threshold analysis: 2b checkpoint (honest holdout) ---"
python "$ANALYSIS_SCRIPT" \
    --model          hf_classifier          \
    --hf_model_id    "$OUT_2B/best"         \
    --hf_toxic_label toxic                   \
    --output_dir     "$ANALYSIS_DIR/2b_soft" \
    --cache_dir      "$CACHE_DIR"            \
    --reddit_fr_path /tmp/test_set_honest.csv \
    --device         cpu                     \
    --t_step         0.05

echo "Sub-experiment 2b complete: $(date)"

# ── 2c: Synthetic data (5 epochs, Phase 3 Track A) ───────────────────────────
echo ""
echo "=========================================="
echo "Sub-experiment 2c: 5 epochs, Reddit-FR + 1500 synthetic items"
echo "=========================================="
OUT_2C="$HOME/code/results/tier1_finetuned_synthetic"
python "$FINETUNE_SCRIPT" \
    --reddit_fr_path      "$REDDIT_FR"       \
    --output_dir          "$OUT_2C"          \
    --cache_dir           "$CACHE_DIR"       \
    --epochs              5                  \
    --lr                  2e-5               \
    --batch_size          16                 \
    --grad_accum          2                  \
    --max_length          128                \
    --test_fraction       0.1                \
    --val_fraction        0.1                \
    --seed                42                 \
    --synthetic_data_path "$SYNTHETIC_DIR"   \
    --device              cuda

echo ""
echo "--- Threshold analysis: 2c checkpoint (honest holdout) ---"
python "$ANALYSIS_SCRIPT" \
    --model          hf_classifier               \
    --hf_model_id    "$OUT_2C/best"              \
    --hf_toxic_label toxic                        \
    --output_dir     "$ANALYSIS_DIR/2c_synthetic" \
    --cache_dir      "$CACHE_DIR"                 \
    --reddit_fr_path /tmp/test_set_honest.csv     \
    --device         cpu                          \
    --t_step         0.05

echo "Sub-experiment 2c complete: $(date)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "All Group 2 experiments complete: $(date)"
echo ""
echo "Threshold analysis results (honest 511-sample holdout):"
for name in 2a_e10 2b_soft 2c_synthetic; do
    dir="$ANALYSIS_DIR/$name"
    if [ -f "$dir/threshold_analysis.json" ]; then
        echo "  $name:"
        python3 -c "
import json
d = json.load(open('$dir/threshold_analysis.json'))
rr = d['single_threshold']['reddit_fr']
b = rr['best']
ops = d['operating_points']['reddit_fr']
print(f'    Reddit-FR best F1: {b[\"f1\"]:.3f} at T={b[\"t\"]:.2f}')
for op in ops:
    print(f'    {op[\"label\"]}: T1_FNR={op[\"tier1_fnr\"]:.1%}  deferral={op[\"deferral_rate\"]:.1%}')
"
    fi
done
echo ""
echo "Success criterion: T1_FNR at ~25% deferral < 15%"
echo "=========================================="
