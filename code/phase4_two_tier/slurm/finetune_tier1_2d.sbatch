#!/bin/bash
#SBATCH --job-name=finetune-tier1-2d
#SBATCH --output=/home/sural/code/logs/finetune_tier1/%j.out
#SBATCH --error=/home/sural/code/logs/finetune_tier1/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Phase 4 Group 2d — Combined soft labels + synthetic data
#
# Combines the two Group 2 improvements that worked independently:
#   2b: label_smoothing=0.05 → reduces T1_FPR (23.8%→13.5%) by preventing
#       overconfident "safe" predictions on ambiguous content
#   2c: synthetic data (+1,500 items) → breaks T_high from 0.95→0.80, best
#       honest F1 (0.668) and HC-FR (0.816) via distribution diversity
#
# Hypothesis: effects stack — combined model should achieve T1_FPR ~13–15%
# AND maintain T_high ≤ 0.80 with potentially lower T1_FNR than either alone.
#
# Only 2 epochs needed: all Group 2 checkpoints peaked at epoch 2.
#
# Output: ~/code/results/tier1_finetuned_2d/
# Analysis: ~/code/results/tier1_comparison_honest/2d_combined/
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 4 Group 2d — Soft Labels + Synthetic Data"
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
OUT_2D="$HOME/code/results/tier1_finetuned_2d"
ANALYSIS_DIR="$HOME/code/results/tier1_comparison_honest"

mkdir -p "$HOME/code/logs/finetune_tier1"
mkdir -p "$ANALYSIS_DIR/2d_combined"

# ── Prepare honest test set CSV (511 held-out samples, no leakage) ───────────
echo ""
echo "Preparing honest test set CSV..."
python3 -c "
import json, pandas as pd
d = json.load(open('$TEST_SET_JSON'))
pd.DataFrame(d['samples']).to_csv('/tmp/test_set_honest.csv', index=False)
print(f'  Written {len(d[\"samples\"])} samples to /tmp/test_set_honest.csv')
"

# ── 2d: Soft labels + synthetic data (2 epochs) ───────────────────────────────
echo ""
echo "=========================================="
echo "Sub-experiment 2d: 2 epochs, soft labels {0.05, 0.95} + synthetic data"
echo "  label_smoothing : 0.05"
echo "  synthetic_data  : $SYNTHETIC_DIR"
echo "  epochs          : 2  (all Group 2 variants peaked at epoch 2)"
echo "=========================================="

python "$FINETUNE_SCRIPT" \
    --reddit_fr_path      "$REDDIT_FR"       \
    --output_dir          "$OUT_2D"          \
    --cache_dir           "$CACHE_DIR"       \
    --epochs              2                  \
    --lr                  2e-5               \
    --batch_size          16                 \
    --grad_accum          2                  \
    --max_length          128                \
    --test_fraction       0.1                \
    --val_fraction        0.1                \
    --seed                42                 \
    --label_smoothing     0.05               \
    --synthetic_data_path "$SYNTHETIC_DIR"   \
    --device              cuda

echo ""
echo "--- Threshold analysis: 2d checkpoint (honest 511-sample holdout) ---"
python "$ANALYSIS_SCRIPT" \
    --model          hf_classifier                   \
    --hf_model_id    "$OUT_2D/best"                  \
    --hf_toxic_label toxic                            \
    --output_dir     "$ANALYSIS_DIR/2d_combined"     \
    --cache_dir      "$CACHE_DIR"                    \
    --reddit_fr_path /tmp/test_set_honest.csv        \
    --device         cpu                             \
    --t_step         0.05

echo ""
echo "=========================================="
echo "Sub-experiment 2d complete: $(date)"
echo ""
echo "Threshold analysis results (honest 511-sample holdout):"
python3 -c "
import json, glob

dirs = glob.glob('$ANALYSIS_DIR/2d_combined/*/threshold_analysis.json')
for dpath in dirs:
    d = json.load(open(dpath))
    rr = d['single_threshold'].get('reddit_fr', {})
    b = rr.get('best', {})
    ops = d['operating_points'].get('reddit_fr', [])
    print(f'  Reddit-FR best F1: {b.get(\"f1\", \"?\"):.3f} at T={b.get(\"t\", \"?\"):.2f}')
    for op in ops:
        print(f'  {op[\"label\"]}: T1_FNR={op[\"tier1_fnr\"]:.1%}  T1_FPR={op[\"tier1_fpr\"]:.1%}  deferral={op[\"deferral_rate\"]:.1%}  T_high={op[\"t_high\"]:.2f}')
"
echo ""
echo "Reference — Group 2 results on same honest holdout:"
echo "  2b (soft only):       T1_FNR=28.3% T1_FPR=13.5% deferral=22.5% T_high=0.95"
echo "  2c (synthetic only):  T1_FNR=25.2% T1_FPR=18.5% deferral=12.3% T_high=0.80"
echo "  Target: T1_FNR < 25%, T1_FPR < 14%, T_high < 0.90"
echo "=========================================="
