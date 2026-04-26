#!/bin/bash
#SBATCH --job-name=score-t1-finetuned
#SBATCH --output=/home/sural/code/logs/score_two_tier/%j.out
#SBATCH --error=/home/sural/code/logs/score_two_tier/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Phase 4 — Two-Tier Scoring: fine-tuned Tier 1
#
# Scores all 511 held-out test samples with:
#   Tier 1: fine-tuned Detoxify-M checkpoint (3 epochs MSE regression on Reddit-FR)
#   Tier 2: SG-2b + Reddit-FR LoRA adapter
#
# All samples scored by both tiers; per-sample (score, pred, ms) saved to
# raw_scores.json. Run simulate_thresholds.py offline to sweep all threshold configs.
#
# Output: ~/code/results/two_tier_scores/finetuned/raw_scores.json
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 4 — Two-Tier Scoring (Fine-tuned Tier 1)"
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

# Fix GLIBCXX_3.4.29 mismatch on compute nodes
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

SCRIPT="$HOME/code/phase4_two_tier/score_two_tier.py"
TIER1_DIR="$HOME/code/results/tier1_detoxify_finetuned/best"
TEST_SET="$HOME/code/results/tier1_detoxify_finetuned/test_set.json"
TIER2_ADAPTER="$HOME/code/results/lora_adapters/shieldgemma_2b/reddit_fr/best"
OUTPUT_DIR="$HOME/code/results/two_tier_scores/finetuned"
CACHE_DIR="$HOME/datasets/cache"

mkdir -p "$HOME/code/logs/score_two_tier"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Tier 1: fine-tuned Detoxify-M ($TIER1_DIR)"
echo "Tier 2: SG-2b + Reddit-FR LoRA"
echo "Test set: $TEST_SET  (511 held-out samples)"
echo "=========================================="

python "$SCRIPT" \
    --tier1_model_id    "$TIER1_DIR"   \
    --tier1_toxic_label toxic           \
    --tier2_adapter_dir "$TIER2_ADAPTER" \
    --test_set_json     "$TEST_SET"    \
    --output_dir        "$OUTPUT_DIR"  \
    --cache_dir         "$CACHE_DIR"   \
    --device            cuda

echo ""
echo "=========================================="
echo "Scoring complete: $(date)"
echo "Scores saved to: $OUTPUT_DIR/raw_scores.json"
echo ""
echo "Next step — run offline simulation:"
echo "  python code/phase4_two_tier/simulate_thresholds.py \\"
echo "      --scores_json $OUTPUT_DIR/raw_scores.json \\"
echo "      --output_dir  $OUTPUT_DIR/simulation"
echo "=========================================="
