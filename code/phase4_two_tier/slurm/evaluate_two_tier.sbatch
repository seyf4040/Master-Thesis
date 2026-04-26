#!/bin/bash
#SBATCH --job-name=eval-two-tier
#SBATCH --output=/home/sural/code/logs/eval_two_tier/%j.out
#SBATCH --error=/home/sural/code/logs/eval_two_tier/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Phase 4 — End-to-end two-tier moderation evaluation
#
# Tier 1: Fine-tuned Detoxify-M checkpoint (~/code/results/tier1_detoxify_finetuned/best)
# Tier 2: SG-2b + Reddit-FR LoRA adapter  (~/code/results/lora_adapters/shieldgemma_2b/reddit_fr/best)
#
# Runs two operating points from the threshold analysis:
#   - Conservative (low deferral): T_low=0.05, T_high=0.95
#   - Balanced     (mid deferral): T_low=0.15, T_high=0.85
#
# Adjust T_low / T_high based on the operating points printed by:
#   finetune_detoxify_tier1.sbatch → threshold analysis output
#
# Success criterion:
#   Combined F1 ≥ 0.662 (SG-2b alone Reddit-FR baseline) at deferral_rate ≤ 60%
#   Combined FNR ≤ 0.20 (max 1 in 5 hateful items allowed to leak through Tier 1)
#
# Output: ~/code/results/two_tier_eval/{conservative,balanced}/
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 4 — Two-Tier End-to-End Evaluation"
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

SCRIPT="$HOME/code/phase4_two_tier/evaluate_two_tier.py"
TIER1_DIR="$HOME/code/results/tier1_detoxify_finetuned/best"
TIER2_ADAPTER="$HOME/code/results/lora_adapters/shieldgemma_2b/reddit_fr/best"
TEST_SET="$HOME/code/results/tier1_detoxify_finetuned/test_set.json"
OUTPUT_ROOT="$HOME/code/results/two_tier_eval"
CACHE_DIR="$HOME/datasets/cache"

mkdir -p "$HOME/code/logs/eval_two_tier"
mkdir -p "$OUTPUT_ROOT"

# ── Run 1: Conservative operating point ──────────────────────────────────────
# Low deferral (~10-15%). Only the most ambiguous content reaches Tier 2.
# Expected: high Tier 1 FNR, but compute savings are large.
echo ""
echo "=========================================="
echo "Run 1/2: Conservative (T_low=0.05, T_high=0.95)"
echo "=========================================="
python "$SCRIPT" \
    --test_set_json     "$TEST_SET"       \
    --tier1_model_id    "$TIER1_DIR"      \
    --tier1_t_low       0.05              \
    --tier1_t_high      0.95              \
    --tier2_adapter_dir "$TIER2_ADAPTER"  \
    --output_dir        "$OUTPUT_ROOT/conservative" \
    --cache_dir         "$CACHE_DIR"      \
    --device            cuda              \
    --full_comparison

echo "Run 1 complete: $(date)"

# ── Run 2: Balanced operating point ──────────────────────────────────────────
# Mid deferral (~25%). Balances compute savings with acceptable T1 FNR.
# This is the primary comparison point from the threshold analysis.
echo ""
echo "=========================================="
echo "Run 2/2: Balanced (T_low=0.15, T_high=0.85)"
echo "=========================================="
python "$SCRIPT" \
    --test_set_json     "$TEST_SET"       \
    --tier1_model_id    "$TIER1_DIR"      \
    --tier1_t_low       0.15              \
    --tier1_t_high      0.85              \
    --tier2_adapter_dir "$TIER2_ADAPTER"  \
    --output_dir        "$OUTPUT_ROOT/balanced" \
    --cache_dir         "$CACHE_DIR"      \
    --device            cuda

echo "Run 2 complete: $(date)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "All runs complete: $(date)"
echo ""
echo "Results:"
for dir in conservative balanced; do
    echo "  $dir:"
    if [ -f "$OUTPUT_ROOT/$dir/summary.txt" ]; then
        grep -E "f1|fnr|deferral" "$OUTPUT_ROOT/$dir/summary.txt" | head -5 | sed 's/^/    /'
    fi
done
echo ""
echo "Full results: $OUTPUT_ROOT/"
echo "=========================================="
