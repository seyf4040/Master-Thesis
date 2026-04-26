#!/bin/bash
#SBATCH --job-name=threshold-analysis
#SBATCH --output=/home/sural/code/logs/threshold_analysis/%j.out
#SBATCH --error=/home/sural/code/logs/threshold_analysis/%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Phase 1 — Detoxify-multilingual two-tier threshold analysis
#
# Scores all samples in hatecheck_fr, french_hate_superset, reddit_fr,
# then sweeps (T_low, T_high) to find optimal operating points for the
# two-tier moderation architecture.
#
# Raw scores are saved to raw_scores.json so plots can be regenerated
# without re-running the model:
#   python analyze_threshold_detoxify.py \
#       --scores_json ~/code/results/threshold_analysis/raw_scores.json \
#       --output_dir  ~/code/results/threshold_analysis
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Detoxify-multilingual Threshold Analysis"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

mkdir -p "$HOME/code/logs/threshold_analysis"

python "$HOME/code/phase1_baseline/analyze_threshold_detoxify.py" \
    --output_dir    "$HOME/code/results/threshold_analysis"   \
    --cache_dir     "$HOME/datasets/cache"                    \
    --reddit_fr_path "$HOME/datasets/reddit/balanced/data-fr/test-fr.csv" \
    --device        cpu                                        \
    --t_step        0.05

echo ""
echo "=========================================="
echo "Analysis complete: $(date)"
echo "Results in: $HOME/code/results/threshold_analysis/"
echo "  raw_scores.json          — raw scores per dataset"
echo "  threshold_analysis.json  — sweep results + operating points"
echo "  fig_score_distributions.png"
echo "  fig_threshold_sweep.png"
echo "  fig_two_threshold_*.png  — heatmaps per dataset"
echo "=========================================="
