#!/bin/bash
#SBATCH --job-name=full-baseline
#SBATCH --output=/home/sural/code/logs/full_baseline/%j.out
#SBATCH --error=/home/sural/code/logs/full_baseline/%j.err
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Full baseline evaluation: all 10 models × all available datasets
# Author: Ural Seyfullah
#
# Customise by passing --datasets / --models to the python script below.
# Examples:
#   Fast models only:   --models detoxify_multilingual,detoxify_unbiased,koalaai,ethicaleye,citizenlab
#   French datasets:    --datasets hatecheck_fr,french_hate_superset,reddit_fr
#   Quick smoke test:   --max_samples 50

echo "=========================================="
echo "Full Baseline Evaluation"
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
OUTPUT_DIR="$HOME/code/results/full_baseline"
SCRIPT="$HOME/code/run_full_baseline.py"

OPENAI_PATH="$HOME/datasets/openai/samples-1680.jsonl"
TOXIGEN_PATH="$HOME/datasets/toxigen/toxigen_train.jsonl"
REDDIT_EN="$HOME/datasets/reddit/balanced/data-en/test-en.csv"
REDDIT_FR="$HOME/datasets/reddit/balanced/data-fr/test-fr.csv"
CACHE_DIR="$HOME/datasets/cache"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/code/logs/full_baseline"

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "Starting evaluation..."
echo ""

python "$SCRIPT" \
    --output_dir      "$OUTPUT_DIR"   \
    --cache_dir       "$CACHE_DIR"    \
    --openai_path     "$OPENAI_PATH"  \
    --toxigen_path    "$TOXIGEN_PATH" \
    --reddit_en_path  "$REDDIT_EN"    \
    --reddit_fr_path  "$REDDIT_FR"    \
    --datasets        all             \
    --models          all             \
    --max_samples_toxigen 5000        \
    --max_samples_civil   5000        \
    --threshold       0.5

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Job completed : $(date)"
echo "Output size   : $(du -sh $OUTPUT_DIR)"
echo "=========================================="
