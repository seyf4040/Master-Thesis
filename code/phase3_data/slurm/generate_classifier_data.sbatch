#!/bin/bash
#SBATCH --job-name=synth-classifier
#SBATCH --output=/home/sural/code/logs/synthetic_classifier/%j.out
#SBATCH --error=/home/sural/code/logs/synthetic_classifier/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Synthetic classifier data generation — Phase 3, Track B
# Generates French solidarity_exchange and commercial_listing examples
# for the Shareish content-type classifier.
#
# Priority: commercial_listing (label=0) is critically underrepresented
# (only 51 real examples from 2ememain.be). Solidarity is secondary.
#
# Estimated time: ~1h per class at 300 samples (Mistral-7B on A5000)
#
# To generate commercial only:
#   Change CLASSES to "commercial_listing"

echo "=========================================="
echo "Synthetic Classifier Data — Phase 3"
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

python -c "import langdetect" 2>/dev/null || pip install langdetect --quiet

# ── Configuration ─────────────────────────────────────────────────────────────
CLASSES="all"           # "all", "solidarity_exchange", or "commercial_listing"
N_SAMPLES=300
BATCH_SIZE=8
MODEL="mistral_7b_instruct"

OUTPUT_DIR="$HOME/code/data/synthetic_classifier"
CACHE_DIR="$HOME/datasets/cache"
SCRIPT="$HOME/code/phase3_data/generate_classifier_data.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/code/logs/synthetic_classifier"

echo ""
echo "Configuration:"
echo "  Classes    : $CLASSES"
echo "  n_samples  : $N_SAMPLES"
echo "  batch_size : $BATCH_SIZE"
echo "  model      : $MODEL"
echo "  output_dir : $OUTPUT_DIR"
echo ""

python "$SCRIPT" \
    --classes    "$CLASSES"    \
    --model      "$MODEL"      \
    --n_samples  "$N_SAMPLES"  \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir  "$CACHE_DIR"  \
    --device     cuda          \
    --seed       42

echo ""
echo "=========================================="
echo "Job completed : $(date)"
echo "Items collected:"
wc -l "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (no output files)"
echo "=========================================="
