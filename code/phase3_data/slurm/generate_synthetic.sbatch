#!/bin/bash
#SBATCH --job-name=synth-data
#SBATCH --output=/home/sural/code/logs/synthetic/%j.out
#SBATCH --error=/home/sural/code/logs/synthetic/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Synthetic French training data generation — Phase 2, Track A
# Generates 300 examples per HateCheck-FR functionality using Mistral-7B-Instruct.
#
# Functionalities:
#   slur_h           label=1  explicit slur
#   spell_leet_h     label=1  leet-speak obfuscation
#   spell_char_del_h label=1  character-deletion obfuscation
#   derog_impl_h     label=1  implicit derogation
#   counter_quote_nh label=0  counter-speech quoting hate
#
# Customise by passing --functionality <name> to generate a single category,
# or leave as 'all' to generate all five sequentially.
#
# Examples:
#   Generate only slur examples:
#     Edit FUNCTIONALITY below to "slur_h"
#   Generate 500 examples per category:
#     Edit N_SAMPLES below to 500

echo "=========================================="
echo "Synthetic Data Generation — Phase 2"
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

# Install langdetect if not already present
python -c "import langdetect" 2>/dev/null || pip install langdetect --quiet

# ── Configuration ─────────────────────────────────────────────────────────────
FUNCTIONALITY="all"          # "all" or one of: slur_h, spell_leet_h, spell_char_del_h, derog_impl_h, counter_quote_nh
N_SAMPLES=300                # examples to generate per functionality
BATCH_SIZE=10                # examples requested per LLM call
MODEL="mistral_7b_instruct"  # generator model

OUTPUT_DIR="$HOME/code/data/synthetic"
CACHE_DIR="$HOME/datasets/cache"
SCRIPT="$HOME/code/phase3_data/generate_synthetic_data.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/code/logs/synthetic"

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "Configuration:"
echo "  Functionality : $FUNCTIONALITY"
echo "  n_samples     : $N_SAMPLES"
echo "  batch_size    : $BATCH_SIZE"
echo "  model         : $MODEL"
echo "  output_dir    : $OUTPUT_DIR"
echo ""

python "$SCRIPT" \
    --functionality "$FUNCTIONALITY" \
    --model         "$MODEL"         \
    --n_samples     "$N_SAMPLES"     \
    --batch_size    "$BATCH_SIZE"    \
    --output_dir    "$OUTPUT_DIR"    \
    --cache_dir     "$CACHE_DIR"     \
    --device        cuda             \
    --seed          42

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Job completed : $(date)"
echo "Output:"
ls -lh "$OUTPUT_DIR"
echo "Line counts per file:"
wc -l "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (no JSONL files yet)"
echo "=========================================="
