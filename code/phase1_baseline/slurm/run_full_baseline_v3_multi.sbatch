#!/bin/bash
#SBATCH --job-name=full-baseline-v3-multi
#SBATCH --output=/home/sural/code/logs/full_baseline/%j.out
#SBATCH --error=/home/sural/code/logs/full_baseline/%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Full baseline evaluation v3 — multiple runs for mean ± std statistics
# Aggregate afterward with:
#   python aggregate_runs.py --results_dir $OUTPUT_DIR
# Author: Ural Seyfullah

N_RUNS=3   # <-- change to run more or fewer times

echo "=========================================="
echo "Full Baseline Evaluation v3 — Multi-Run"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "N_RUNS      : $N_RUNS"
echo "=========================================="

if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. Exiting."
    exit 1
fi
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

OUTPUT_DIR="$HOME/code/results/full_baseline_v3"
SCRIPT="$HOME/code/phase1_baseline/run_full_baseline_v3.py"

OPENAI_PATH="$HOME/datasets/openai/samples-1680.jsonl"
TOXIGEN_PATH="$HOME/datasets/toxigen/toxigen_train.jsonl"
REDDIT_EN="$HOME/datasets/reddit/balanced/data-en/test-en.csv"
REDDIT_FR="$HOME/datasets/reddit/balanced/data-fr/test-fr.csv"
CACHE_DIR="$HOME/datasets/cache"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/code/logs/full_baseline"

for RUN_ID in $(seq 1 $N_RUNS); do
    echo ""
    echo "=========================================="
    echo "RUN $RUN_ID / $N_RUNS  —  $(date)"
    echo "=========================================="

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
        --threshold       0.5             \
        --run_id          $RUN_ID

    echo "Run $RUN_ID completed at $(date)"
done

echo ""
echo "=========================================="
echo "Aggregating results across $N_RUNS runs..."
echo "=========================================="

python "$HOME/code/shared/aggregate_runs.py" \
    --results_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "All $N_RUNS runs completed : $(date)"
echo "Output size : $(du -sh $OUTPUT_DIR)"
echo "=========================================="
