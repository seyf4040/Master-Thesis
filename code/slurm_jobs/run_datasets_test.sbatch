#!/bin/bash
#SBATCH --job-name=test-datasets
#SBATCH --output=/home/sural/code/logs/test_datasets/%j.out
#SBATCH --error=/home/sural/code/logs/test_datasets/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Job configuration for Dataset Evaluation
# Author: Ural Seyfullah
# Purpose: Test availability and access of datasets for content moderation
# Part of Master's Thesis: Deep Learning for Content Moderation on Shareish

echo "=========================================="
echo "Dataset Evaluation Job"
echo "=========================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# GPU check - FAIL IMMEDIATELY if no GPU
echo "Checking for GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "❌ ERROR: No GPU detected! Exiting immediately."
    echo "This job requires a GPU. Please check:"
    echo "  - #SBATCH --partition is set to a GPU partition"
    echo "  - #SBATCH --gres=gpu:1 is present"
    exit 1
fi

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis 

# Set paths
CACHE_DIR="$HOME/datasets/cache"
OPENAI_DATA_PATH="$HOME/datasets/openai/samples-1680.jsonl"
OUTPUT_DIR="$HOME/code/results/test_datasets"
SCRIPT_PATH="$HOME/code/dataset_experiments/test_datasets.py"

# Create necessary directories
echo "Creating directories..."
mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $HOME/code/logs
mkdir -p $(dirname $SCRIPT_PATH)

# Copy script to the correct location if it doesn't exist
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "⚠️  Script not found at $SCRIPT_PATH"
    echo "Please ensure test_datasets.py is in the correct location"
    exit 1
fi

# Run the evaluation script
echo "=========================================="
echo "Starting dataset evaluation..."
echo "=========================================="
echo "Cache directory: $CACHE_DIR"
echo "OpenAI data: $OPENAI_DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run evaluation
# Note: You can add --skip_datasets civil_comments toxigen to skip large datasets
python $SCRIPT_PATH \
    --cache_dir $CACHE_DIR \
    --openai_data_path $OPENAI_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --max_toxigen_samples 10000 \
    --max_civil_samples 10000 \
    --toxicity_threshold 0.5
#    --skip_datasets civil_comments  # Uncomment to skip specific datasets

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Print disk usage
echo "Disk usage in cache directory:"
du -sh $CACHE_DIR

echo "Disk usage in output directory:"
du -sh $OUTPUT_DIR

# Show result files if job succeeded
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
else
    echo ""
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
    echo "Check error log for details: ${SLURM_ERROR}"
fi

echo ""
echo "All done!"
