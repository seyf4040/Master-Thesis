#!/bin/bash
#SBATCH --job-name=test-models
#SBATCH --output=/home/sural/code/logs/test_models/%j.out
#SBATCH --error=/home/sural/code/logs/test_models/%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Job configuration for HuggingFace model testing
# Author: Ural Seyfullah
# Purpose: Test availability and access of HuggingFace Transformers library models

echo "=========================================="
echo "Models Evaluation Job"
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
DATA_PATH="$HOME/datasets/openai/samples-1680.jsonl"
OUTPUT_DIR="$HOME/code/results/test_models"
SCRIPT_PATH="$HOME/code/model_experiments/test_hf_models.py"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p $HOME/code/logs

# Run the inference script
echo "=========================================="
echo "Starting tests..."
echo "=========================================="

# Run test
python $SCRIPT_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
#    --max_samples 10

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

# Print disk usage
echo "Disk usage in output directory:"
du -sh $OUTPUT_DIR

echo "All done!" 