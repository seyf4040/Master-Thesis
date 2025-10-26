#!/bin/bash
#SBATCH --job-name=llama_guard_toxigen
#SBATCH --output=/home/sural/code/logs/llama_guard_%j.out
#SBATCH --error=/home/sural/code/logs/llama_guard_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Job configuration for Llama Guard 3 inference on ToxiGen
# Author: Ural Seyfullah
# Purpose: First job submission to test pipeline and infrastructure

echo "=========================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis  

# Verify Python and packages
echo "Python version:"
python --version

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"

echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

# Set paths
DATA_PATH="$HOME/datasets/toxigen/toxigen_train.jsonl"
OUTPUT_DIR="$HOME/code/results/llama_guard_toxigen"
SCRIPT_PATH="$HOME/code/model_experiments/llama_guard_toxigen.py"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p $HOME/code/logs

# Run the inference script
echo "=========================================="
echo "Starting inference..."
echo "=========================================="

python $SCRIPT_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_name meta-llama/Llama-Guard-3-8B \
    --batch_size 8 \
    --max_samples 10

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

# Print disk usage
echo "Disk usage in output directory:"
du -sh $OUTPUT_DIR

# Copy results to home directory for persistence
echo "Copying results to home directory for backup..."
cp -r $OUTPUT_DIR/* $HOME/results_backup/llama_guard_$(date +%Y%m%d)/

echo "All done!"