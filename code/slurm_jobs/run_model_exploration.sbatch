#!/bin/bash
#SBATCH --job-name=model-exploration
#SBATCH --output=/home/sural/code/logs/model_exploration/%j.out
#SBATCH --error=/home/sural/code/logs/model_exploration/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Job configuration for Model Exploration Notebook
# Author: Ural Seyfullah
# Purpose: Run comprehensive model testing and comparison via Jupyter notebook
# Fixed: Handles kernel specification issues

echo "=========================================="
echo "Model Exploration Job"
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
echo "=========================================="

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis 

# Verify Python and kernel
echo "Python version:"
which python
python --version

# Install/verify ipykernel in the thesis environment
echo "Setting up IPython kernel..."
python -m pip install ipykernel --quiet
python -m ipykernel install --user --name thesis --display-name "Python (thesis)"

# List available kernels
echo "Available Jupyter kernels:"
jupyter kernelspec list

# Set paths
NOTEBOOK_PATH="$HOME/code/model_experiments/model_exploration.ipynb"
OUTPUT_DIR="$HOME/code/results/model_exploration"
LOGS_DIR="$HOME/code/logs/model_exploration"

# Create necessary directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR

# Check if notebook exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "❌ ERROR: Notebook not found at $NOTEBOOK_PATH"
    echo "Please ensure the notebook is uploaded to the cluster"
    exit 1
fi

# Run the notebook with explicit kernel specification
echo "=========================================="
echo "Executing notebook: $NOTEBOOK_PATH"
echo "Using kernel: thesis"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Execute notebook with the thesis kernel
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=thesis \
    --ExecutePreprocessor.timeout=86400 \
    --allow-errors \
    --output-dir=$OUTPUT_DIR \
    --output=model_exploration_executed.ipynb \
    $NOTEBOOK_PATH

# Check execution status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Notebook executed successfully"
    echo "Output notebook saved to: $OUTPUT_DIR/model_exploration_executed.ipynb"
else
    echo "❌ Notebook execution failed with exit code $EXIT_CODE"
    echo "Check the error log for details: $LOGS_DIR/$SLURM_JOB_ID.err"
    
    # Try to show which cell failed
    if [ -f "$OUTPUT_DIR/model_exploration_executed.ipynb" ]; then
        echo "Partial output notebook was saved - check for execution errors"
    fi
    
    exit $EXIT_CODE
fi

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

# Print disk usage
echo "Disk usage in output directory:"
du -sh $OUTPUT_DIR

# Print summary of outputs
echo "Generated files:"
ls -lh $OUTPUT_DIR

echo "All done!"
