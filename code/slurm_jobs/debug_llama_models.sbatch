#!/bin/bash
#SBATCH --cpus-per-task=1        # Number of CPU cores to allocate
#SBATCH --export=ALL
#SBATCH --job-name "JUPYTER"
#SBATCH --mem-per-cpu=4000       # Memory to allocate in MB per allocated CPU core
#SBATCH --output "jupyter.log"   # Log-file (important!)
#SBATCH --gres=gpu:0             # Number of GPU's
#SBATCH --time="7-00:00:00"      # Max execution time

# Activate conda environment
# echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis 

# Start Jupyter Lab
jupyter lab --ip='*' --no-browser