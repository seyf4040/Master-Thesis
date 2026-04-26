#!/bin/bash
#SBATCH --job-name=finetune-lora-joint-balanced
#SBATCH --output=/home/sural/code/logs/finetune_lora/%j.out
#SBATCH --error=/home/sural/code/logs/finetune_lora/%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Phase 2 LoRA fine-tuning — balanced joint dataset experiment
#
# Trains LG-1B and SG-2b on FHS + Reddit-FR, but with FHS training partition
# subsampled to match Reddit-FR training size (~3,665 samples each → ~1:1 ratio).
#
# Motivation: the unbalanced joint run (french_joint) showed FHS signal dominating
# gradient updates (FHS: 13k train vs Reddit-FR: 3.6k train → 3.5:1 ratio), causing
# Reddit-FR F1 to drop from 0.662 (single) to 0.632 (joint). Balancing the corpus
# eliminates this asymmetry.
#
# Test sets are IDENTICAL to french_joint and single-dataset runs (same seed=42 split,
# val/test not subsampled) — enabling direct 4-way comparison:
#   Baseline → Single LoRA → Joint LoRA → Joint Balanced LoRA
#
# After completion, evaluate with:
#   sbatch ~/code/phase2_lora/slurm/eval_lora.sbatch
#
# Expected runtime: ~2-3h (half the training data of french_joint)
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 2 LoRA Fine-tuning — Balanced Joint"
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

SCRIPT="$HOME/code/phase2_lora/finetune_lora.py"
ADAPTER_ROOT="$HOME/code/results/lora_adapters"
CACHE_DIR="$HOME/datasets/cache"
REDDIT_FR="$HOME/datasets/reddit/balanced/data-fr/test-fr.csv"

mkdir -p "$HOME/code/logs/finetune_lora"
mkdir -p "$ADAPTER_ROOT"

# ── Shared hyperparameters ────────────────────────────────────────────────────
# epochs=1: all single-dataset runs converged at epoch 1 (overfitting after)
# All other hyperparameters identical to french_joint for fair comparison.
LR=2e-4
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
MAX_LENGTH=512
VAL_FRACTION=0.1
SEED=42

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Run 1: Llama Guard 3 1B × french_joint_balanced ─────────────────────────
echo ""
echo "=========================================="
echo "Run 1/2: llama_guard_1b × french_joint_balanced"
echo "=========================================="
python "$SCRIPT" \
    --model          llama_guard_1b          \
    --dataset        french_joint_balanced   \
    --output_dir     "$ADAPTER_ROOT"         \
    --cache_dir      "$CACHE_DIR"            \
    --reddit_fr_path "$REDDIT_FR"            \
    --epochs         1                       \
    --lr             $LR                     \
    --lora_r         $LORA_R                 \
    --lora_alpha     $LORA_ALPHA             \
    --lora_dropout   $LORA_DROPOUT           \
    --batch_size     4                       \
    --grad_accum     4                       \
    --max_length     $MAX_LENGTH             \
    --val_fraction   $VAL_FRACTION           \
    --seed           $SEED

echo "Run 1 completed: $(date)"

# ── Run 2: ShieldGemma 2B × french_joint_balanced ────────────────────────────
echo ""
echo "=========================================="
echo "Run 2/2: shieldgemma_2b × french_joint_balanced"
echo "=========================================="
python "$SCRIPT" \
    --model                  shieldgemma_2b         \
    --dataset                french_joint_balanced  \
    --output_dir             "$ADAPTER_ROOT"        \
    --cache_dir              "$CACHE_DIR"           \
    --reddit_fr_path         "$REDDIT_FR"           \
    --epochs                 1                      \
    --lr                     $LR                    \
    --lora_r                 $LORA_R                \
    --lora_alpha             $LORA_ALPHA            \
    --lora_dropout           $LORA_DROPOUT          \
    --batch_size             2                      \
    --grad_accum             8                      \
    --max_length             $MAX_LENGTH            \
    --val_fraction           $VAL_FRACTION          \
    --seed                   $SEED                  \
    --gradient_checkpointing

echo "Run 2 completed: $(date)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Balanced joint fine-tuning complete: $(date)"
echo "Adapters saved to: $ADAPTER_ROOT"
echo ""
echo "Adapter layout:"
find "$ADAPTER_ROOT/llama_guard_1b/french_joint_balanced" \
     "$ADAPTER_ROOT/shieldgemma_2b/french_joint_balanced" \
    -name "adapter_config.json" 2>/dev/null | sort
echo ""
echo "Next step:"
echo "  sbatch ~/code/phase2_lora/slurm/eval_lora.sbatch"
echo "=========================================="
