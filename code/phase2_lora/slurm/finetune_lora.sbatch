#!/bin/bash
#SBATCH --job-name=finetune-lora
#SBATCH --output=/home/sural/code/logs/finetune_lora/%j.out
#SBATCH --error=/home/sural/code/logs/finetune_lora/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Phase 2 LoRA fine-tuning — Llama Guard 3 1B and ShieldGemma 2B
# Targets: French Hate Superset and Reddit-FR (biggest F1 gaps from Phase 1)
#
# After completion, evaluate with:
#   sbatch code/slurm_jobs/eval_lora.sbatch
# or locally:
#   python code/run_full_baseline_lora.py \
#       --output_dir ~/code/results/full_baseline_lora \
#       --lora_adapter_llama ~/code/results/lora_adapters/llama_guard_1b/french_hate_superset/best \
#       --lora_adapter_gemma ~/code/results/lora_adapters/shieldgemma_2b/french_hate_superset/best \
#       --datasets all --models llama_guard_1b_lora,shieldgemma_2b_lora --no_skip
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 2 LoRA Fine-tuning"
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
EPOCHS=3
LR=2e-4
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
BATCH_SIZE=4
GRAD_ACCUM=4          # effective batch = 4 × 4 = 16
MAX_LENGTH=512
VAL_FRACTION=0.1
SEED=42

# ── Run 1: Llama Guard 3 1B × French Hate Superset ───────────────────────────
echo ""
echo "=========================================="
echo "Run 1/4: llama_guard_1b × french_hate_superset"
echo "=========================================="
python "$SCRIPT" \
    --model          llama_guard_1b               \
    --dataset        french_hate_superset          \
    --output_dir     "$ADAPTER_ROOT"               \
    --cache_dir      "$CACHE_DIR"                  \
    --epochs         $EPOCHS                       \
    --lr             $LR                           \
    --lora_r         $LORA_R                       \
    --lora_alpha     $LORA_ALPHA                   \
    --lora_dropout   $LORA_DROPOUT                 \
    --batch_size     $BATCH_SIZE                   \
    --grad_accum     $GRAD_ACCUM                   \
    --max_length     $MAX_LENGTH                   \
    --val_fraction   $VAL_FRACTION                 \
    --seed           $SEED

echo "Run 1 completed: $(date)"

# ── Run 2: Llama Guard 3 1B × Reddit-FR ──────────────────────────────────────
echo ""
echo "=========================================="
echo "Run 2/4: llama_guard_1b × reddit_fr"
echo "=========================================="
python "$SCRIPT" \
    --model          llama_guard_1b               \
    --dataset        reddit_fr                     \
    --output_dir     "$ADAPTER_ROOT"               \
    --cache_dir      "$CACHE_DIR"                  \
    --reddit_fr_path "$REDDIT_FR"                  \
    --epochs         $EPOCHS                       \
    --lr             $LR                           \
    --lora_r         $LORA_R                       \
    --lora_alpha     $LORA_ALPHA                   \
    --lora_dropout   $LORA_DROPOUT                 \
    --batch_size     $BATCH_SIZE                   \
    --grad_accum     $GRAD_ACCUM                   \
    --max_length     $MAX_LENGTH                   \
    --val_fraction   $VAL_FRACTION                 \
    --seed           $SEED

echo "Run 2 completed: $(date)"

# ── Run 3: ShieldGemma 2B × French Hate Superset ─────────────────────────────
echo ""
echo "=========================================="
echo "Run 3/4: shieldgemma_2b × french_hate_superset"
echo "=========================================="
python "$SCRIPT" \
    --model          shieldgemma_2b                \
    --dataset        french_hate_superset           \
    --output_dir     "$ADAPTER_ROOT"                \
    --cache_dir      "$CACHE_DIR"                   \
    --epochs         $EPOCHS                        \
    --lr             $LR                            \
    --lora_r         $LORA_R                        \
    --lora_alpha     $LORA_ALPHA                    \
    --lora_dropout   $LORA_DROPOUT                  \
    --batch_size     $BATCH_SIZE                    \
    --grad_accum     $GRAD_ACCUM                    \
    --max_length     $MAX_LENGTH                    \
    --val_fraction   $VAL_FRACTION                  \
    --seed           $SEED

echo "Run 3 completed: $(date)"

# ── Run 4: ShieldGemma 2B × Reddit-FR ────────────────────────────────────────
echo ""
echo "=========================================="
echo "Run 4/4: shieldgemma_2b × reddit_fr"
echo "=========================================="
python "$SCRIPT" \
    --model          shieldgemma_2b                \
    --dataset        reddit_fr                      \
    --output_dir     "$ADAPTER_ROOT"                \
    --cache_dir      "$CACHE_DIR"                   \
    --reddit_fr_path "$REDDIT_FR"                   \
    --epochs         $EPOCHS                        \
    --lr             $LR                            \
    --lora_r         $LORA_R                        \
    --lora_alpha     $LORA_ALPHA                    \
    --lora_dropout   $LORA_DROPOUT                  \
    --batch_size     $BATCH_SIZE                    \
    --grad_accum     $GRAD_ACCUM                    \
    --max_length     $MAX_LENGTH                    \
    --val_fraction   $VAL_FRACTION                  \
    --seed           $SEED

echo "Run 4 completed: $(date)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "All fine-tuning runs complete: $(date)"
echo "Adapters saved to: $ADAPTER_ROOT"
echo ""
echo "Adapter layout:"
find "$ADAPTER_ROOT" -name "adapter_config.json" | sort
echo ""
echo "Next step: run eval_lora.sbatch or run_full_baseline_lora.py"
echo "=========================================="
