#!/bin/bash
#SBATCH --job-name=eval-lora
#SBATCH --output=/home/sural/code/logs/eval_lora/%j.out
#SBATCH --error=/home/sural/code/logs/eval_lora/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Phase 2 evaluation — baseline vs LoRA on identical held-out test sets
#
# Must be submitted AFTER finetune_lora.sbatch has completed.
#
# Runs the following comparisons for each fine-tuning dataset
# (french_hate_superset, reddit_fr):
#
#   1. Baseline  (llama_guard_1b, shieldgemma_2b) on held-out test set
#   2. LoRA      (llama_guard_1b_lora, shieldgemma_2b_lora) on same test set
#
# Then runs a full 8-dataset eval for the LoRA models (no test set restriction),
# to see how fine-tuning generalises beyond the training distribution.
#
# Output layout:
#   results/phase2_eval/
#     french_hate_superset/
#       baseline/   ← base models on 20% held-out test set
#       lora/       ← LoRA models on same 20% held-out test set
#     reddit_fr/
#       baseline/
#       lora/
#     lora_full/    ← LoRA models on all 8 full datasets
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 2 Evaluation — Baseline vs LoRA"
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

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT="$HOME/code/phase2_lora/run_full_baseline_lora.py"
ADAPTER_ROOT="$HOME/code/results/lora_adapters"
RESULTS_ROOT="$HOME/code/results/phase2_eval"

OPENAI_PATH="$HOME/datasets/openai/samples-1680.jsonl"
TOXIGEN_PATH="$HOME/datasets/toxigen/toxigen_train.jsonl"
REDDIT_EN="$HOME/datasets/reddit/balanced/data-en/test-en.csv"
REDDIT_FR="$HOME/datasets/reddit/balanced/data-fr/test-fr.csv"
CACHE_DIR="$HOME/datasets/cache"

mkdir -p "$HOME/code/logs/eval_lora"

# ── Sanity checks (warn only — missing adapters are skipped, not fatal) ────────
for DATASET in french_hate_superset reddit_fr; do
    for MODEL in llama_guard_1b shieldgemma_2b; do
        ADAPTER="$ADAPTER_ROOT/$MODEL/$DATASET/best/adapter_config.json"
        if [ ! -f "$ADAPTER" ]; then
            echo "WARNING: adapter not found, will be skipped: $ADAPTER"
        fi
        TEST_SET="$ADAPTER_ROOT/$MODEL/$DATASET/test_set.json"
        if [ ! -f "$TEST_SET" ]; then
            echo "WARNING: test_set.json not found, will be skipped: $TEST_SET"
        fi
    done
done
echo "Sanity check done — missing adapters will be skipped."

# Common args passed to every python call
COMMON_ARGS="
    --cache_dir           $CACHE_DIR
    --openai_path         $OPENAI_PATH
    --toxigen_path        $TOXIGEN_PATH
    --reddit_en_path      $REDDIT_EN
    --reddit_fr_path      $REDDIT_FR
    --max_samples_toxigen 5000
    --max_samples_civil   5000
    --threshold           0.5
    --no_skip
"

# ── Loop over fine-tuning datasets ────────────────────────────────────────────
for ADAPTER_DATASET in french_hate_superset reddit_fr; do

    ADAPTER_LLAMA="$ADAPTER_ROOT/llama_guard_1b/${ADAPTER_DATASET}/best"
    ADAPTER_GEMMA="$ADAPTER_ROOT/shieldgemma_2b/${ADAPTER_DATASET}/best"
    TEST_SET="$ADAPTER_ROOT/llama_guard_1b/${ADAPTER_DATASET}/test_set.json"

    # Skip entire dataset block if llama adapter or test set is missing
    if [ ! -f "$ADAPTER_LLAMA/adapter_config.json" ] || [ ! -f "$TEST_SET" ]; then
        echo "SKIPPING $ADAPTER_DATASET — missing adapter or test_set.json"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Dataset: $ADAPTER_DATASET"
    echo "Test set: $TEST_SET"
    echo "=========================================="

    # ── 1. Baseline on held-out test set ──────────────────────────────────────
    OUT_BASELINE="$RESULTS_ROOT/${ADAPTER_DATASET}/baseline"
    mkdir -p "$OUT_BASELINE"
    echo ""
    echo "--- [1/2] Baseline (llama_guard_1b, shieldgemma_2b) on test set ---"

    python "$SCRIPT" \
        --output_dir    "$OUT_BASELINE"  \
        --test_set_path "$TEST_SET"      \
        --datasets      "$ADAPTER_DATASET" \
        --models        llama_guard_1b,shieldgemma_2b \
        $COMMON_ARGS

    echo "Baseline done: $(date)"

    # ── 2. LoRA on same held-out test set ─────────────────────────────────────
    OUT_LORA="$RESULTS_ROOT/${ADAPTER_DATASET}/lora"
    mkdir -p "$OUT_LORA"
    echo ""
    echo "--- [2/2] LoRA on same test set ---"

    # Build model list and adapter args depending on what's available
    LORA_MODELS="llama_guard_1b_lora"
    GEMMA_ARGS=""
    if [ -f "$ADAPTER_GEMMA/adapter_config.json" ]; then
        LORA_MODELS="llama_guard_1b_lora,shieldgemma_2b_lora"
        GEMMA_ARGS="--lora_adapter_gemma $ADAPTER_GEMMA"
    else
        echo "  NOTE: SG-2b adapter missing for $ADAPTER_DATASET — running LG-1B LoRA only"
    fi

    python "$SCRIPT" \
        --output_dir          "$OUT_LORA"        \
        --test_set_path       "$TEST_SET"         \
        --lora_adapter_llama  "$ADAPTER_LLAMA"    \
        $GEMMA_ARGS                               \
        --datasets            "$ADAPTER_DATASET"  \
        --models              $LORA_MODELS        \
        $COMMON_ARGS

    echo "LoRA done: $(date)"

done

# ── Full 8-dataset eval for LoRA models (french_hate_superset adapter) ────────
# Shows how fine-tuning on one dataset generalises to the others.
# Uses french_hate_superset adapters as primary; change ADAPTER_DATASET to reddit_fr
# and resubmit if you want the other adapter's generalisation profile.
echo ""
echo "=========================================="
echo "Full 8-dataset eval — LoRA (french_hate_superset adapter)"
echo "=========================================="

OUT_FULL="$RESULTS_ROOT/lora_full"
mkdir -p "$OUT_FULL"

python "$SCRIPT" \
    --output_dir          "$OUT_FULL"                                              \
    --lora_adapter_llama  "$ADAPTER_ROOT/llama_guard_1b/french_hate_superset/best" \
    --lora_adapter_gemma  "$ADAPTER_ROOT/shieldgemma_2b/french_hate_superset/best" \
    --datasets            all                                                      \
    --models              llama_guard_1b_lora,shieldgemma_2b_lora                  \
    $COMMON_ARGS

echo "Full eval done: $(date)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "All evaluations complete: $(date)"
echo ""
echo "Results layout:"
find "$RESULTS_ROOT" -name "summary.txt" | sort | while read f; do
    echo "  $f"
done
echo "=========================================="
